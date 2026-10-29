"""
Fine-tune Cosmos Reason (LoRA) on digital-twin vial images and save the adapter
(.safetensors). Prompts and answers are built from the labels + recipe table.

LABELS FILE (CSV or JSONL), one row per rendered image. Columns:
  image            path to the image (absolute, or relative to --img-root)
  recipe_id        5..14, the recipe this vial was dosed with
  prev_recipe_id   recipe that ran just before (blank/0 if none)
  vial_in_batch    1, 2 or 3 (position within the recipe's batch)
  contaminated     1 if the twin rendered carryover discoloration, else 0
  observed_color   optional, e.g. "green" (what the contaminated vial looks like)

If your data generator names columns differently, edit load_rows() only.

Run (inside the NVIDIA PyTorch container on the Spark):
  pip install transformers==4.51.3 qwen-vl-utils peft accelerate pillow
  python train_cosmos_lora.py --labels /workspace/data/labels.csv \
      --img-root /workspace/data --holdout 7 --epochs 2

Add --push (with HF_TOKEN set in the environment) to fix the base-model id in the
saved adapter and upload it to Hugging Face under adapters/<name of the --out folder>.
Give every run its own --out folder, e.g. --out /workspace/models/run-01.

TRAIN ON TWIN DATA ONLY. Real data is for testing.
"""
import argparse, csv, json, os, random, re
import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

# ---- Recipe table (from the dataset README): mL per tank ----
RECIPES = {
    5: {100: 10, 200: 10, 400: 10}, 6: {100: 30, 200: 10},
    7: {100: 20, 400: 50},          8: {100: 20, 200: 20, 400: 20},
    9: {100: 10, 200: 35, 400: 35}, 10: {100: 45, 200: 15, 400: 15},
    11: {100: 10, 200: 10, 400: 50}, 12: {100: 10, 200: 50, 400: 10},
    13: {200: 50, 400: 20},         14: {100: 60, 400: 10},
}
TANK = {100: "Tank 100", 200: "Tank 200", 400: "Tank 400"}
ORDER = [200, 400, 100]  # fixed dispense order


def describe(rid):
    parts = [f"{TANK[t]} {RECIPES[rid][t]} mL" for t in ORDER if t in RECIPES[rid]]
    return f"Recipe {rid}: " + ", then ".join(parts) + f" (total {sum(RECIPES[rid].values())} mL)"


def build_example(r):
    rid = int(r["recipe_id"])
    prev = int(r.get("prev_recipe_id") or 0)
    k = int(r.get("vial_in_batch") or 0)
    prev_txt = describe(prev) if prev in RECIPES else "none (start of production run)"
    
    prompt = (
        f"Station 3 Inspection. Current {describe(rid)}. Previous {prev_txt}. "
        f"This is vial {k} of the batch. Inspect the vial image: Does the liquid match "
        f"Recipe {rid}? If contaminated, output machine control action STOP_MACHINE, identify "
        f"the vial, and detail the chemical root cause from the previous recipe."
    )
    
    if int(r.get("contaminated") or 0) and prev in RECIPES:
        obs = (r.get("observed_color") or "discolored").strip()
        # Find which tank ingredients in prev recipe caused the issue
        prev_tanks = [TANK[t] for t in ORDER if t in RECIPES[prev] and t != 100]
        curr_tanks = [TANK[t] for t in ORDER if t in RECIPES[rid] and t != 100]
        
        answer = (
            f"[ACTION: STOP_MACHINE]\n"
            f"ERROR: Fluid Contamination Detected.\n"
            f"- Recipe Being Filled: Recipe {rid} ({describe(rid).split(': ', 1)[1]})\n"
            f"- Batch Position: Vial {k} of 3\n"
            f"- Observed Appearance: {obs} (MISMATCH: does not match standard Recipe {rid})\n"
            f"- Root Cause: Carryover from previous Recipe {prev} ({describe(prev).split(': ', 1)[1]}). "
            f"Residual fluid from {', '.join(prev_tanks)} remained in the shared dispensing nozzle/manifold "
            f"and cross-contaminated the incoming fluid from {', '.join(curr_tanks)}.\n"
            f"- Machine Directive: Line halted. Initiate nozzle flush cycle before resuming."
        )
    else:
        answer = (
            f"[ACTION: CONTINUE]\n"
            f"STATUS: Normal.\n"
            f"- Recipe Being Filled: Recipe {rid}\n"
            f"- Batch Position: Vial {k} of 3\n"
            f"- Verification: Liquid color matches Recipe {rid} specification. No carryover detected."
        )
    return prompt, answer

def load_rows(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path) if l.strip()]
    return list(csv.DictReader(open(path, newline="")))


ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="")
ap.add_argument("--smoke", action="store_true", help="make dummy images+labels and use them (pipeline test)")
ap.add_argument("--dry-run", action="store_true", help="print example prompts/answers and exit (no model, no GPU)")
ap.add_argument("--push", action="store_true",
                help="after training, fix the base-model id and upload the adapter to Hugging Face (needs HF_TOKEN)")
ap.add_argument("--hf-repo", default="PourSightAI/CosmosReason3Hackathon", help="Hugging Face repo to upload to")
ap.add_argument("--hf-base-id", default="nvidia/Cosmos-Reason1-7B",
                help="real Hugging Face id of the base model (replaces the local path in the adapter files)")
ap.add_argument("--img-root", default="")
ap.add_argument("--base", default="/workspace/models/cosmos-reason")
ap.add_argument("--out", default="/workspace/models/cosmos-lora")
ap.add_argument("--holdout", default="", help="comma-separated recipe ids to leave out, e.g. 7,9")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8)
args = ap.parse_args()


def make_smoke(root="/workspace/data/smoke"):
    """Dummy data in the real labels format. Vial 1 of each batch is 'contaminated' (green)."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "labels.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "recipe_id", "prev_recipe_id", "vial_in_batch", "contaminated", "observed_color"])
        n = 0
        for rid in (6, 7, 8):
            for k in (1, 2, 3):
                dirty = k == 1
                img = Image.new("RGB", (640, 480), (200, 200, 200))
                ImageDraw.Draw(img).rectangle([270, 120, 370, 400],
                                              fill=(40, 180, 90) if dirty else (40, 80, 220))
                fn = f"{n:03d}.png"
                img.save(os.path.join(root, fn))
                w.writerow([fn, rid, rid - 1, k, int(dirty), "green" if dirty else ""])
                n += 1
    return path, root


if args.smoke:
    args.labels, args.img_root = make_smoke()
    args.out, args.epochs, args.accum = "/workspace/models/smoke-lora", 1, 2
    args.holdout = args.holdout or "7"
if not args.labels:
    raise SystemExit("Give --labels <file>, or use --smoke to test with dummy data.")

rows = load_rows(args.labels)

if args.dry_run:  # check the label -> prompt/answer conversion without loading the model
    shown = set()
    for r in rows:
        key = int(r.get("contaminated") or 0)
        if key not in shown:
            shown.add(key)
            pr, an = build_example(r)
            print(f"\n--- contaminated={key} ---\nPROMPT: {pr}\nANSWER: {an}")
    raise SystemExit(f"\nDry run OK: {len(rows)} rows read.")
held = {int(x) for x in args.holdout.split(",") if x.strip()}
train_rows = [r for r in rows if int(r["recipe_id"]) not in held]
held_rows = [r for r in rows if int(r["recipe_id"]) in held]
print(f"{len(train_rows)} train rows, {len(held_rows)} held out (recipes {sorted(held)})")
os.makedirs(args.out, exist_ok=True)
if held_rows:  # so the evaluation script knows what was never seen
    with open(os.path.join(args.out, "heldout.json"), "w") as f:
        json.dump(held_rows, f)

processor = AutoProcessor.from_pretrained(args.base)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.base, torch_dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()


def encode(r):
    prompt_txt, answer = build_example(r)
    img = Image.open(os.path.join(args.img_root, r["image"])).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_txt}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    full = prompt + answer + processor.tokenizer.eos_token
    pi = processor(text=[prompt], images=[img], return_tensors="pt")
    fi = processor(text=[full], images=[img], return_tensors="pt")
    labels = fi["input_ids"].clone()
    labels[:, : pi["input_ids"].shape[1]] = -100  # learn only the answer
    fi["labels"] = labels
    return {k: v.to("cuda") for k, v in fi.items()}


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
model.train()
for epoch in range(args.epochs):
    random.shuffle(train_rows)
    for i, r in enumerate(train_rows):
        loss = model(**encode(r)).loss
        (loss / args.accum).backward()
        if (i + 1) % args.accum == 0 or i == len(train_rows) - 1:
            opt.step(); opt.zero_grad()
        if i % 10 == 0:
            print(f"epoch {epoch} {i}/{len(train_rows)} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(args.out)  # writes adapter_model.safetensors; checkpoint each epoch
    print(f"saved adapter to {args.out}", flush=True)


def publish(folder, repo_id, base_id):
    """Make the adapter portable (real base-model id instead of a local path), then upload it."""
    readme = os.path.join(folder, "README.md")
    if os.path.exists(readme):
        with open(readme) as f:
            text = f.read()
        text = re.sub(r"(?m)^base_model:.*$", f"base_model: {base_id}", text)
        text = re.sub(r"(?m)^- base_model:adapter:.*$", f"- base_model:adapter:{base_id}", text)
        with open(readme, "w") as f:
            f.write(text)
    cfg = os.path.join(folder, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            conf = json.load(f)
        conf["base_model_name_or_path"] = base_id
        with open(cfg, "w") as f:
            json.dump(conf, f, indent=2)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set: adapter is saved locally but was NOT uploaded.")
        return
    from huggingface_hub import HfApi
    dest = "adapters/" + os.path.basename(folder.rstrip("/"))
    try:
        HfApi().upload_folder(repo_id=repo_id, folder_path=folder, path_in_repo=dest)
        print(f"uploaded to {repo_id} in folder {dest}")
    except Exception as e:  # keep the trained adapter even if the upload fails
        print(f"Upload failed: {e}\nThe adapter is still saved at {folder}.")


if args.push:
    publish(args.out, args.hf_repo, args.hf_base_id)
print("done")