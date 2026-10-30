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
from pathlib import Path
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
TANK = {100: "T-100 clear", 200: "T-200 orange", 400: "T-400 blue"}
ORDER = [200, 400, 100]  # fixed dispense order


def describe(rid):
    parts = [f"{TANK[t]} {RECIPES[rid][t]} mL" for t in ORDER if t in RECIPES[rid]]
    return f"Recipe {rid}: " + ", then ".join(parts) + f" (total {sum(RECIPES[rid].values())} mL)"


def build_example(r):
    rid = int(r["recipe_id"])
    prev = int(r.get("prev_recipe_id") or 0)
    k = int(r.get("vial_in_batch") or 0)
    prev_txt = describe(prev) if prev in RECIPES else "none (first fill of the run)"
    prompt = (f"Current {describe(rid)}. Previous {prev_txt}. "
              f"This is vial {k} of the batch. Does the liquid color match the current recipe? "
              f"If not, explain the likely cause.")
    if r["contaminated"] == 1:
        if prev not in RECIPES:
            raise ValueError(f"Contaminated vial for Recipe {rid} has no valid previous recipe")
        obs = (r.get("observed_color") or "discolored").strip()
        answer = (f"No. The liquid looks {obs}, not the expected color for Recipe {rid}. "
                  f"Likely carryover from Recipe {prev}, which dispensed "
                  f"{describe(prev).split(': ', 1)[1]}. Its residue stayed in the shared "
                  f"mixing path and entered the first vial of the new batch.")
    else:
        answer = f"Yes. The liquid color matches Recipe {rid}. No sign of carryover."
    return prompt, answer


REQUIRED_COLUMNS = {"image", "recipe_id", "prev_recipe_id", "vial_in_batch", "contaminated"}


def load_rows(path):
    """Load and normalize the simulator's labels.csv (or JSONL) safely."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Labels file not found: {path}")

    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            # Strip whitespace/BOM from headers and values.
            reader.fieldnames = [h.strip().lstrip("\ufeff") if h else h for h in reader.fieldnames]
            rows = [{(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items()} for row in reader]

    if not rows:
        raise ValueError(f"No data rows found in {path}")
    columns = set(rows[0])
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError("Labels file is missing required column(s): " + ", ".join(missing)
                         + ". Expected the simulator labels.csv format.")

    normalized = []
    errors = []
    for line_no, row in enumerate(rows, start=2):
        try:
            image = str(row.get("image", "")).strip()
            if not image:
                raise ValueError("image path is blank")
            rid = int(row["recipe_id"])
            prev_raw = row.get("prev_recipe_id", "")
            prev = int(prev_raw) if str(prev_raw or "").strip() else 0
            vial = int(row["vial_in_batch"])
            contaminated_raw = str(row["contaminated"]).strip().lower()
            if contaminated_raw in {"1", "true", "yes"}:
                contaminated = 1
            elif contaminated_raw in {"0", "false", "no"}:
                contaminated = 0
            else:
                raise ValueError(f"invalid contaminated value {contaminated_raw!r}; use 0 or 1")
            if rid not in RECIPES:
                raise ValueError(f"unknown recipe_id {rid}; known recipes: {sorted(RECIPES)}")
            if prev and prev not in RECIPES:
                raise ValueError(f"unknown prev_recipe_id {prev}")
            if vial not in (1, 2, 3):
                raise ValueError(f"vial_in_batch must be 1, 2, or 3 (got {vial})")
            if contaminated and not prev:
                raise ValueError("contaminated=1 but prev_recipe_id is blank/0")
            row["image"] = image
            row["recipe_id"] = rid
            row["prev_recipe_id"] = prev
            row["vial_in_batch"] = vial
            row["contaminated"] = contaminated
            row["observed_color"] = str(row.get("observed_color") or "").strip()
            normalized.append(row)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"row {line_no}: {exc}")

    if errors:
        preview = "\n  ".join(errors[:10])
        suffix = f"\n  ... and {len(errors)-10} more" if len(errors) > 10 else ""
        raise ValueError(f"Found {len(errors)} invalid label row(s):\n  {preview}{suffix}")
    return normalized


def resolve_image_path(image_value, img_root):
    """Support absolute paths and paths such as images/run000_v01_r5.png."""
    image_path = Path(image_value).expanduser()
    if image_path.is_absolute():
        candidate = image_path
    else:
        if not img_root:
            raise ValueError("--img-root is required when CSV image paths are relative")
        candidate = Path(img_root).expanduser() / image_path
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Image referenced by labels CSV does not exist: {candidate}")
    return candidate


ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="")
ap.add_argument("--smoke", action="store_true", help="make dummy images+labels and use them (pipeline test)")
ap.add_argument("--dry-run", action="store_true", help="print example prompts/answers and exit (no model, no GPU)")
ap.add_argument("--push", action="store_true",
                help="after training, fix the base-model id and upload the adapter to Hugging Face (needs HF_TOKEN)")
ap.add_argument("--hf-repo", default="PourSightAI/CosmosReason3Hackathon", help="Hugging Face repo to upload to")
ap.add_argument("--hf-base-id", default="nvidia/Cosmos-Reason1-7B",
                help="real Hugging Face id of the base model (replaces the local path in the adapter files)")
ap.add_argument("--img-root", default="", help="dataset folder containing the images/ directory")
ap.add_argument("--holdout-runs", default="", help="comma-separated run_id values to hold out (recommended to avoid image leakage)")
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
clean_count = sum(r["contaminated"] == 0 for r in rows)
contaminated_count = len(rows) - clean_count
print(f"Loaded {len(rows)} valid rows: {clean_count} clean, {contaminated_count} contaminated")

if args.dry_run:  # check CSV, image paths, and prompt/answer conversion without loading the model
    if args.img_root:
        missing_images = []
        for row in rows:
            try:
                resolve_image_path(row["image"], args.img_root)
            except FileNotFoundError as exc:
                missing_images.append(str(exc))
        if missing_images:
            raise FileNotFoundError("Missing image(s) referenced by CSV:\n" + "\n".join(missing_images[:10]))
    shown = set()
    for r in rows:
        key = int(r.get("contaminated") or 0)
        if key not in shown:
            shown.add(key)
            pr, an = build_example(r)
            print(f"\n--- contaminated={key} ---\nPROMPT: {pr}\nANSWER: {an}")
    raise SystemExit(f"\nDry run OK: {len(rows)} rows read.")
# Prefer holding out complete production runs: neighboring vials from one run
# are highly similar, so splitting individual images would leak near-duplicates.
held_runs = {int(x.strip()) for x in args.holdout_runs.split(",") if x.strip()}
held_recipes = {int(x.strip()) for x in args.holdout.split(",") if x.strip()}
if held_runs and "run_id" not in rows[0]:
    raise ValueError("--holdout-runs requires a run_id column in labels.csv")
if held_runs:
    train_rows = [r for r in rows if int(r["run_id"]) not in held_runs]
    held_rows = [r for r in rows if int(r["run_id"]) in held_runs]
    split_description = f"runs {sorted(held_runs)}"
elif held_recipes:
    train_rows = [r for r in rows if r["recipe_id"] not in held_recipes]
    held_rows = [r for r in rows if r["recipe_id"] in held_recipes]
    split_description = f"recipes {sorted(held_recipes)}"
else:
    train_rows, held_rows = rows, []
    split_description = "no holdout"
if not train_rows:
    raise ValueError("Training split is empty; change --holdout or --holdout-runs")
if not args.img_root:
    raise ValueError("Pass --img-root pointing to the dataset folder, e.g. the folder containing images/")
print(f"{len(train_rows)} train rows, {len(held_rows)} held out ({split_description})")
# Fail before loading the large model if a CSV image path is wrong.
for row in rows:
    resolve_image_path(row["image"], args.img_root)
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
    image_path = resolve_image_path(r["image"], args.img_root)
    with Image.open(image_path) as source_image:
        img = source_image.convert("RGB")
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