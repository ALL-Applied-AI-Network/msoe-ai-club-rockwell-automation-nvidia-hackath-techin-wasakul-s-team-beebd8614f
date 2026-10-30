#!/usr/bin/env python3
"""Continued LoRA fine-tuning on the additional generated dataset.

Uses ONLY generated full_dataset_5000 images and plan_5000.csv.
The target is the first vial of a current three-vial batch. Inputs contain
9 earlier chronological images plus all 3 current-batch images.
"""

import argparse
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

REPO = "PourSightAI/CosmosReason3Hackathon"
SUBFOLDER = "cosmos3-nano"
DEFAULT_DATA = Path("/workspace/twin_data/full_dataset_5000")

SYSTEM_PROMPT = """You assess possible liquid carryover in an industrial vial-filling process.
You receive chronological visual history followed by images from a current
recipe batch. Assess ONLY the first vial in the current batch. The other current
batch images are reference evidence, not additional contamination targets.

Do not assume a recipe identity, expected color, tank identity, ingredient,
dispense order, or exact liquid volume. Do not infer hidden simulator metadata.
Recipes can legitimately differ and can share colors or ingredients. Compare
large color differences carefully. When colors are similar, apparent fill-level
differences become especially useful; do not let small fill differences outweigh
a large clear color difference. A small approximately constant pipe residual may
have a greater proportional effect in a smaller-volume recipe.

The first vial is a useful carryover candidate, not automatically contaminated.
Compare it with the earlier visual history and with both later current-batch vials.
Agreement between the later two alone does not clear vial 1: vial 1 itself must
visually match both in color and fill level before that evidence can refute suspicion.
Focus on strong visual evidence; ambiguous cases should remain uncertain.

Return only this JSON object, with no markdown:
{"assessment":"potential_carryover"}
or
{"assessment":"no_clear_carryover_signal"}
"""


def resolve_image(image_value: str, data_root: Path) -> Path:
    p = Path(image_value)
    candidates = [p] if p.is_absolute() else [
        data_root / p,
        data_root / "images" / p.name,
        data_root.parent / p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find image {image_value!r}; tried: "
        + ", ".join(str(x) for x in candidates)
    )


def load_examples(data_root: Path):
    csv_path = data_root / "plan_5000.csv"
    if not csv_path.is_file():
        csv_path = data_root.parent / "plan_5000.csv"
    image_root = data_root / "images"
    if not csv_path.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            f"Expected {csv_path} and image directory {image_root}. "
            "This script is intentionally restricted to full_dataset_5000."
        )

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"run_id", "vial_index", "vial_in_batch", "image",
                "prev_recipe_id", "recipe_id", "contaminated"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"plan_5000.csv missing columns: {sorted(missing)}")

    runs = defaultdict(list)
    for row in rows:
        runs[str(row["run_id"])].append(row)
    for seq in runs.values():
        seq.sort(key=lambda r: int(r["vial_index"]))

    examples = []
    for run_id, seq in runs.items():
        for end in range(11, len(seq)):
            current = seq[end - 2:end + 1]
            history = seq[end - 11:end - 2]
            if len(current) != 3 or len(history) != 9:
                continue
            if [int(r["vial_in_batch"]) for r in current] != [1, 2, 3]:
                continue
            if [int(r["vial_in_batch"]) for r in history] != [1, 2, 3] * 3:
                continue
            ci = [int(r["vial_index"]) for r in current]
            hi = [int(r["vial_index"]) for r in history]
            if ci != list(range(ci[0], ci[0] + 3)):
                continue
            if hi != list(range(hi[0], hi[0] + 9)) or ci[0] != hi[-1] + 1:
                continue
            # Metadata is used only to select and label examples; none is sent to the model.
            if str(current[0]["prev_recipe_id"]).strip() in ("", "0"):
                continue
            if current[0]["recipe_id"] == current[0]["prev_recipe_id"]:
                continue
            label = int(current[0]["contaminated"])
            if label not in (0, 1):
                continue
            paths = [resolve_image(r["image"], data_root) for r in history + current]
            examples.append({"run": run_id, "label": label, "paths": paths,
                             "vial_index": ci[0]})
    if not examples:
        raise RuntimeError("No eligible 12-image transition examples were found.")
    return examples


def split_by_run(examples, validation_fraction, seed):
    by_run = defaultdict(list)
    for ex in examples:
        by_run[ex["run"]].append(ex)
    run_ids = sorted(by_run, key=str)
    rng = random.Random(seed)
    rng.shuffle(run_ids)
    n_val = max(1, round(len(run_ids) * validation_fraction)) if len(run_ids) > 1 else 0
    val_runs = set(run_ids[:n_val])
    train = [e for e in examples if e["run"] not in val_runs]
    val = [e for e in examples if e["run"] in val_runs]
    if not train or not val:
        raise RuntimeError("Need at least two distinct runs for a run-level train/validation split.")
    return train, val, sorted(val_runs, key=str)


def make_messages(ex, data_root):
    content = [{"type": "text", "text": SYSTEM_PROMPT}]

    # Earlier images are supplied in chronological order without revealing
    # recipe boundaries or the number of earlier batches.
    for i in range(9):
        path = ex["paths"][i]
        with Image.open(path) as im:
            image = im.convert("RGB")
        content.append({"type": "text", "text": f"Earlier chronological image {i + 1}"})
        content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": "CURRENT RECIPE BATCH"})
    for i in range(3):
        path = ex["paths"][9 + i]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if i == 0:
            label = "First current-batch vial: candidate for assessment"
        else:
            label = "Additional current-batch vial: reference evidence"
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": image})

    return [{"role": "user", "content": content}]

def encode_example(ex, processor, device):
    user_messages = make_messages(ex, None)
    target = (
        '{"assessment":"potential_carryover"}' if ex["label"] == 1
        else '{"assessment":"no_clear_carryover_signal"}'
    )
    full_messages = user_messages + [{"role": "assistant", "content": target}]
    prompt_inputs = processor.apply_chat_template(
        user_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    full_inputs = processor.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt"
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    # Do not train on padding if the processor adds any.
    if "attention_mask" in full_inputs:
        labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    return {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in full_inputs.items()}



def evaluate_classification(model, processor, examples, device):
    """Greedy held-out classification; uses no hidden labels in model inputs."""
    model.eval()
    predictions = []
    tp = fp = tn = fn = 0
    unparsed = 0

    for i, ex in enumerate(examples, 1):
        messages = make_messages(ex, None)
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=32, do_sample=False
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated = processor.batch_decode(
            output[:, prompt_length:], skip_special_tokens=True
        )[0].strip()

        if "potential_carryover" in generated:
            pred = 1
        elif "no_clear_carryover_signal" in generated:
            pred = 0
        else:
            pred = None
            unparsed += 1

        actual = int(ex["label"])
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        elif pred == 0 and actual == 1:
            fn += 1

        predictions.append({
            "run": ex["run"],
            "vial_index": ex["vial_index"],
            "actual": actual,
            "predicted": pred,
            "generated_text": generated,
        })
        print(
            f"Validation {i}/{len(examples)}: actual={actual} "
            f"predicted={pred} output={generated[:100]!r}",
            flush=True,
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2

    metrics = {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "unparsed": unparsed,
        "validation_count": len(examples),
    }
    return metrics, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/workspace/twin_data/full_dataset_additional_5000"))
    ap.add_argument("--init-adapter", type=Path, default=Path("/workspace/twin_data/cosmos3_vial_lora_full"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/twin_data/cosmos3_vial_lora_v2"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-examples", type=int, default=0,
                    help="Optional cap for a short pilot; 0 uses all examples")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check data selection/split and print counts; do not load model")
    ap.add_argument("--smoke", action="store_true",
                    help="Train on at most 4 examples for one epoch (pipeline test)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_examples(args.data_root)
    train, val, val_runs = split_by_run(examples, args.val_fraction, args.seed)
    if args.smoke:
        # Select both labels so the weighted sampler can be tested.
        rng = random.Random(args.seed)
        by_label = defaultdict(list)
        for ex in train:
            by_label[ex["label"]].append(ex)
        if not by_label[0] or not by_label[1]:
            raise RuntimeError("Smoke test requires both clean and contaminated training examples.")
        smoke_train = []
        for label in (1, 0):
            rng.shuffle(by_label[label])
            smoke_train.extend(by_label[label][:2])
        rng.shuffle(smoke_train)
        train = smoke_train

        val_by_label = defaultdict(list)
        for ex in val:
            val_by_label[ex["label"]].append(ex)
        if not val_by_label[0] or not val_by_label[1]:
            raise RuntimeError("Smoke test requires both labels in validation.")
        val = [val_by_label[1][0], val_by_label[0][0]]
        args.epochs = 1
    elif args.max_train_examples:
        train = train[:args.max_train_examples]

    print(f"Generated dataset only: {args.data_root}")
    print(f"Eligible sequences: {len(examples)} across {len(set(e['run'] for e in examples))} runs")
    print(f"Train: {len(train)} {dict(Counter(e['label'] for e in train))}")
    print(f"Validation: {len(val)} {dict(Counter(e['label'] for e in val))}")
    print(f"Held-out run IDs: {val_runs}")
    if args.dry_run:
        print("Dry run complete; no model loaded and no files written.")
        return

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            "PEFT is not installed in cosmos3-env. Stop here; do not install blindly. "
            "We should first verify a PEFT version compatible with this Transformers build."
        ) from exc

    processor = AutoProcessor.from_pretrained(REPO, subfolder=SUBFOLDER)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        REPO, subfolder=SUBFOLDER, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = PeftModel.from_pretrained(
        model, args.init_adapter, is_trainable=True
    )
    model.print_trainable_parameters()
    device = model.device

    # Sample approximately 70% contaminated and 30% clean examples.
    # Label 1 = contaminated; label 0 = clean.
    counts = Counter(e["label"] for e in train)
    if 0 not in counts or 1 not in counts:
        raise RuntimeError(f"Training split must contain both classes: {dict(counts)}")
    class_probability = {1: 0.70, 0: 0.30}
    weights = [
        class_probability[e["label"]] / counts[e["label"]]
        for e in train
    ]
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sampled = list(iter(sampler))
        total_loss = 0.0
        for step, idx in enumerate(sampled, 1):
            batch = encode_example(train[idx], processor, device)
            output = model(**batch)
            loss = output.loss
            group_start = ((step - 1) // args.accum) * args.accum
            group_size = min(args.accum, len(sampled) - group_start)
            (loss / group_size).backward()
            total_loss += float(loss.detach())
            if step % args.accum == 0 or step == len(sampled):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if step % 10 == 0 or step == len(sampled):
                print(f"epoch {epoch + 1}/{args.epochs} example {step}/{len(sampled)} "
                      f"loss={float(loss.detach()):.4f}", flush=True)
            del batch, output, loss
        train_loss = total_loss / max(1, len(sampled))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ex in val:
                batch = encode_example(ex, processor, device)
                out = model(**batch)
                val_losses.append(float(out.loss))
                del batch, out
        val_loss = sum(val_losses) / max(1, len(val_losses))

        classification_metrics, val_predictions = evaluate_classification(
            model, processor, val, device
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "optimizer_steps": global_step,
            "classification": classification_metrics,
        }
        history.append(record)
        print("EPOCH SUMMARY:", json.dumps(record), flush=True)
        (args.out / f"validation_predictions_epoch_{epoch + 1}.json").write_text(
            json.dumps(val_predictions, indent=2)
        )
        model.save_pretrained(args.out)
        processor.save_pretrained(args.out)
        (args.out / "training_metrics.json").write_text(json.dumps({
            "base_model": f"{REPO}/{SUBFOLDER}",
            "dataset": str(args.data_root),
            "held_out_runs": val_runs,
            "train_examples": len(train),
            "validation_examples": len(val),
            "label_counts_train": dict(counts),
            "history": history,
        }, indent=2))
        print(f"Saved LoRA adapter checkpoint: {args.out}", flush=True)

    print("Training complete. Base model remains unchanged.")


if __name__ == "__main__":
    main()
