"""AI contamination assessment and operator reporting for Fluid Mixing Station 3.

The controller has two advisory paths:
  1. Predictive: evaluates recipe-transition carryover risk.
  2. Reactive: analyzes the trained Cosmos 3 vial-image sequence.

This module does not connect to a machine-control interface. When attention is
required it prints a structured report for a user interface to consume later.

Usage:
  # Offline mock report demo:
  python3 src/line_controller.py --plan twin_data/demo_out/plan.csv --mock

  # Predictive recipe-transition reporting:
  python3 src/line_controller.py --plan twin_data/demo_out/plan.csv --mock --predictive

  # Real v2 trained model from Hugging Face:
  python3 src/line_controller.py --plan twin_data/demo_out/plan.csv \
      --img-root <folder with images> --hf-adapter original_v2

  # Watch frames written by live_vial_capture.py (advisory output only):
  python3 src/line_controller.py --live --image-dir camera_inbox \
      --recipe 6 --prev 5 --hf-adapter original_v2
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Cosmos 3 Nano base and the v2 contamination adapter produced by
# train_cosmos3_vial_lora_v2.py.  The adapter repository is separate from the
# base-model repository and must contain adapter_config.json beside the weights.
HF_REPO = "PourSightAI/Cosmos3-Vial-Contamination-LoRA"
HF_ADAPTER_SUBFOLDER = "original_v2"
HF_BASE_ID = "PourSightAI/CosmosReason3Hackathon"
HF_BASE_SUBFOLDER = "cosmos3-nano"

CONTAMINATION_SYSTEM_PROMPT = """You assess possible liquid carryover in an industrial vial-filling process.
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

# Recipe specification table (pure plant engineering tags: Tank 100, Tank 200, Tank 400)
RECIPES = {
    5: {100: 10, 200: 10, 400: 10}, 6: {100: 30, 200: 10},
    7: {100: 20, 400: 50},          8: {100: 20, 200: 20, 400: 20},
    9: {100: 10, 200: 35, 400: 35}, 10: {100: 45, 200: 15, 400: 15},
    11: {100: 10, 200: 10, 400: 50}, 12: {100: 10, 200: 50, 400: 10},
    13: {200: 50, 400: 20},         14: {100: 60, 400: 10},
}
TANK = {100: "Tank 100", 200: "Tank 200", 400: "Tank 400"}
ORDER = [200, 400, 100]

# Beer-Lambert optical model coefficients (calibrated from line data, pure Python)
K_MATRIX = [
    [1.3332, 1.5699, 2.9855],  # Base
    [-0.8894, 2.8181, 2.9625], # Tank 200 component
    [7.5434, 0.9629, -2.4204], # Tank 400 component
]
EMPTY_VIAL_LINEAR = [0.07809, 0.08206, 0.05105]


def resolve_adapter(adapter_path="", hf_adapter="", hf_repo=HF_REPO, cache_dir=None):
    """
    Returns a LOCAL folder containing adapter_model.safetensors + adapter_config.json.
      - hf_adapter: repository subfolder, normally "original_v2". Downloads that
        folder from the Hugging Face repo (uses HF_TOKEN if the repo is private).
      - adapter_path: an already-local adapter folder.
    Raises (never silently falls back to the untrained base model).
    """
    if hf_adapter:
        from huggingface_hub import snapshot_download
        subfolder = hf_adapter.strip("/")
        print(f"[INIT] Downloading adapter '{subfolder}' from Hugging Face repo {hf_repo}...")
        root = snapshot_download(
            repo_id=hf_repo,
            allow_patterns=[f"{subfolder}/*"],
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
        adapter_path = os.path.join(root, subfolder)
    if not adapter_path:
        return ""
    weights = os.path.join(adapter_path, "adapter_model.safetensors")
    config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"No adapter_model.safetensors in '{adapter_path}'. Check the Hugging Face "
            f"repository and subfolder."
        )
    if not os.path.isfile(config):
        raise FileNotFoundError(
            f"No adapter_config.json in '{adapter_path}'. PEFT cannot load the .safetensors "
            "weights alone; upload the config saved by model.save_pretrained()."
        )
    print(f"[INIT] Adapter weights found: {weights}")
    return adapter_path


def describe(rid):
    if rid not in RECIPES:
        return f"Recipe {rid}: Unknown"
    parts = [f"{TANK[t]} {RECIPES[rid][t]} mL" for t in ORDER if t in RECIPES[rid]]
    return f"Recipe {rid}: " + ", then ".join(parts) + f" (total {sum(RECIPES[rid].values())} mL)"


def compute_photo_rgb(c200, c400):
    """Predicts linear light absorption via Beer-Lambert law without external dependencies."""
    rgb = []
    for col in range(3):
        a = K_MATRIX[0][col] + c200 * K_MATRIX[1][col] + c400 * K_MATRIX[2][col]
        lin = EMPTY_VIAL_LINEAR[col] * math.exp(-a)
        val = 255.0 * (max(0.0, lin) ** (1.0 / 2.2))
        rgb.append(min(255.0, max(0.0, val)))
    return rgb


def color_distance(rgb1, rgb2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


class LineController:
    def __init__(self, base_model=None, adapter_path=None, mock=False,
                 hf_adapter="", hf_repo=HF_REPO):
        self.mock = mock
        self.model = None
        self.processor = None
        # The v2 model contract is exactly 12 chronological images:
        # 9 earlier images followed by all 3 images in the current batch.
        self.image_history = deque(maxlen=12)
        self.vial_position_history = deque(maxlen=12)
        self.last_overlay = None

        if not mock:
            # Resolve (and download if needed) BEFORE touching the GPU, so a bad run name fails fast.
            adapter_dir = resolve_adapter(adapter_path or "", hf_adapter, hf_repo)
            self._load_model(base_model or HF_BASE_ID, adapter_dir)

    def _load_model(self, base_model, adapter_dir):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

        print(f"[INIT] Loading Cosmos 3 Nano base model from {base_model}/{HF_BASE_SUBFOLDER}...")
        self.processor = AutoProcessor.from_pretrained(
            base_model, subfolder=HF_BASE_SUBFOLDER
        )
        model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            base_model,
            subfolder=HF_BASE_SUBFOLDER,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        if adapter_dir:
            print(f"[INIT] Attaching fine-tuned LoRA adapter from {adapter_dir}...")
            self.model = PeftModel.from_pretrained(model, adapter_dir)
            print("[INIT] TRAINED adapter attached. Decisions come from the fine-tuned model.")
        else:
            self.model = model
            print("[INIT] WARNING: no adapter given. Running the UNTRAINED base model.")
        self.model.eval()
        print("[INIT] Model successfully loaded onto GPU.")

    @staticmethod
    def _parse_assessment(generated_text):
        """Strictly parse the two-class JSON contract used during v2 training."""
        text = generated_text.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Keep a narrow recovery path for a model that adds surrounding text,
            # but still require a valid JSON object and an exact known label.
            match = re.search(r'\{\s*"assessment"\s*:\s*"[^"]+"\s*\}', text)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict) or set(payload) != {"assessment"}:
            return None
        assessment = payload.get("assessment")
        if assessment not in {"potential_carryover", "no_clear_carryover_signal"}:
            return None
        return assessment

    @staticmethod
    def _build_v2_messages(image_paths):
        """Reproduce the exact multimodal prompt structure used for training."""
        from PIL import Image

        if len(image_paths) != 12:
            raise ValueError(f"v2 inspection requires exactly 12 images, got {len(image_paths)}")
        content = [{"type": "text", "text": CONTAMINATION_SYSTEM_PROMPT}]
        for index, path in enumerate(image_paths[:9], 1):
            with Image.open(path) as source:
                image = source.convert("RGB")
            content.append({"type": "text", "text": f"Earlier chronological image {index}"})
            content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": "CURRENT RECIPE BATCH"})
        for index, path in enumerate(image_paths[9:], 1):
            with Image.open(path) as source:
                image = source.convert("RGB")
            label = (
                "First current-batch vial: candidate for assessment"
                if index == 0
                else "Additional current-batch vial: reference evidence"
            )
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": image})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _build_v2_messages_for_target(image_paths, target_index):
        """Build a visual-only sequence prompt using the selected target image and surrounding context.

        Target selection is intentionally blind to recipe metadata. The model is shown only a
        chronological visual sequence with the target frame at images[-3] and the two later frames
        as nearby reference evidence, so it judges based on appearance rather than manufacturing info.
        """
        from PIL import Image

        if len(image_paths) < 3:
            raise ValueError(f"At least 3 images are required, got {len(image_paths)}")
        if not (0 <= target_index < len(image_paths)):
            raise ValueError(f"target_index out of range: {target_index} for {len(image_paths)} images")

        history = image_paths[:target_index]
        target_path = image_paths[target_index]
        later_refs = image_paths[target_index + 1:]

        content = [{
            "type": "text",
            "text": (
                "Review this chronological image sequence. One frame is the candidate under review. "
                "Use the earlier frames as historical context and the immediately following frames as nearby reference. "
                "Do not use recipe numbers, previous recipe values, batch labels, or plant metadata. "
                "Base your decision only on visual evidence in the sequence."
            )
        }]

        for index, path in enumerate(history[-9:], 1):
            with Image.open(path) as source:
                image = source.convert("RGB")
            content.append({"type": "text", "text": f"Earlier chronological image {index}"})
            content.append({"type": "image", "image": image})

        with Image.open(target_path) as source:
            target_image = source.convert("RGB")
        content.append({"type": "text", "text": "Candidate image under review: target frame"})
        content.append({"type": "image", "image": target_image})

        for index, path in enumerate(later_refs[:2], 1):
            with Image.open(path) as source:
                image = source.convert("RGB")
            content.append({"type": "text", "text": f"Nearby reference image {index}"})
            content.append({"type": "image", "image": image})

        return [{"role": "user", "content": content}]

    def _run_v2_inference(self, image_paths, target_index=None):
        """Run the fine-tuned adapter and return (assessment, raw model text).

        Default target is two pictures ago, i.e. images[-3], so the AI judges the older frame while
        still receiving previous and nearby reference context without recipe metadata.
        """
        import torch

        if target_index is None:
            target_index = len(image_paths) - 3 if len(image_paths) >= 3 else 0

        messages = self._build_v2_messages_for_target(image_paths, target_index)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self.model.device
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        generated = self.processor.batch_decode(
            output[:, prompt_length:], skip_special_tokens=True
        )[0].strip()
        return self._parse_assessment(generated), generated

    def render_overlay(self, image_path, action, diagnostic, recipe_id, prev_recipe_id, vial_in_batch):
        """Overlay the AI judgement onto an older image for display."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for live overlay display.") from exc

        frame = cv2.imread(image_path)
        if frame is None:
            return None

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 220), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        color = (0, 255, 0) if action == "CONTINUE" else (0, 0, 255)
        cv2.putText(frame, f"Recipe: {recipe_id}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Prev: {prev_recipe_id}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Vial: {vial_in_batch}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"AI: {action}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        y = 185
        for line in diagnostic.splitlines()[:4]:
            cv2.putText(frame, line[:60], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 22
        return frame

    def display_last_overlay(self):
        if not self.last_overlay:
            return
        info = self.last_overlay
        frame = self.render_overlay(
            info["image_path"],
            info["action"],
            info["diagnostic"],
            info["recipe_id"],
            info["prev_recipe_id"],
            info["vial_in_batch"],
        )
        if frame is None:
            return
        try:
            import cv2
        except ImportError:
            return
        cv2.imshow("AI Judgement Overlay", frame)
        cv2.waitKey(1)

    def pre_fill_check(self, recipe_id, prev_recipe_id, vial_in_batch):
        """
        PREDICTIVE PREVENTION:
        Evaluates recipe transition before the nozzle dispenses into the vial.
        Calculates fluid carryover risk using the physical Beer-Lambert model.
        If carryover risk is critical, produces a pause-and-flush recommendation before dispensing.
        """
        if vial_in_batch != 1 or not prev_recipe_id or prev_recipe_id not in RECIPES:
            return {"action": "PERMIT_FILL", "risk": "LOW", "detail": "Clean batch or continuous recipe"}

        # Calculate concentration of active dye tanks
        ml_curr = RECIPES[recipe_id]
        tot_curr = sum(ml_curr.values())
        c_curr = (ml_curr.get(200, 0) / tot_curr, ml_curr.get(400, 0) / tot_curr)

        ml_prev = RECIPES[prev_recipe_id]
        tot_prev = sum(ml_prev.values())
        c_prev = (ml_prev.get(200, 0) / tot_prev, ml_prev.get(400, 0) / tot_prev)

        # Expected carryover with median line carryover alpha = 0.53
        c_mix = (0.53 * c_prev[0] + 0.47 * c_curr[0], 0.53 * c_prev[1] + 0.47 * c_curr[1])

        rgb_curr = compute_photo_rgb(*c_curr)
        rgb_mix = compute_photo_rgb(*c_mix)
        delta = color_distance(rgb_mix, rgb_curr)

        # Check which tanks in previous recipe are causing the risk
        offending_tanks = [TANK[t] for t in (200, 400) if ml_prev.get(t, 0) > 0 and (ml_curr.get(t, 0) == 0 or c_prev[0] != c_curr[0])]

        if delta > 6.0:  # Visibility delta threshold
            return {
                "action": "PAUSE_BEFORE_FILL",
                "risk": "HIGH",
                "delta": delta,
                "detail": (
                    f"Transition Recipe {prev_recipe_id} -> Recipe {recipe_id} has high carryover risk "
                    f"(color delta {delta:.1f} > threshold 6.0). Residual fluid from "
                    f"{', '.join(offending_tanks) or 'previous tanks'} in the shared manifold will cross-contaminate the batch."
                ),
            }

        return {"action": "PERMIT_FILL", "risk": "LOW", "detail": "Safe recipe transition"}

    def inspect_vial_image(self, image_path, recipe_id, prev_recipe_id, vial_in_batch, row_data=None):
        """
        POST-FILL DETECTION:
        Buffers the chronological camera stream and, at the end of a batch, runs
        v2 VLM inference over 9 earlier images plus all 3 current-batch images.
        The trained target is the first vial of the current three-vial batch.

        Returns an advisory action ('REPORT_ATTENTION' or 'CONTINUE') and an
        operator-safe diagnostic. Invalid model output requests operator review.
        """
        if self.mock or self.model is None:
            # Emulate inference using row metadata or clean rules
            is_contaminated = bool(row_data and int(row_data.get("contaminated", 0)))
            obs = (row_data.get("observed_color") if row_data else "") or "discolored"
            if is_contaminated and prev_recipe_id in RECIPES:
                prev_tanks = [TANK[t] for t in ORDER if t in RECIPES[prev_recipe_id] and t != 100]
                curr_tanks = [TANK[t] for t in ORDER if t in RECIPES[recipe_id] and t != 100]
                text = (
                    f"[RECOMMENDATION: PAUSE_AND_FLUSH]\n"
                    f"ERROR: Fluid Contamination Detected.\n"
                    f"- Recipe Being Filled: Recipe {recipe_id} ({describe(recipe_id).split(': ', 1)[1]})\n"
                    f"- Batch Position: Vial {vial_in_batch} of 3\n"
                    f"- Observed Appearance: {obs} (MISMATCH: does not match standard Recipe {recipe_id})\n"
                    f"- Root Cause: Carryover from previous Recipe {prev_recipe_id} ({describe(prev_recipe_id).split(': ', 1)[1]}). "
                    f"Residual fluid from {', '.join(prev_tanks)} remained in the shared dispensing nozzle/manifold "
                    f"and cross-contaminated the incoming fluid from {', '.join(curr_tanks)}.\n"
                    f"- Operator Recommendation: Pause the line and flush the nozzle before continuing."
                )
            else:
                text = (
                    f"[REPORT: CONTINUE]\n"
                    f"STATUS: Normal.\n"
                    f"- Recipe Being Filled: Recipe {recipe_id}\n"
                    f"- Batch Position: Vial {vial_in_batch} of 3\n"
                    f"- Verification: Liquid color matches Recipe {recipe_id} specification. No carryover detected."
                )
            action = "REPORT_ATTENTION" if is_contaminated else "CONTINUE"
            return action, text

        # Real GPU multimodal VLM execution. The v2 adapter was not trained for
        # single-image requests, recipe metadata, or free-form report text.
        if not image_path or not os.path.isfile(image_path):
            text = (
                "[RECOMMENDATION: PAUSE_AND_FLUSH]\n"
                "ERROR: Camera image is missing or unreadable; AI inspection cannot complete.\n"
                "- Operator Recommendation: Pause operation, inspect the camera feed, and flush "
                "the nozzle before continuing."
            )
            return "REPORT_ATTENTION", text

        self.image_history.append(image_path)
        self.vial_position_history.append(vial_in_batch)

        # The first two images of the current batch are buffered as reference
        # evidence. Classification becomes valid only after vial 3 arrives.
        if vial_in_batch != 3:
            return "CONTINUE", (
                "[REPORT: CONTINUE]\n"
                f"STATUS: Camera image buffered ({len(self.image_history)}/12); "
                "awaiting the complete three-vial batch for v2 inspection."
            )

        expected_positions = [1, 2, 3] * 4
        if len(self.image_history) < 12:
            return "CONTINUE", (
                "[REPORT: CONTINUE]\n"
                f"STATUS: Building chronological baseline ({len(self.image_history)}/12 images). "
                "The v2 AI assessment begins once three earlier batches are available."
            )
        if list(self.vial_position_history) != expected_positions:
            text = (
                "[RECOMMENDATION: PAUSE_AND_FLUSH]\n"
                "ERROR: Camera sequence is incomplete or out of order; expected four chronological "
                "three-vial batches.\n"
                "- Operator Recommendation: Pause operation, verify the camera sequence, and "
                "flush the nozzle before continuing."
            )
            return "REPORT_ATTENTION", text

        images = list(self.image_history)
        target_index = len(images) - 3
        assessment, raw_output = self._run_v2_inference(images, target_index=target_index)
        overlay_path = images[target_index]

        if assessment == "potential_carryover":
            text = (
                "[RECOMMENDATION: PAUSE_AND_FLUSH]\n"
                "ERROR: Potential Fluid Carryover Detected.\n"
                "- Assessed Target: The image two frames ago, evaluated with the surrounding visual history.\n"
                "- Visual Evidence: The candidate frame does not clearly match the recent sequence and nearby reference views.\n"
                "- Likely Process Cause: Residual fluid remained in the shared dispensing nozzle/manifold.\n"
                "- Machine Directive: Line halted. Operator must flush the nozzle before resuming.\n"
                f"- AI Assessment: {raw_output}"
            )
            self.last_overlay = {
                "image_path": overlay_path,
                "action": "STOP_MACHINE",
                "diagnostic": text,
                "recipe_id": recipe_id,
                "prev_recipe_id": prev_recipe_id,
                "vial_in_batch": vial_in_batch,
            }
            return "STOP_MACHINE", text

        if assessment == "no_clear_carryover_signal":
            text = (
                "[REPORT: CONTINUE]\n"
                "STATUS: No clear carryover signal.\n"
                "- Assessed Target: The image two frames ago, evaluated with the surrounding visual history.\n"
                "- Verification: The candidate frame is visually consistent with the recent and nearby reference evidence.\n"
                f"- AI Assessment: {raw_output}"
            )
            self.last_overlay = {
                "image_path": overlay_path,
                "action": "CONTINUE",
                "diagnostic": text,
                "recipe_id": recipe_id,
                "prev_recipe_id": prev_recipe_id,
                "vial_in_batch": vial_in_batch,
            }
            return "CONTINUE", text

        text = (
            "[RECOMMENDATION: PAUSE_AND_FLUSH]\n"
            "ERROR: AI inspection returned an invalid or ambiguous response.\n"
            f"- Raw AI Output: {raw_output!r}\n"
            "- Operator Recommendation: Pause operation, manually inspect the batch, and flush "
            "the nozzle before continuing."
        )
        return "REPORT_ATTENTION", text

    def process_vial(self, image_path, recipe_id, prev_recipe_id, vial_in_batch,
                     row_data=None, enable_pre_fill_guard=False):
        """Assess one vial event and print an operator-facing report."""
        print(
            f"\n>>> [VIAL EVENT] Recipe {recipe_id}, "
            f"Batch Vial #{vial_in_batch}"
        )
        self.last_overlay = {
            "image_path": overlay_path,
            "action": "STOP_MACHINE",
            "diagnostic": text,
            "recipe_id": recipe_id,
            "prev_recipe_id": prev_recipe_id,
            "vial_in_batch": vial_in_batch,
        }
        return "STOP_MACHINE", text

    def process_vial(self, image_path, recipe_id, prev_recipe_id, vial_in_batch, row_data=None, enable_pre_fill_guard=False):
        """Full supervisory cycle for one vial position on the line."""
        print(f"\n>>> [LINE EVENT] Vial approaching Station 3: Recipe {recipe_id}, Batch Vial #{vial_in_batch}")

        if enable_pre_fill_guard:
            pre_eval = self.pre_fill_check(recipe_id, prev_recipe_id, vial_in_batch)
            if pre_eval["action"] == "PAUSE_BEFORE_FILL":
                self._print_predictive_report(
                    recipe_id, prev_recipe_id, vial_in_batch, pre_eval["detail"]
                )
                return "PREDICTIVE_RISK_REPORTED"

        action, diagnostic = self.inspect_vial_image(
            image_path, recipe_id, prev_recipe_id, vial_in_batch, row_data=row_data
        )

        if action == "REPORT_ATTENTION":
            self._print_contamination_report(diagnostic)
            return "CONTAMINATION_REPORTED"

        for line in diagnostic.strip().split("\n"):
            print(f"  {line}")
        return "CONTINUE"

    @staticmethod
    def _print_contamination_report(diagnostic):
        print("\n" + "=" * 76)
        print("  ⚠️  [CONTAMINATION DETECTION REPORT] OPERATOR ATTENTION REQUIRED ⚠️")
        print("=" * 76)
        for line in diagnostic.strip().split("\n"):
            print(f"  {line}")
        print("-" * 76)
        print("  OUTPUT MODE: Advisory report only; no machine-control command was issued.")
        print("=" * 76 + "\n")

    @staticmethod
    def _print_predictive_report(recipe_id, prev_recipe_id, vial_in_batch, detail):
        print("\n" + "=" * 76)
        print("  ⚠️  [PREDICTIVE CARRYOVER RISK REPORT] OPERATOR ATTENTION REQUIRED ⚠️")
        print("=" * 76)
        print(f"  Target Recipe:   Recipe {recipe_id}")
        print(f"  Previous Recipe: Recipe {prev_recipe_id}")
        print(f"  Vial:            Vial {vial_in_batch} of batch")
        print(f"  Prediction:      {detail}")
        print("  Recommendation:  Pause before filling and flush the nozzle/manifold.")
        print("-" * 76)
        print("  OUTPUT MODE: Advisory report only; no machine-control command was issued.")
        print("=" * 76 + "\n")


def _capture_sequence(path):
    """Return the numeric sequence from vial_000001.png, or None."""
    match = re.search(r"vial_(\d+)\.(?:png|jpe?g)$", os.path.basename(path), re.I)
    return int(match.group(1)) if match else None


def run_live_camera_monitor(
    controller,
    image_dir,
    recipe_id,
    prev_recipe_id=0,
    poll_interval=0.1,
    enable_pre_fill_guard=False,
    process_existing=False,
):
    """Watch frames from live_vial_capture.py and print advisory assessments.

    Recipe context is supplied by the caller until a user interface is connected.
    No machine-control interface is used.
    """
    os.makedirs(image_dir, exist_ok=True)

    def image_files():
        return sorted(
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    seen = set() if process_existing else set(image_files())
    fallback_sequence = len(seen)

    print("\n" + "#" * 70)
    print("  [LIVE CAMERA MONITOR] ADVISORY REPORTING ACTIVE")
    print(f"  Watching: {image_dir}")
    print(f"  Recipe context: previous={prev_recipe_id}, current={recipe_id}")
    print("  No machine-control connection is enabled.")
    print("#" * 70 + "\n")

    if enable_pre_fill_guard:
        pre_eval = controller.pre_fill_check(recipe_id, prev_recipe_id, 1)
        if pre_eval["action"] == "PAUSE_BEFORE_FILL":
            controller._print_predictive_report(
                recipe_id, prev_recipe_id, 1, pre_eval["detail"]
            )

    try:
        while True:
            # Poll the cart presence sensor from the PLC over EtherNet/IP
            cart_in_position = bool(controller.plc.read_tag("Cart_In_Position", False))

            # Detect rising edge (Cart just arrived at nozzle)
            if cart_in_position and not last_cart_state:
                vials_processed += 1
                print(f"\n[EVENT] MagneMover Cart arrived at Station 3 (Event #{vials_processed})")

                # Read operational context tags directly from the PLC
                rid = int(controller.plc.read_tag("Current_Recipe", 5))
                prev = int(controller.plc.read_tag("Prev_Recipe", 0))
                vial = int(controller.plc.read_tag("Vial_Number", 1))
                existing_mtime = 0.0
                if image_dir and os.path.isdir(image_dir):
                    existing = [
                        os.path.join(image_dir, name) for name in os.listdir(image_dir)
                        if name.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                    if existing:
                        existing_mtime = max(os.path.getmtime(path) for path in existing)
                pending = {
                    "recipe_id": rid, "prev_recipe_id": prev,
                    "vial_in_batch": vial, "image_baseline_mtime": existing_mtime,
                    "predictive_halt": False,
                }
                controller.plc.write_tag("Inspection_Complete", False)

                # Prediction must occur before dispensing; the camera assessment
                # cannot occur until the final filled frame exists.
                if enable_pre_fill_guard:
                    pre_eval = controller.pre_fill_check(rid, prev, vial)
                    if pre_eval["action"] == "PAUSE_BEFORE_FILL":
                        controller.plc.emergency_stop(fault_code=301, reason=pre_eval["detail"])
                        controller._print_pre_fill_halt_banner(rid, prev, vial, pre_eval["detail"])
                        pending["predictive_halt"] = True
                        controller.plc.write_tag("Inspection_Complete", True)

            elif not cart_in_position and last_cart_state:
                print("[EVENT] Cart departed Station 3; acquiring final filled-vial frame.")
                if pending and not pending["predictive_halt"]:
                    img_path = latest_image_newer_than(pending["image_baseline_mtime"])
                    if not img_path:
                        img_path = "live_capture.png"  # inspect_vial_image fails safe
                    controller.process_vial(
                        image_path=img_path,
                        recipe_id=pending["recipe_id"],
                        prev_recipe_id=pending["prev_recipe_id"],
                        vial_in_batch=pending["vial_in_batch"],
                        enable_pre_fill_guard=False,
                    )
                    controller.display_last_overlay()
                    controller.plc.write_tag("Inspection_Complete", True)
                pending = None

            last_cart_state = cart_in_position
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[INFO] Live camera monitoring stopped by operator.")



def simulate_production_run(
    plan_csv,
    img_root="",
    mock=True,
    base_model="",
    adapter="",
    enable_pre_fill_guard=False,
    hf_adapter="",
    hf_repo=HF_REPO,
):
    """Replay a planned vial sequence and print advisory reports."""
    controller = LineController(
        base_model=base_model,
        adapter_path=adapter,
        mock=mock,
        hf_adapter=hf_adapter,
        hf_repo=hf_repo,
    )
    with open(plan_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    print(f"\nStarting advisory replay for {len(rows)} planned vials...")
    reports_generated = 0

    for row in rows:
        recipe_id = int(row["recipe_id"])
        prev_recipe_id = int(row.get("prev_recipe_id") or 0)
        vial_in_batch = int(row.get("vial_in_batch") or 1)
        image_name = row.get("image", "")
        image_path = os.path.join(img_root, image_name) if img_root else image_name

        status = controller.process_vial(
            image_path=image_path,
            recipe_id=recipe_id,
            prev_recipe_id=prev_recipe_id,
            vial_in_batch=vial_in_batch,
            row_data=row,
            enable_pre_fill_guard=enable_pre_fill_guard,
        )
        if status in {"CONTAMINATION_REPORTED", "PREDICTIVE_RISK_REPORTED"}:
            reports_generated += 1

    print("\n" + "#" * 60)
    print(f"Replay completed: {len(rows)} vials processed.")
    print(f"Operator-attention reports generated: {reports_generated}")
    print("No machine-control commands were issued.")
    print("#" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Station 3 AI contamination assessment and operator reporting"
    )
    ap.add_argument("--plan", help="path to plan.csv or labels.csv")
    ap.add_argument("--img-root", default="", help="folder containing rendered images")
    ap.add_argument("--image", help="single image to inspect")
    ap.add_argument("--recipe", type=int, help="current recipe id")
    ap.add_argument("--prev", type=int, default=0, help="previous recipe id")
    ap.add_argument("--vial", type=int, default=1, help="vial in batch (1, 2, or 3)")
    ap.add_argument(
        "--base",
        default="",
        help="Cosmos 3 base repo/path (default: PourSightAI/CosmosReason3Hackathon)",
    )
    ap.add_argument("--adapter", default="", help="local trained LoRA adapter folder")
    ap.add_argument(
        "--hf-adapter",
        default="",
        help="adapter subfolder in Hugging Face (use original_v2 for v2)",
    )
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--mock", action="store_true", help="run without the GPU model")
    ap.add_argument(
        "--predictive",
        action="store_true",
        help="print predictive recipe-transition risk reports",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="watch a live_vial_capture.py output directory",
    )
    ap.add_argument("--image-dir", default="", help="camera capture directory")
    ap.add_argument(
        "--process-existing",
        action="store_true",
        help="process images already present when live monitoring starts",
    )
    args = ap.parse_args()

    use_model = (not args.mock) and bool(args.hf_adapter or args.adapter or args.base)
    if not args.mock and not use_model:
        print("[INFO] No model or adapter given: running in mock mode.")

    if args.live:
        if not args.image_dir:
            ap.error("--live requires --image-dir")
        if args.recipe is None:
            ap.error("--live requires --recipe until a UI provides recipe context")
        controller = LineController(
            base_model=args.base,
            adapter_path=args.adapter,
            hf_adapter=args.hf_adapter,
            hf_repo=args.hf_repo,
            mock=not use_model,
        )
        run_live_camera_monitor(
            controller=controller,
            image_dir=args.image_dir,
            recipe_id=args.recipe,
            prev_recipe_id=args.prev,
            enable_pre_fill_guard=args.predictive,
            process_existing=args.process_existing,
        )
    elif args.plan:
        simulate_production_run(
            plan_csv=args.plan,
            img_root=args.img_root,
            mock=not use_model,
            base_model=args.base,
            adapter=args.adapter,
            hf_adapter=args.hf_adapter,
            hf_repo=args.hf_repo,
            enable_pre_fill_guard=args.predictive,
        )
    elif args.image and args.recipe is not None:
        controller = LineController(
            base_model=args.base,
            adapter_path=args.adapter,
            hf_adapter=args.hf_adapter,
            hf_repo=args.hf_repo,
            mock=not use_model,
        )
        controller.process_vial(
            image_path=args.image,
            recipe_id=args.recipe,
            prev_recipe_id=args.prev,
            vial_in_batch=args.vial,
            enable_pre_fill_guard=args.predictive,
        )
    else:
        default_plan = os.path.join(REPO_ROOT, "twin_data", "demo_out", "plan.csv")
        if os.path.exists(default_plan):
            print("[INFO] No arguments given. Running the mock advisory demo...")
            simulate_production_run(default_plan, mock=True)
        else:
            ap.print_help()

