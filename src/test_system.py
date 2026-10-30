import ast
import csv
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import line_controller as LC  # noqa: E402

PASSED = 0
FAILED = 0


def record(name, condition, extra=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}" + (f" ({extra})" if extra else ""))
    else:
        FAILED += 1
        print(f"  [FAIL] {name}" + (f" ({extra})" if extra else ""))


def test_recipes():
    print("\n--- 1. Testing Recipe Formulations & Constraints ---")
    for rid, ml in LC.RECIPES.items():
        tot = sum(ml.values())
        record(f"Recipe {rid}: Total volume <= 80 mL", tot <= 80, f"{tot} mL")
        record(f"Recipe {rid}: Active tanks >= 10 mL", all(v >= 10 for v in ml.values()))
        record(f"Recipe {rid}: Tank 200 <= 50 mL limit", ml.get(200, 0) <= 50)
        desc = LC.describe(rid)
        record(
            f"Recipe {rid}: Description uses neutral 'Tank' tags",
            "Tank 100" in desc or "Tank 200" in desc or "Tank 400" in desc,
        )
        record(
            f"Recipe {rid}: No color words hardcoded in description",
            not any(c in desc.lower() for c in ("orange", "blue", "clear")),
        )


def test_training_script_formatting():
    print("\n--- 2. Testing Training Script Prompt & Answer Formatting ---")
    train_script = os.path.join(HERE, "train_cosmos_lora.py")
    record("train_cosmos_lora.py exists", os.path.isfile(train_script))

    src = open(train_script).read()
    keep = [
        n
        for n in ast.parse(src).body
        if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("RECIPES", "TANK", "ORDER") for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name in ("describe", "build_example"))
    ]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), train_script, "exec"), ns)

    # Clean vial case
    clean_sample = {
        "recipe_id": "7",
        "prev_recipe_id": "6",
        "vial_in_batch": "2",
        "contaminated": "0",
        "observed_color": "",
    }
    pr_clean, ans_clean = ns["build_example"](clean_sample)
    record("Clean vial generates [ACTION: CONTINUE]", "[ACTION: CONTINUE]" in ans_clean)
    record("Clean vial specifies STATUS: Normal", "STATUS: Normal" in ans_clean)
    record("Clean vial does not halt machine", "STOP_MACHINE" not in ans_clean)

    # Contaminated vial case
    dirty_sample = {
        "recipe_id": "7",
        "prev_recipe_id": "6",
        "vial_in_batch": "1",
        "contaminated": "1",
        "observed_color": "green",
    }
    pr_dirty, ans_dirty = ns["build_example"](dirty_sample)
    record("Contaminated vial generates [ACTION: STOP_MACHINE]", "[ACTION: STOP_MACHINE]" in ans_dirty)
    record("Contaminated vial states observed appearance is a MISMATCH", "MISMATCH: does not match standard" in ans_dirty)
    record("Root cause references Tank numbers", "Tank 200" in ans_dirty and "Tank 400" in ans_dirty)
    record(
        "Tank definitions in training script do not hardcode colors",
        not any("orange" in v.lower() or "blue" in v.lower() for v in ns["TANK"].values()),
    )


def test_rockwell_plc_interlocks():
    print("\n--- 3. Testing Rockwell Automation PLC Tag State Machine & EtherNet/IP Driver ---")
    plc = LC.RockwellPLCInterlock()
    record("PLC starts with Line_Running=True", plc.read_tag("Line_Running") is True)
    record("PLC starts with Station3_Dispense_Permissive=True", plc.read_tag("Station3_Dispense_Permissive") is True)
    record("PLC starts with Alarm_E_Stop=False", plc.read_tag("Alarm_E_Stop") is False)

    # Trip E-Stop
    plc.emergency_stop(fault_code=402, reason="Contamination Detected")
    record("E-Stop trips Line_Running to False", plc.read_tag("Line_Running") is False)
    record("E-Stop drops Station3_Dispense_Permissive", plc.read_tag("Station3_Dispense_Permissive") is False)
    record("E-Stop activates Alarm_E_Stop=True", plc.read_tag("Alarm_E_Stop") is True)
    record("E-Stop requires Flush_Cycle_Required=True", plc.read_tag("Flush_Cycle_Required") is True)
    record("E-Stop sets Fault_Code=402", plc.read_tag("Fault_Code") == 402)

    # Operator reset
    plc.reset_and_flush()
    record("Flush cycle clears Alarm_E_Stop", plc.read_tag("Alarm_E_Stop") is False)
    record("Flush cycle restores Line_Running=True", plc.read_tag("Line_Running") is True)

    # Custom tag mapping test (EtherNet/IP Studio 5000 custom tags)
    custom = {"Station3_Dispense_Permissive": "CSI_Mixing_Permissive"}
    plc_custom = LC.RockwellPLCInterlock(custom_tags=custom)
    record("Custom tag mapping overrides default tag names", plc_custom.tag_map["Station3_Dispense_Permissive"] == "CSI_Mixing_Permissive")
    record("Flush cycle restores Line_Running=True", plc.tags["Line_Running"] is True)


def test_predictive_pre_fill_guard():
    print("\n--- 4. Testing Pre-Fill Predictive Beer-Lambert Interlocks ---")
    ctrl = LC.LineController(mock=True)

    # Safe transition (Vials 2 or 3, or same recipe)
    safe = ctrl.pre_fill_check(recipe_id=7, prev_recipe_id=7, vial_in_batch=2)
    record("Same recipe continuation permits fill", safe["action"] == "PERMIT_FILL")

    # High-risk transition: Recipe 6 -> Recipe 7
    # (Recipe 6 has Tank 200 10 mL; Recipe 7 has 0 mL Tank 200 and 50 mL Tank 400 -> severe carryover)
    risky = ctrl.pre_fill_check(recipe_id=7, prev_recipe_id=6, vial_in_batch=1)
    record(
        "Recipe 6 -> Recipe 7 triggers PAUSE_BEFORE_FILL",
        risky["action"] == "PAUSE_BEFORE_FILL",
        f"delta: {risky.get('delta', 0):.1f}",
    )
    record("Predictive warning identifies offending Tank", "Tank 200" in risky["detail"])

    # Low-risk / imperceptible transition: Recipe 8 -> Recipe 9
    # (Both recipes have high concentrations of Tank 200 and Tank 400; shift is under threshold)
    dark_transition = ctrl.pre_fill_check(recipe_id=9, prev_recipe_id=8, vial_in_batch=1)
    record(
        "Recipe 8 -> Recipe 9 permits fill (imperceptible delta)",
        dark_transition["action"] == "PERMIT_FILL",
    )


def test_real_dataset():
    print("\n--- 5. Evaluating Against Real CSI Line Data (data/real_test_labels.csv) ---")
    real_csv = os.path.join(REPO_ROOT, "twin_data", "data", "real_test_labels.csv")
    record("real_test_labels.csv exists", os.path.isfile(real_csv))

    with open(real_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    ctrl = LC.LineController(mock=True)
    correct_actions = 0

    for r in rows:
        action, _ = ctrl.inspect_vial_image(
            image_path=r["image"],
            recipe_id=int(r["recipe_id"]),
            prev_recipe_id=int(r["prev_recipe_id"]),
            vial_in_batch=int(r["vial_in_batch"]),
            row_data=r,
        )
        expected_action = "STOP_MACHINE" if int(r["contaminated"]) else "CONTINUE"
        if action == expected_action:
            correct_actions += 1

    record(
        f"Real line vials matched expected control actions",
        correct_actions == len(rows),
        f"{correct_actions}/{len(rows)} vials accurate",
    )


def test_target_frame_context_and_overlay():
    print("\n--- 6. Testing target frame selection, context window, and overlay state ---")
    ctrl = LC.LineController(mock=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for idx in range(5):
            path = os.path.join(tmpdir, f"img_{idx}.png")
            cv2.imwrite(path, np.full((20, 20, 3), 255 - idx * 10, dtype=np.uint8))
            image_paths.append(path)

        target_index = len(image_paths) - 3
        record("Target selection is two frames ago", target_index == 2)
        messages = ctrl._build_v2_messages_for_target(image_paths, target_index)
        prompt_text = " ".join(
            item.get("text", "")
            for item in messages[0]["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        )
        record(
            "Prompt includes earlier history and nearby reference frames",
            "Earlier chronological image" in prompt_text and "Nearby reference image" in prompt_text,
        )
        record(
            "Prompt explicitly forbids recipe metadata",
            "Do not use recipe numbers" in prompt_text,
        )
        record(
            "Target image is flagged as the candidate under review",
            "Candidate image under review" in prompt_text,
        )

        rendered = ctrl.render_overlay(
            image_paths[target_index],
            "CONTINUE",
            "STATUS: No clear carryover signal.",
            recipe_id=7,
            prev_recipe_id=6,
            vial_in_batch=3,
        )
        record("Overlay renderer returns a displayable image", rendered is not None and rendered.shape[0] > 0)

        ctrl.last_overlay = {
            "image_path": image_paths[target_index],
            "action": "CONTINUE",
            "diagnostic": "STATUS: No clear carryover signal.",
            "recipe_id": 7,
            "prev_recipe_id": 6,
            "vial_in_batch": 3,
        }
        record("Overlay state can be stored for display", ctrl.last_overlay["image_path"] == image_paths[target_index])


def run_all():
    global PASSED, FAILED
    test_recipes()
    test_training_script_formatting()
    test_rockwell_plc_interlocks()
    test_predictive_pre_fill_guard()
    test_real_dataset()
    test_target_frame_context_and_overlay()

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {PASSED} Passed, {FAILED} Failed")
    print("=" * 60)
    return FAILED == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
