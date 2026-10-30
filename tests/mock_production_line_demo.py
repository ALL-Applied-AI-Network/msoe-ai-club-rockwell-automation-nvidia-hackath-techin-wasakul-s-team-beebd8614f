#!/usr/bin/env python3
"""Terminal demo for the Station 3 controller with no GPU, camera, or PLC.

Run from the repository root:
    python3 tests/mock_production_line_demo.py --scenario all

The demo prints the same operator stop banners as line_controller.py and checks
the in-memory PLC values that would be sent through EtherNet/IP in live mode.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from line_controller import LineController  # noqa: E402


OUTPUT_TAGS = (
    "Line_Running",
    "Station3_Dispense_Permissive",
    "MagneMover_Infeed_Active",
    "Alarm_Carryover_Fault",
    "Alarm_E_Stop",
    "Flush_Cycle_Required",
    "Fault_Code",
)


def banner(title):
    print("\n" + "#" * 78)
    print(f"# MOCK TEST: {title}")
    print("#" * 78)


def output_snapshot(controller):
    snapshot = {name: controller.plc.read_tag(name) for name in OUTPUT_TAGS}
    print("\nMachine output values:")
    print(json.dumps(snapshot, indent=2))
    return snapshot


def assert_running(snapshot):
    assert snapshot["Line_Running"] is True
    assert snapshot["Station3_Dispense_Permissive"] is True
    assert snapshot["MagneMover_Infeed_Active"] is True
    assert snapshot["Alarm_Carryover_Fault"] is False
    assert snapshot["Alarm_E_Stop"] is False
    assert snapshot["Flush_Cycle_Required"] is False
    assert snapshot["Fault_Code"] == 0


def assert_halted(snapshot, fault_code):
    assert snapshot["Line_Running"] is False
    assert snapshot["Station3_Dispense_Permissive"] is False
    assert snapshot["MagneMover_Infeed_Active"] is False
    assert snapshot["Alarm_Carryover_Fault"] is True
    assert snapshot["Alarm_E_Stop"] is True
    assert snapshot["Flush_Cycle_Required"] is True
    assert snapshot["Fault_Code"] == fault_code


def clean_test():
    banner("Clean vial continues production")
    controller = LineController(mock=True)
    status = controller.process_vial(
        image_path="mock_clean.png",
        recipe_id=6,
        prev_recipe_id=5,
        vial_in_batch=1,
        row_data={"contaminated": "0", "observed_color": "expected color"},
    )
    assert status == "CONTINUE", status
    assert_running(output_snapshot(controller))
    controller.plc.close()
    print("\nPASS: Clean vial left all production permissives enabled.")


def reactive_contamination_test():
    banner("Reactive AI contamination detection")
    controller = LineController(mock=True)
    status = controller.process_vial(
        image_path="mock_contaminated.png",
        recipe_id=6,
        prev_recipe_id=5,
        vial_in_batch=1,
        row_data={"contaminated": "1", "observed_color": "unexpected mixed color"},
    )
    assert status == "POST_FILL_STOP", status
    assert_halted(output_snapshot(controller), fault_code=402)
    print("\nPASS: Reactive contamination set every halt/flush output correctly.")

    banner("Operator flush and reset")
    controller.plc.reset_and_flush()
    assert_running(output_snapshot(controller))
    controller.plc.close()
    print("\nPASS: Simulated operator flush restored the permissives.")


def predictive_contamination_test():
    banner("Predictive halt before filling")
    controller = LineController(mock=True)
    status = controller.process_vial(
        image_path="not_used_before_fill.png",
        recipe_id=6,
        prev_recipe_id=5,
        vial_in_batch=1,
        row_data={"contaminated": "0"},
        enable_pre_fill_guard=True,
    )
    assert status == "PRE_FILL_HALT", status
    assert_halted(output_snapshot(controller), fault_code=301)
    controller.plc.close()
    print("\nPASS: Risky Recipe 5 -> Recipe 6 transition halted before dispensing.")


SCENARIOS = {
    "clean": clean_test,
    "reactive": reactive_contamination_test,
    "predictive": predictive_contamination_test,
}


def main():
    parser = argparse.ArgumentParser(
        description="Mock production-line output and operator-message demo"
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS],
        default="all",
        help="test one path or run the complete demo",
    )
    args = parser.parse_args()

    selected = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]
    try:
        for test in selected:
            test()
    except AssertionError as error:
        print(f"\nFAIL: Controller output did not match the expected result: {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("ALL REQUESTED MOCK PRODUCTION-LINE TESTS PASSED")
    print("No GPU, Hugging Face model, camera, network, or physical PLC was used.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
