#!/usr/bin/env python3
"""Terminal demo for advisory contamination reports without hardware control."""

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from line_controller import LineController  # noqa: E402


def banner(title):
    print("\n" + "#" * 78)
    print(f"# MOCK TEST: {title}")
    print("#" * 78)


def capture_call(callback):
    stream = io.StringIO()
    with redirect_stdout(stream):
        status = callback()
    output = stream.getvalue()
    print(output, end="")
    return status, output


def clean_test():
    banner("Clean vial report")
    controller = LineController(mock=True)
    status, output = capture_call(lambda: controller.process_vial(
        image_path="mock_clean.png",
        recipe_id=6,
        prev_recipe_id=5,
        vial_in_batch=1,
        row_data={"contaminated": "0", "observed_color": "expected color"},
    ))
    assert status == "CONTINUE", status
    assert "[REPORT: CONTINUE]" in output
    assert "CONTAMINATION DETECTION REPORT" not in output
    print("\nPASS: Clean vial produced a continue report and no alert.")


def reactive_contamination_test():
    banner("Reactive AI contamination report")
    controller = LineController(mock=True)
    status, output = capture_call(lambda: controller.process_vial(
        image_path="mock_contaminated.png",
        recipe_id=6,
        prev_recipe_id=5,
        vial_in_batch=1,
        row_data={"contaminated": "1", "observed_color": "unexpected mixed color"},
    ))
    assert status == "CONTAMINATION_REPORTED", status
    assert "CONTAMINATION DETECTION REPORT" in output
    assert "Fluid Contamination Detected" in output
    assert "flush the nozzle" in output
    assert "no machine-control command was issued" in output
    print("\nPASS: Contamination produced the complete operator-facing report.")


def predictive_contamination_test():
    banner("Predictive carryover-risk report")
    controller = LineController(mock=True)
    status, output = capture_call(lambda: controller.process_vial(
        image_path="not_used_before_fill.png",
        recipe_id=6,
        prev_recipe_id=5,
        vial_in_batch=1,
        row_data={"contaminated": "0"},
        enable_pre_fill_guard=True,
    ))
    assert status == "PREDICTIVE_RISK_REPORTED", status
    assert "PREDICTIVE CARRYOVER RISK REPORT" in output
    assert "Pause before filling" in output
    assert "no machine-control command was issued" in output
    print("\nPASS: Risky transition produced a predictive operator report.")


SCENARIOS = {
    "clean": clean_test,
    "reactive": reactive_contamination_test,
    "predictive": predictive_contamination_test,
}


def main():
    parser = argparse.ArgumentParser(description="Advisory report terminal demo")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    args = parser.parse_args()
    selected = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]

    try:
        for test in selected:
            test()
    except AssertionError as error:
        print(f"\nFAIL: Report output did not match expectations: {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("ALL REQUESTED ADVISORY REPORT TESTS PASSED")
    print("No hardware-control or network interface was used.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

