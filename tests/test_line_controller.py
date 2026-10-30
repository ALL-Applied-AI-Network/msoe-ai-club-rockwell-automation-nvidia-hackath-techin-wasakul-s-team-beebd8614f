import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import line_controller as LC  # noqa: E402


class TestLineControllerMock(unittest.TestCase):
    def setUp(self):
        self.controller = LC.LineController(mock=True)

    def test_describe_and_risk_model(self):
        self.assertIn("Tank 200", LC.describe(6))
        self.assertIn("Tank 400", LC.describe(7))

        safe = self.controller.pre_fill_check(recipe_id=7, prev_recipe_id=7, vial_in_batch=2)
        self.assertEqual(safe["action"], "PERMIT_FILL")

        risky = self.controller.pre_fill_check(recipe_id=7, prev_recipe_id=6, vial_in_batch=1)
        self.assertEqual(risky["action"], "PAUSE_BEFORE_FILL")
        self.assertIn("Tank 200", risky["detail"])

    def test_mock_reactive_contamination_report(self):
        status, report = self.controller.process_vial(
            image_path="mock_contaminated.png",
            recipe_id=6,
            prev_recipe_id=5,
            vial_in_batch=1,
            row_data={"contaminated": "1", "observed_color": "unexpected mixed color"},
        )

        self.assertEqual(status, "CONTAMINATION_REPORTED")
        self.assertIn("CONTAMINATION DETECTION REPORT", report)
        self.assertIn("Fluid Contamination Detected", report)
        self.assertIn("flush the nozzle", report.lower())

    def test_mock_predictive_guard_report(self):
        status, report = self.controller.process_vial(
            image_path="unused.png",
            recipe_id=7,
            prev_recipe_id=6,
            vial_in_batch=1,
            row_data={"contaminated": "0"},
            enable_pre_fill_guard=True,
        )

        self.assertEqual(status, "PREDICTIVE_RISK_REPORTED")
        self.assertIn("PREDICTIVE CARRYOVER RISK REPORT", report)
        self.assertIn("Pause before filling", report)
        self.assertIn("no machine-control command was issued", report.lower())

    def test_overlay_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "sample.png")
            Image.new("RGB", (640, 480), color=(255, 255, 255)).save(image_path)

            frame = self.controller.render_overlay(
                image_path=image_path,
                action="CONTINUE",
                diagnostic="STATUS: No clear carryover signal.",
                recipe_id=7,
                prev_recipe_id=6,
                vial_in_batch=3,
            )

            self.assertIsNotNone(frame)
            self.assertGreater(frame.shape[0], 0)
            self.assertGreater(frame.shape[1], 0)

    def test_target_frame_prompt_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = []
            for idx in range(5):
                image_path = os.path.join(tmpdir, f"img_{idx}.png")
                Image.new("RGB", (64, 64), color=(255 - idx * 10, 100, 50)).save(image_path)
                image_paths.append(image_path)

            target_index = len(image_paths) - 3
            messages = LC.LineController._build_v2_messages_for_target(image_paths, target_index)
            prompt_text = " ".join(
                item.get("text", "")
                for item in messages[0]["content"]
                if isinstance(item, dict) and item.get("type") == "text"
            )

            self.assertIn("Earlier chronological image", prompt_text)
            self.assertIn("Candidate image under review", prompt_text)
            self.assertIn("Nearby reference image", prompt_text)
            self.assertIn("Do not use recipe numbers", prompt_text)


if __name__ == "__main__":
    unittest.main()
