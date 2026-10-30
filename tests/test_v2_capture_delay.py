#!/usr/bin/env python3
"""Regression test for the two-capture v2 target delay."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from line_controller import LineController, V2_TARGET_DELAY_IMAGES  # noqa: E402


class V2CaptureDelayTest(unittest.TestCase):
    def test_vial_one_waits_for_two_following_reference_images(self):
        self.assertEqual(V2_TARGET_DELAY_IMAGES, 2)

        controller = LineController(mock=True)
        controller.mock = False
        controller.model = object()

        inference_calls = []

        def fake_inference(image_paths):
            inference_calls.append(list(image_paths))
            return "no_clear_carryover_signal", (
                '{"assessment":"no_clear_carryover_signal"}'
            )

        controller._run_v2_inference = fake_inference

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(12):
                path = Path(directory) / f"vial_{index + 1:06d}.png"
                path.touch()
                paths.append(str(path))

            positions = [1, 2, 3] * 3
            for path, position in zip(paths[:9], positions):
                controller.inspect_vial_image(path, 5, 0, position)

            action, vial_one_status = controller.inspect_vial_image(
                paths[9], 6, 5, 1
            )
            self.assertEqual(action, "CONTINUE")
            self.assertIn("2 more following reference images", vial_one_status)
            self.assertEqual(inference_calls, [])

            action, vial_two_status = controller.inspect_vial_image(
                paths[10], 6, 5, 2
            )
            self.assertEqual(action, "CONTINUE")
            self.assertIn("1 more following reference image", vial_two_status)
            self.assertEqual(inference_calls, [])

            action, _ = controller.inspect_vial_image(paths[11], 6, 5, 3)
            self.assertEqual(action, "CONTINUE")
            self.assertEqual(len(inference_calls), 1)

            submitted = inference_calls[0]
            self.assertEqual(len(submitted), 12)
            self.assertEqual(submitted[9], paths[9])
            self.assertEqual(submitted[10], paths[10])
            self.assertEqual(submitted[11], paths[11])

    def test_contamination_report_attaches_image_from_two_vials_earlier(self):
        controller = LineController(mock=True)
        controller.mock = False
        controller.model = object()
        controller._run_v2_inference = lambda image_paths: (
            "potential_carryover",
            '{"assessment":"potential_carryover"}',
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(12):
                path = Path(directory) / f"vial_{index + 1:06d}.png"
                path.touch()
                paths.append(str(path))

            output = io.StringIO()
            status = None
            with redirect_stdout(output):
                for index, (path, position) in enumerate(
                    zip(paths, [1, 2, 3] * 4)
                ):
                    current_recipe = 5 if index < 9 else 6
                    previous_recipe = 0 if index < 9 else 5
                    status = controller.process_vial(
                        path,
                        current_recipe,
                        previous_recipe,
                        position,
                    )

            self.assertEqual(status, "CONTAMINATION_REPORTED")
            self.assertEqual(controller.last_report["image_path"], paths[9])
            self.assertEqual(controller.last_report["type"], "contamination")
            self.assertIn(f"[REPORT IMAGE] {paths[9]}", output.getvalue())
            self.assertNotIn(f"[REPORT IMAGE] {paths[11]}", output.getvalue())


if __name__ == "__main__":
    unittest.main()

