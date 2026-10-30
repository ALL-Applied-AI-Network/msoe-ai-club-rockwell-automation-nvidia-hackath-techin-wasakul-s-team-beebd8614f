"""Capture the final, fully filled vial frame from a live vial_fill camera.

This is the live counterpart to find_key_frames.py/sharpen_key_frames.py. It
keeps the most recent still frame while a carrier is present and writes that
frame atomically when the carrier begins to leave. The full frame is preserved
because the Cosmos 3 v2 adapter was trained on full images.
"""

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

FILL_BOX = (280, 300, 345, 400)
HOLDER_BOX = (270, 205, 370, 250)
MOVE_BOX = (250, 195, 390, 260)
EMPTY = 100
STILL = 0.5
HOLDER_MOVED = 6
MIN_PRESENT_FRAMES = 3


def crop(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def parse_source(value):
    return int(value) if value.isdigit() else value


def next_sequence(out_dir):
    existing = sorted(out_dir.glob("vial_*.png"))
    if not existing:
        return 1
    try:
        return int(existing[-1].stem.split("_")[-1]) + 1
    except ValueError:
        return len(existing) + 1


def save_capture(frame, out_dir, sequence, manifest_path):
    name = f"vial_{sequence:06d}.png"
    final_path = out_dir / name
    temporary_path = out_dir / f".{name}.tmp.png"
    if not cv2.imwrite(str(temporary_path), frame):
        raise RuntimeError(f"OpenCV could not write {temporary_path}")
    os.replace(temporary_path, final_path)

    new_manifest = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        fields = ["Sequence number", "Image", "Capture timestamp", "vial_in_batch",
                  "batch_number", "v2_assessment_ready"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_manifest:
            writer.writeheader()
        writer.writerow({
            "Sequence number": sequence,
            "Image": name,
            "Capture timestamp": datetime.now(timezone.utc).isoformat(),
            "vial_in_batch": ((sequence - 1) % 3) + 1,
            "batch_number": ((sequence - 1) // 3) + 1,
            "v2_assessment_ready": sequence >= 12 and sequence % 3 == 0,
        })
        handle.flush()
        os.fsync(handle.fileno())
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Capture final vial frames from the live vial_fill camera")
    parser.add_argument("--source", default="0", help="OpenCV camera index, video path, or RTSP URL")
    parser.add_argument("--out", type=Path, default=Path("camera_inbox"))
    parser.add_argument("--show", action="store_true", help="show a local preview; press q to stop")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "capture_manifest.csv"
    sequence = next_sequence(args.out)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video source {args.source!r}")

    previous_move_box = None
    previous_present = False
    present_frames = 0
    last_still_frame = None
    armed = False
    waiting_for_empty = False

    print(f"Watching vial_fill source {args.source!r}; saving final frames to {args.out}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stream ended or returned an unreadable frame")
            height, width = frame.shape[:2]
            if width < MOVE_BOX[2] or height < FILL_BOX[3]:
                raise RuntimeError(
                    f"Camera frame is {width}x{height}; configured regions require at least "
                    f"{MOVE_BOX[2]}x{FILL_BOX[3]}."
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            move_box = crop(gray, MOVE_BOX)
            present = float(crop(gray, HOLDER_BOX).mean()) < EMPTY
            motion = (float(np.abs(move_box - previous_move_box).mean())
                      if previous_move_box is not None else 0.0)

            if waiting_for_empty:
                if not present:
                    waiting_for_empty = False
                previous_move_box = move_box.copy()
                previous_present = present
                continue

            if present:
                present_frames = present_frames + 1 if previous_present else 1
                if motion < STILL:
                    last_still_frame = frame.copy()
                if present_frames >= MIN_PRESENT_FRAMES and last_still_frame is not None:
                    armed = True

            departed = armed and previous_present and (not present or motion > HOLDER_MOVED)
            if departed:
                captured = save_capture(last_still_frame, args.out, sequence, manifest_path)
                print(f"Captured final vial frame: {captured}", flush=True)
                sequence += 1
                armed = False
                waiting_for_empty = True
                present_frames = 0
                last_still_frame = None

            if not present and not previous_present:
                present_frames = 0

            if args.show:
                cv2.rectangle(frame, HOLDER_BOX[:2], HOLDER_BOX[2:], (0, 255, 0), 1)
                cv2.putText(frame, "ARMED" if armed else "WAITING", (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("vial_fill final-frame capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            previous_move_box = move_box.copy()
            previous_present = present
    except KeyboardInterrupt:
        print("Capture stopped by operator.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
