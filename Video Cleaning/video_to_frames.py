"""
Turn the vial_fill videos into pictures (one every 2 seconds), in the SAME
layout as the original photos, so find_key_frames.py works on them unchanged.

Run from inside csi-vial-fill-dataset/:   python3 video_to_frames.py
Only reads the videos. Writes:
  vial_fill_from_video/images/          the pictures
  vial_fill_from_video/image_index.csv  filename, video_s, video_id

video_s = seconds into the video. This matches the video 5 label times.
"""
from pathlib import Path

import cv2
import pandas as pd

CAM = "vial_fill"
EVERY_S = 2.0          # take one picture every 2 seconds (same as the photos)
OUT = Path("vial_fill_from_video")

(OUT / "images").mkdir(parents=True, exist_ok=True)
rows = []
for vid in range(1, 6):
    path = f"{CAM}/videos/{CAM}_video_{vid}.mp4"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"could not open {path}, skipping")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = round(fps * EVERY_S)
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            s = i / fps
            name = f"{CAM}_video{vid}_{s:07.1f}s.jpg"
            cv2.imwrite(str(OUT / "images" / name), frame)
            rows.append({"filename": name, "video_s": round(s, 2), "video_id": vid})
            saved += 1
        i += 1
    print(f"video {vid}: {i / fps:6.0f} s long -> {saved} pictures")

pd.DataFrame(rows).to_csv(OUT / "image_index.csv", index=False)
print(f"\nSaved {len(rows)} pictures to {OUT}/")
