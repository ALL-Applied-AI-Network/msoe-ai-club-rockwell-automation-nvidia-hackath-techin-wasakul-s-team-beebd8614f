"""
Step 3: find bad images (broken, too dark, too bright, blurry, exact duplicates).
Run from inside csi-vial-fill-dataset/:   python3 check_images.py
It only READS images. Results go to bad_images.csv.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

CAMERAS = ["cap_seal", "tank_level", "vial_fill"]
DARK = 20          # average brightness below this (0-255) = too dark
BRIGHT = 240       # average brightness above this = too bright
BLUR_RATIO = 0.25  # sharpness below 25% of the camera's typical sharpness = blurry

rows = []  # camera, filename, issue, value

for cam in CAMERAS:
    folder = Path(cam) / "images"
    files = sorted(folder.glob("*.jpg"))
    stats = []      # (filename, brightness, sharpness)
    hashes = {}     # md5 -> first filename seen

    for f in files:
        # exact duplicate check
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in hashes:
            rows.append([cam, f.name, "duplicate", hashes[h]])
        else:
            hashes[h] = f.name

        # broken file check
        try:
            g = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
        except Exception as e:
            rows.append([cam, f.name, "broken", str(e)])
            continue

        brightness = g.mean()
        # sharpness = variance of the Laplacian (low = blurry)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
               + g[1:-1, :-2] + g[1:-1, 2:])
        stats.append((f.name, brightness, lap.var()))

    typical = np.median([s for _, _, s in stats]) if stats else 0
    for name, b, s in stats:
        if b < DARK:
            rows.append([cam, name, "too_dark", round(float(b), 1)])
        elif b > BRIGHT:
            rows.append([cam, name, "too_bright", round(float(b), 1)])
        if typical and s < BLUR_RATIO * typical:
            rows.append([cam, name, "blurry", round(float(s), 1)])

    n_bad = sum(1 for r in rows if r[0] == cam)
    print(f"{cam}: {len(files)} images checked, {n_bad} flagged "
          f"(typical sharpness {typical:.1f})")

with open("bad_images.csv", "w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["camera", "filename", "issue", "value"])
    w.writerows(rows)

print("\nFlag counts:")
counts = {}
for cam, _, issue, _ in rows:
    counts[(cam, issue)] = counts.get((cam, issue), 0) + 1
for (cam, issue), n in sorted(counts.items()):
    print(f"  {cam:12s} {issue:12s} {n}")
if not rows:
    print("  none — all images look fine")
print("\nDetails saved to bad_images.csv")
