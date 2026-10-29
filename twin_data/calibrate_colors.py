"""
Fit the color equations in recipes.py to the REAL labeled vials (video 5).

For each of the 30 labeled vials it measures:
  - the liquid color in its clean key frame (from sharpen_key_frames.py), and
  - the SAME spot when that vial was still empty (start of its fill, from the video),
then fits dye absorbance numbers on the clean vials (vial 2 and 3 of each recipe)
and measures how much old liquid the first vials contain (carryover alpha).

Run from inside csi-vial-fill-dataset/ (after the Video Cleaning scripts):
    python3 calibrate_colors.py --out-dir <this folder>
Writes color_model.json, data/real_vial_colors.csv, and data/real_test_labels.csv
(the 30 real vials in train_cosmos_lora.py's label format, for TESTING the model). Already done once; the
results are included, so you only rerun this if the real data changes.
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--keyframes-csv", default="key_frames_sharp.csv")
ap.add_argument("--keyframes-dir", default="key_frames_sharp")
ap.add_argument("--recipes-csv", default="vial_fill/labels/video5_vial_recipes.csv")
ap.add_argument("--video", default="vial_fill/videos/vial_fill_video_5.mp4")
ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()

LIQUID_BOX = (292, 345, 328, 392)   # x1, y1, x2, y2: low in the vial, below 30 mL line


def median_rgb(bgr):
    x1, y1, x2, y2 = LIQUID_BOX
    return np.median(bgr[y1:y2, x1:x2].reshape(-1, 3), axis=0)[::-1]


keys = pd.read_csv(a.keyframes_csv)
keys = keys[keys.video_id == 5].sort_values("video_s").reset_index(drop=True)
rec = pd.read_csv(a.recipes_csv).fillna(0)
assert len(keys) == len(rec) == 30, "expected 30 key frames and 30 labeled vials for video 5"

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS)
rows = []
for i, r in rec.iterrows():
    P = median_rgb(cv2.imread(os.path.join(a.keyframes_dir, keys.filename[i])))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((r.start_s + 0.2) * fps))   # empty vial, just arrived
    E = median_rgb(np.mean([cap.read()[1] for _ in range(9)], axis=0))
    rows.append({"vial": int(r.vial), "recipe_id": int(r.recipe_id),
                 "t100_ml": r.t100_ml, "t200_ml": r.t200_ml, "t400_ml": r.t400_ml,
                 "total_ml": r.total_ml, "filled_r": P[0], "filled_g": P[1], "filled_b": P[2],
                 "empty_r": E[0], "empty_g": E[1], "empty_b": E[2]})
D = pd.DataFrame(rows)

lin = lambda v: (np.clip(v, 1, 255) / 255.0) ** 2.2
P = lin(D[["filled_r", "filled_g", "filled_b"]].values)
E = lin(D[["empty_r", "empty_g", "empty_b"]].values)
A = -np.log(P / E)
C = np.c_[D.t200_ml / D.total_ml, D.t400_ml / D.total_ml]
X = np.c_[np.ones(len(D)), C]
clean = ((D.vial - 1) % 3 != 0).values          # vials 2 and 3 of each batch
k, *_ = np.linalg.lstsq(X[clean], A[clean], rcond=None)

# carryover: share of previous recipe's liquid in each first vial (video 5 runs 5,6,...,14)
alphas = []
for rid in range(6, 15):
    i = D.index[(D.recipe_id == rid) & ((D.vial - 1) % 3 == 0)][0]
    j = D.index[D.recipe_id == rid - 1][0]
    grid = np.linspace(0, 1, 101)
    err = [np.sum(((np.r_[1, g * C[j] + (1 - g) * C[i]]) @ k - A[i]) ** 2) for g in grid]
    alphas.append(round(float(grid[int(np.argmin(err))]), 2))

model = {
    "k": k.round(4).tolist(),
    "empty_vial_linear": E.mean(0).round(5).tolist(),
    "twin_strength": 1.0,   # tune on the Spark if rendered colors look too weak/strong
    "carryover_alpha_real": alphas,
    "carryover_alpha_range": [min(alphas), max(alphas)],
    "note": "fitted by calibrate_colors.py from video 5 (k rows: base, orange, blue; cols R,G,B)",
}
os.makedirs(os.path.join(a.out_dir, "data"), exist_ok=True)
with open(os.path.join(a.out_dir, "color_model.json"), "w") as f:
    json.dump(model, f, indent=1)
D.round(1).to_csv(os.path.join(a.out_dir, "data", "real_vial_colors.csv"), index=False)

# real test set in the training-script format (image paths relative to csi-vial-fill-dataset/)
real_rgb = D[["filled_r", "filled_g", "filled_b"]].values
import recipes as R  # noqa: E402
test = []
for i, r in D.iterrows():
    k_in_batch = (int(r.vial) - 1) % 3 + 1
    prev = int(r.recipe_id) - 1 if r.recipe_id > 5 else 0    # video 5 ran recipes 5, 6, ..., 14 in order
    clean_rgb = real_rgb[((D.recipe_id == r.recipe_id) & ((D.vial - 1) % 3 != 0)).values].mean(0)
    visible = k_in_batch == 1 and prev and np.linalg.norm(real_rgb[i] - clean_rgb) > 6
    test.append({"image": os.path.join(a.keyframes_dir, keys.filename[i]), "recipe_id": int(r.recipe_id),
                 "prev_recipe_id": prev,
                 "vial_in_batch": k_in_batch, "contaminated": int(bool(visible)),
                 "observed_color": R.color_name(real_rgb[i]) if visible else "", "real_vial": int(r.vial)})
pd.DataFrame(test).to_csv(os.path.join(a.out_dir, "data", "real_test_labels.csv"), index=False)

pred = 255 * (E.mean(0) * np.exp(-(X @ k))) ** (1 / 2.2)
print("recipe   real color (clean vials)   predicted")
for rid in sorted(D.recipe_id.unique()):
    m = (D.recipe_id == rid).values & clean
    print(f"{rid:>6}   {np.round(D[m][['filled_r','filled_g','filled_b']].mean().values).astype(int)}"
          f"{'':>14}{np.round(pred[m].mean(0)).astype(int)}")
print(f"carryover alpha in real first vials: {alphas} (median {np.median(alphas):.2f})")
print("wrote color_model.json and data/real_vial_colors.csv")
