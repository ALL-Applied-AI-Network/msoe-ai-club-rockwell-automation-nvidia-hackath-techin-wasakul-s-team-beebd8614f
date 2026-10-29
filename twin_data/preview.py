"""
Two previews, so you can SEE the data before/after rendering.

1) Plan preview (any computer, no GPU): draws each planned vial with its color,
   fill level and labels, so you can check the plan makes sense.
       python3 preview.py --plan plan.csv --out plan_preview.png --n 24

2) Compare renders to the real camera (after render_plan.py on the Spark):
   puts each rendered recipe next to a REAL photo of the same recipe and prints
   both liquid colors, to check the twin looks like the real line.
       python3 preview.py --compare /workspace/data/twin_run1 \
           --real-dir key_frames_sharp --real-csv key_frames_sharp.csv \
           --recipes-csv vial_fill/labels/video5_vial_recipes.csv --out compare.png
   If renders are clearly too pale/strong, change "twin_strength" in
   color_model.json, re-make the plan, and render again.
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

LIQUID_BOX = (292, 345, 328, 392)   # same spot calibrate_colors.py measures


def show(rgb):
    """Real-camera colors are dark (dark track behind the vial). Brighten ALL by the
    same factor for display, so dark recipes still look dark next to bright ones."""
    return tuple(int(v) for v in np.clip(np.asarray(rgb, float) * 3.5, 0, 255))


def plan_preview(rows, out, n):
    rows = rows[:n]
    W, H, cols = 150, 230, 8
    sheet = Image.new("RGB", (cols * W, ((len(rows) + cols - 1) // cols) * H), (235, 235, 235))
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x, y = (i % cols) * W, (i // cols) * H
        top, bottom, left, right = y + 30, y + 170, x + 55, x + 95
        full = 110.0                                   # glass holds ~110 mL in the twin
        level = bottom - (bottom - top) * float(r["actual_total_ml"]) / full
        line80 = bottom - (bottom - top) * 80 / full
        color = show([float(r["photo_r"]), float(r["photo_g"]), float(r["photo_b"])])
        d.rectangle([left, level, right, bottom], fill=color)
        d.rectangle([left, top, right, bottom], outline=(90, 90, 90), width=2)
        d.line([left - 6, line80, right + 6, line80], fill=(200, 0, 0))   # 80 mL limit
        bad = int(r["contaminated"]) or r["fill_defect"]
        d.text((x + 5, y + 5), f"R{r['recipe_id']} vial {r['vial_in_batch']}  prev R{r['prev_recipe_id']}",
               fill=(0, 0, 0))
        d.text((x + 5, y + 176), f"{r['actual_total_ml']}/{r['expected_total_ml']} mL", fill=(0, 0, 0))
        d.text((x + 5, y + 190), ("CARRYOVER " + r["observed_color"]) if int(r["contaminated"])
               else f"ok {r['expected_color']}", fill=(170, 0, 0) if int(r["contaminated"]) else (0, 110, 0))
        d.text((x + 5, y + 204), r["fill_defect"] or "", fill=(170, 0, 0))
        if bad:
            d.rectangle([x + 1, y + 1, x + W - 2, y + H - 2], outline=(200, 0, 0), width=2)
    sheet.save(out)
    print(f"plan preview of {len(rows)} vials -> {out}  (red line = 80 mL limit, red box = defect)")


def liquid_color(path):
    x1, y1, x2, y2 = LIQUID_BOX
    return np.median(np.asarray(Image.open(path).convert("RGB"))[y1:y2, x1:x2].reshape(-1, 3), axis=0)


def compare(render_dir, real_dir, real_csv, recipes_csv, out):
    with open(os.path.join(render_dir, "labels.csv"), newline="") as f:
        rendered = [r for r in csv.DictReader(f) if not int(r["contaminated"]) and not r["fill_defect"]]
    with open(real_csv, newline="") as f:
        keys = sorted((r for r in csv.DictReader(f) if r["video_id"] == "5"), key=lambda r: float(r["video_s"]))
    with open(recipes_csv, newline="") as f:
        vials = list(csv.DictReader(f))
    real = {}                                          # recipe -> a clean real photo (vial 3 of batch)
    for v, k in zip(vials, keys):
        if int(v["vial"]) % 3 == 0:
            real[int(v["recipe_id"])] = os.path.join(real_dir, k["filename"])
    pairs, seen = [], set()
    for r in rendered:
        rid = int(r["recipe_id"])
        if rid in real and rid not in seen:
            seen.add(rid)
            pairs.append((rid, os.path.join(render_dir, r["image"]), real[rid]))
    if not pairs:
        raise SystemExit("no clean rendered vials of a known recipe found")
    sheet = Image.new("RGB", (2 * 320, len(pairs) * 260), "white")
    d = ImageDraw.Draw(sheet)
    print("recipe   rendered liquid color   real liquid color")
    for i, (rid, rend, re_) in enumerate(pairs):
        for j, p in enumerate((rend, re_)):
            im = Image.open(p).convert("RGB").crop((170, 40, 490, 440)).resize((320, 240))
            sheet.paste(im, (j * 320, i * 260 + 20))
        d.text((5, i * 260 + 4), f"Recipe {rid}: left = twin render, right = real camera", fill=(0, 0, 0))
        print(f"{rid:>6}   {liquid_color(rend).astype(int)}{'':>12}{liquid_color(re_).astype(int)}")
    sheet.save(out)
    print(f"side-by-side -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", help="plan.csv for a plan preview")
    ap.add_argument("--n", type=int, default=24, help="how many vials to draw")
    ap.add_argument("--compare", help="render output folder (has labels.csv + images/)")
    ap.add_argument("--real-dir", default="key_frames_sharp")
    ap.add_argument("--real-csv", default="key_frames_sharp.csv")
    ap.add_argument("--recipes-csv", default="vial_fill/labels/video5_vial_recipes.csv")
    ap.add_argument("--out", default="preview.png")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare, a.real_dir, a.real_csv, a.recipes_csv, a.out)
    elif a.plan:
        with open(a.plan, newline="") as f:
            plan_preview(list(csv.DictReader(f)), a.out, a.n)
    else:
        ap.error("give --plan or --compare")
