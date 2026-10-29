"""
Randomization: make a PLAN of vials to render (one row per image), with all
labels already filled in. No GPU needed. render_plan.py then renders it.

Each "run" imitates the real line: a random sequence of recipes, 3 vials per
recipe (vial_in_batch 1, 2, 3). Then for every vial:
  - carryover: vial 1 after a recipe change gets old liquid mixed in, with
    alpha drawn from the range measured on the real line (color_model.json)
  - contaminated = 1 only if that carryover is VISIBLE (color changes enough)
  - fill defect (optional): underfilled vial (--defect-rate). No overfills:
    the station doesn't overfill, so a vial with the right color just passes.
  - lighting: all twin lights scaled by a random factor (realism)

Usage:
    python3 make_plan.py --runs 40 --seed 1 --out plan.csv

Label columns (first 6 are exactly what train_cosmos_lora.py reads):
  image, recipe_id, prev_recipe_id, vial_in_batch, contaminated, observed_color,
  run_id, vial_index, t100_ml, t200_ml, t400_ml, expected_total_ml,
  actual_total_ml, fill_defect, carryover_alpha, expected_color,
  photo_r, photo_g, photo_b, twin_r, twin_g, twin_b, fill_scale, light_scale
"""
import argparse
import csv
import random

import recipes as R

VISIBLE = 6.0      # color distance (0-255) needed to call carryover visible; real
                   # 8->9 (dark into dark) is 1.3, every visible real case is >= 10.8
COLUMNS = ["image", "recipe_id", "prev_recipe_id", "vial_in_batch", "contaminated",
           "observed_color", "run_id", "vial_index", "t100_ml", "t200_ml", "t400_ml",
           "expected_total_ml", "actual_total_ml", "fill_defect", "carryover_alpha",
           "expected_color", "photo_r", "photo_g", "photo_b", "twin_r", "twin_g", "twin_b",
           "fill_scale", "light_scale"]


def make_plan(runs=40, recipes_per_run=(3, 6), defect_rate=0.15, light=(0.8, 1.2), seed=1):
    rng = random.Random(seed)
    lo, hi = R.MODEL["carryover_alpha_range"]
    ids = sorted(R.RECIPES)
    rows = []
    for run in range(runs):
        seq = []
        for _ in range(rng.randint(*recipes_per_run)):
            seq.append(rng.choice([i for i in ids if not seq or i != seq[-1]]))  # a change each time
        prev = 0
        n = 0
        for rid in seq:
            ml = R.RECIPES[rid]
            c_now = R.concentrations(ml)
            for k in (1, 2, 3):
                n += 1
                alpha = rng.uniform(lo, hi) if (k == 1 and prev) else 0.0
                c = R.carryover(c_now, R.concentrations(R.RECIPES[prev]), alpha) if alpha else c_now
                visible = alpha > 0 and R.color_distance(c, c_now) > VISIBLE

                expected_ml = sum(ml.values())
                defect = ""
                actual_ml = expected_ml
                if rng.random() < defect_rate:
                    defect = "underfill"            # no overfills (team decision)
                    actual_ml = expected_ml * rng.uniform(0.55, 0.85)
                photo, twin = R.photo_rgb(c), R.twin_rgb(c)
                rows.append({
                    "image": f"run{run:03d}_v{n:02d}_r{rid}.png",
                    "recipe_id": rid, "prev_recipe_id": prev, "vial_in_batch": k,
                    "contaminated": int(visible),
                    "observed_color": R.color_name(photo) if visible else "",
                    "run_id": run, "vial_index": n,
                    "t100_ml": ml.get(100, 0), "t200_ml": ml.get(200, 0), "t400_ml": ml.get(400, 0),
                    "expected_total_ml": expected_ml, "actual_total_ml": round(actual_ml, 1),
                    "fill_defect": defect, "carryover_alpha": round(alpha, 3),
                    "expected_color": R.color_name(R.photo_rgb(c_now)),
                    "photo_r": round(photo[0], 1), "photo_g": round(photo[1], 1), "photo_b": round(photo[2], 1),
                    "twin_r": round(twin[0], 4), "twin_g": round(twin[1], 4), "twin_b": round(twin[2], 4),
                    "fill_scale": round(R.fill_scale(actual_ml), 4),
                    "light_scale": round(rng.uniform(*light), 3),
                })
            prev = rid
    return rows


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="number of simulated production runs")
    ap.add_argument("--seed", type=int, default=1, help="same seed = same plan")
    ap.add_argument("--defect-rate", type=float, default=0.15, help="share of vials underfilled")
    ap.add_argument("--out", default="plan.csv")
    a = ap.parse_args()
    rows = make_plan(a.runs, defect_rate=a.defect_rate, seed=a.seed)
    save(rows, a.out)
    n1 = sum(r["vial_in_batch"] == 1 and r["prev_recipe_id"] > 0 for r in rows)
    print(f"{len(rows)} vials planned -> {a.out}")
    print(f"  visible carryover: {sum(r['contaminated'] for r in rows)} of {n1} first vials after a change")
    print(f"  fill defects: {sum(bool(r['fill_defect']) for r in rows)}")
