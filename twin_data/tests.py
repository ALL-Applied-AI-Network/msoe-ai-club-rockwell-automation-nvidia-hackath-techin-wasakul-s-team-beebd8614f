"""
Checks that the data generator is correct. No GPU needed.

    python3 tests.py                                   # core checks
    python3 tests.py --twin csi_digital_twin_v1 --train-script train_cosmos_lora.py
                                                       # + twin edits + Lydia's script

--twin needs:  pip install usd-core   (Isaac Sim already includes it)
--train-script does NOT need torch: it only borrows the label->prompt code.
"""
import argparse
import ast
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_plan  # noqa: E402
import recipes as R  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def real_clean_colors():
    with open(os.path.join(HERE, "data", "real_vial_colors.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if (int(r["vial"]) - 1) % 3]   # vials 2 and 3
    out = {}
    for r in rows:
        out.setdefault(int(r["recipe_id"]), []).append([float(r[k]) for k in ("filled_r", "filled_g", "filled_b")])
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def plan_as_text_rows(rows):
    """Round-trip through CSV so values are strings, exactly like the training script sees them."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=make_plan.COLUMNS)
    w.writeheader()
    w.writerows(rows)
    buf.seek(0)
    return list(csv.DictReader(buf))


def test_equations():
    print("Equations")
    check("all 10 recipes follow the station rules (<=80 mL, >=10 mL each, T200<=50)",
          all(R.is_valid(m) for m in R.RECIPES.values()))
    real = real_clean_colors()
    worst = max(np.abs(R.photo_rgb(R.concentrations(R.RECIPES[rid])) - real[rid]).max() for rid in real)
    check("predicted color matches real clean vials for every recipe (within 20/255)", worst <= 20,
          f"worst channel error {worst:.0f}")
    near = {frozenset(p) for p in (("dark", "green"), ("teal", "blue"), ("teal", "green"), ("orange", "yellow"))}
    pairs = [(R.color_name(R.photo_rgb(R.concentrations(R.RECIPES[rid]))), R.color_name(real[rid])) for rid in real]
    exact = sum(a == b for a, b in pairs)
    check("predicted color NAME = real photo's name, or a neighbor (dark/green, teal/blue)",
          all(a == b or frozenset((a, b)) in near for a, b in pairs), f"{exact}/{len(pairs)} exact")
    c6, c7 = R.concentrations(R.RECIPES[6]), R.concentrations(R.RECIPES[7])
    mixed = R.carryover(c7, c6, float(np.median(R.MODEL["carryover_alpha_real"])))
    check("carryover orange (R6) into blue (R7) looks GREEN, like the Dataset 2 page",
          R.color_name(R.photo_rgb(mixed)) == "green", R.color_name(R.photo_rgb(mixed)))
    c8, c9 = R.concentrations(R.RECIPES[8]), R.concentrations(R.RECIPES[9])
    check("carryover dark (R8) into dark (R9) is NOT visible (real vial 13 looked normal)",
          R.color_distance(R.carryover(c9, c8, 0.26), c9) < make_plan.VISIBLE)
    check("twin color for Recipe 7 equals the twin's original blue",
          np.allclose(R.twin_rgb(c7), R.TWIN_RECIPE7_COLOR, atol=1e-4))
    t6, t12 = R.twin_rgb(R.concentrations(R.RECIPES[6])), R.twin_rgb(R.concentrations(R.RECIPES[12]))
    check("twin colors for orange recipes are red-dominant (R > G > B)",
          all(t[0] > t[1] > t[2] for t in (t6, t12)))
    print("Fill height")
    check("70 mL (the twin's own animation) = scale 1.0", abs(R.fill_scale(70) - 1) < 0.01,
          f"{R.fill_scale(70):.3f}")
    check("fill height is linear in mL", abs((R.fill_scale(60) - R.fill_scale(30)) -
                                             (R.fill_scale(90) - R.fill_scale(60))) < 1e-9)
    check("fullest real recipe (80 mL) stays inside the glass", R.fill_scale(80) * R.LIQ_TOP_AT_FRAME < R.VIAL_TOP)


def test_plan():
    print("Randomized plan")
    rows = make_plan.make_plan(runs=30, seed=7)
    check("same seed gives the same plan", rows == make_plan.make_plan(runs=30, seed=7))
    check("different seed gives a different plan", rows != make_plan.make_plan(runs=30, seed=8))
    check("has every label column", all(set(make_plan.COLUMNS) == set(r) for r in rows))
    check("vial_in_batch is always 1, 2 or 3", {r["vial_in_batch"] for r in rows} == {1, 2, 3})
    check("only first vials after a recipe change can be contaminated",
          all(r["vial_in_batch"] == 1 and r["prev_recipe_id"] > 0 for r in rows if r["contaminated"]))
    check("observed_color is filled exactly when contaminated",
          all(bool(r["observed_color"]) == bool(r["contaminated"]) for r in rows))
    lo, hi = R.MODEL["carryover_alpha_range"]
    al = [r["carryover_alpha"] for r in rows if r["carryover_alpha"]]
    check("carryover amount stays in the range measured on the real line",
          all(lo - 1e-9 <= x <= hi + 1e-9 for x in al), f"{min(al):.2f}-{max(al):.2f}")
    defects = [r for r in rows if r["fill_defect"]]
    check("no overfilled vials (a vial with the right color and amount just passes)",
          all(r["actual_total_ml"] <= r["expected_total_ml"] for r in rows)
          and all(r["fill_defect"] in ("", "underfill") for r in rows))
    check("underfills really are under", all(r["actual_total_ml"] < r["expected_total_ml"] for r in defects))
    check("no two images share a name", len({r["image"] for r in rows}) == len(rows))
    n_bad = sum(r["contaminated"] for r in rows)
    check("plan has both clean and contaminated vials to learn from", 0 < n_bad < len(rows),
          f"{n_bad} contaminated of {len(rows)}")


def test_training_script(path):
    print("Lydia's training script (labels -> prompt/answer)")
    src = open(path).read()
    keep = [n for n in ast.parse(src).body
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("RECIPES", "TANK", "ORDER")
                                                  for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in ("describe", "build_example", "load_rows"))]
    ns = {"json": __import__("json"), "csv": csv}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    check("her recipe table matches ours", ns["RECIPES"] == R.RECIPES)
    rows = plan_as_text_rows(make_plan.make_plan(runs=10, seed=3))
    try:
        pairs = [ns["build_example"](r) for r in rows]
        ok = True
    except Exception as e:  # noqa: BLE001
        pairs, ok = [], False
        print("   ", e)
    check("her build_example() accepts every row of our labels", ok, f"{len(rows)} rows")
    if ok:
        check("contaminated rows trigger STOP_MACHINE, clean rows trigger CONTINUE", all(
            (("STOP_MACHINE" in a or "carryover from Recipe" in a.lower()) == bool(int(r["contaminated"]))) and
            (("CONTINUE" in a or a.startswith("Yes")) != bool(int(r["contaminated"])))
            for r, (_, a) in zip(rows, pairs)))


def test_twin(twin):
    print("Digital twin edits (USD, no GPU)")
    from pxr import Usd, UsdGeom
    import twin_edit
    stage_file = os.path.join(twin, "stations", "fill_station", "fill_station.usda")
    stage = Usd.Stage.Open(stage_file)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(twin_edit.LIQUID + "/Fill_liquid"))
    top = lambda f: mesh.GetExtentAttr().Get(f)[1][2]  # noqa: E731
    ratio = (top(368) - R.LIQ_BOTTOM) / (top(661) - R.LIQ_BOTTOM)
    check("twin animation: 50 mL blue / 70 mL total height = 50/70 (fill height is linear)",
          abs(ratio - 50 / 70) < 0.01, f"{ratio:.3f} vs {50 / 70:.3f}")
    h = twin_edit.setup(stage)
    check("found the liquid, its material, the camera and the lights", len(h["lights"]) >= 5,
          f"{len(h['lights'])} lights")
    row = make_plan.make_plan(runs=2, seed=5)[3]
    twin_edit.apply_sample(h, row)
    got = twin_edit.read_back(h)
    check("color is set on the twin", np.allclose(got["color"], [row["twin_r"], row["twin_g"], row["twin_b"]],
                                                   atol=1e-4))
    check("fill height is set on the twin", abs(got["scale_z"] - row["fill_scale"]) < 1e-4)
    orig = [o for _, o in h["lights"]]
    check("lights are scaled", np.allclose(got["lights"], np.array(orig) * row["light_scale"], rtol=1e-4))
    check("twin files on disk are NOT changed", not any(l.dirty for l in stage.GetLayerStack(includeSessionLayers=False)))


def run_all(twin=None, train_script=None):
    RESULTS.clear()
    test_equations()
    test_plan()
    if train_script:
        test_training_script(train_script)
    if twin:
        try:
            test_twin(twin)
        except ImportError:
            print("  SKIP  twin tests (pip install usd-core)")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return all(RESULTS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", help="unzipped csi_digital_twin_v1 folder")
    ap.add_argument("--train-script", help="path to train_cosmos_lora.py")
    a = ap.parse_args()
    sys.exit(0 if run_all(a.twin, a.train_script) else 1)
