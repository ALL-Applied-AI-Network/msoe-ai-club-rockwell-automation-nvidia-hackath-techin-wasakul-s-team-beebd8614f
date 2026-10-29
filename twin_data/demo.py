"""
DEMO: proves the data generator works, end to end, without a GPU (~5 seconds).

    python3 demo.py
    python3 demo.py --twin <csi_digital_twin_v1> --train-script <train_cosmos_lora.py>

What it shows:
  1. the color equations reproduce the real vial colors (from video 5)
  2. a small random plan (3 production runs) with carryover + fill defects
  3. a picture of that plan (demo_out/plan_preview.png)
  4. all tests (twin edits + Lydia's training script, if paths given)
  5. the exact prompt/answer Lydia's script will train on, for a clean and a bad vial
  6. the commands to render for real on the Spark
"""
import argparse
import ast
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_plan  # noqa: E402
import preview  # noqa: E402
import recipes as R  # noqa: E402
import tests  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--twin")
ap.add_argument("--train-script")
ap.add_argument("--out", default=os.path.join(HERE, "demo_out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

print("=" * 70 + "\n1) COLOR EQUATIONS vs REAL CAMERA (clean vials, video 5)\n" + "=" * 70)
real = tests.real_clean_colors()
print("recipe  tanks (mL)                   real RGB        predicted RGB   name")
for rid, ml in R.RECIPES.items():
    p = R.photo_rgb(R.concentrations(ml))
    tanks = ", ".join(f"T{t}:{v}" for t, v in ml.items())
    print(f"{rid:>6}  {tanks:<28} {str(real[rid].round().astype(int)):<15} {str(p.round().astype(int)):<15} "
          f"{R.color_name(p)}")
al = R.MODEL["carryover_alpha_real"]
print(f"\nCarryover measured on the real line: first vials contain {min(al):.0%}-{max(al):.0%} old liquid "
      f"(median {sorted(al)[len(al) // 2]:.0%}).")

print("\n" + "=" * 70 + "\n2) RANDOM PLAN (3 production runs)\n" + "=" * 70)
rows = make_plan.make_plan(runs=3, seed=2)
plan_path = os.path.join(a.out, "plan.csv")
make_plan.save(rows, plan_path)
print("run vial recipe prev  batch  mL(actual/expected)  label")
for r in rows[:12]:
    label = (f"CARRYOVER -> {r['observed_color']} (alpha {r['carryover_alpha']:.2f})" if r["contaminated"]
             else "clean") + (f", {r['fill_defect']}" if r["fill_defect"] else "")
    print(f"{r['run_id']:>3} {r['vial_index']:>4} {r['recipe_id']:>6} {r['prev_recipe_id']:>4} {r['vial_in_batch']:>6}"
          f"   {r['actual_total_ml']:>6}/{r['expected_total_ml']:<6}       {label}")
print(f"... {len(rows)} vials total -> {plan_path}")

print("\n" + "=" * 70 + "\n3) PREVIEW PICTURE\n" + "=" * 70)
with open(plan_path, newline="") as f:
    preview.plan_preview(list(csv.DictReader(f)), os.path.join(a.out, "plan_preview.png"), 24)

print("\n" + "=" * 70 + "\n4) TESTS\n" + "=" * 70)
ok = tests.run_all(a.twin, a.train_script)

if a.train_script:
    print("\n" + "=" * 70 + "\n5) WHAT LYDIA'S SCRIPT WILL TRAIN ON\n" + "=" * 70)
    src = open(a.train_script).read()
    keep = [n for n in ast.parse(src).body
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("RECIPES", "TANK", "ORDER") for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in ("describe", "build_example"))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), a.train_script, "exec"), ns)
    text_rows = tests.plan_as_text_rows(rows)
    for want in (0, 1):
        r = next(r for r in text_rows if int(r["contaminated"]) == want)
        q, ans = ns["build_example"](r)
        print(f"\n[{'contaminated' if want else 'clean'} vial: {r['image']}]\nPROMPT: {q}\nANSWER: {ans}")

print("\n" + "=" * 70 + "\n6) NEXT: RENDER FOR REAL ON THE SPARK\n" + "=" * 70)
print("""  python3 make_plan.py --runs 40 --out plan.csv
  <isaac-sim>/python.sh render_plan.py --twin <csi_digital_twin_v1> --plan plan.csv --out <out> --limit 5
  python3 preview.py --compare <out> --out compare.png        (check renders vs real photos)
  <isaac-sim>/python.sh render_plan.py --twin <csi_digital_twin_v1> --plan plan.csv --out <out>
  python train_cosmos_lora.py --labels <out>/labels.csv --img-root <out>""")
print("\nDEMO " + ("PASSED" if ok else "HAD FAILURES"))
sys.exit(0 if ok else 1)
