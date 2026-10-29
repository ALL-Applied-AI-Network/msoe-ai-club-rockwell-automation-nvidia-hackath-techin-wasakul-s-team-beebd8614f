# Twin Data Generator

Makes **labeled training images** of the vial fill station with the digital twin:
random recipes, carryover discoloration, underfills and lighting changes (no overfills).
Every image comes with its correct answer, in the label format `train_cosmos_lora.py` reads.

**Try it first (any computer, no GPU, ~1 second):**
```bash
pip install numpy pillow usd-core
python3 demo.py --twin <csi_digital_twin_v1> --train-script <train_cosmos_lora.py>
```
`--twin` and `--train-script` are optional. The demo prints the color check, a random
plan, 30 tests, and the exact prompt/answer the training script will learn from.

## How it works
```
real photos (video 5) --calibrate_colors.py--> color_model.json   (already done)
recipes + random runs --make_plan.py--------> plan.csv           (any computer)
plan.csv + twin -------render_plan.py-------> images/ + labels.csv  (Spark, Isaac Sim)
labels.csv ------------train_cosmos_lora.py-> trained adapter     (Lydia)
```

## Files
| File | What it does |
|---|---|
| `recipes.py` | **The equations**: recipe → color (dye mixing, fitted to real photos), carryover mix, mL → fill height |
| `color_model.json` | Fitted numbers (made by `calibrate_colors.py`) |
| `calibrate_colors.py` | Measures the 30 real labeled vials and fits the color numbers; also writes the real test set |
| `make_plan.py` | **Randomization**: runs of recipes (3 vials each), carryover on vial 1, underfills, lighting (no overfills) |
| `twin_edit.py` | Sets color / fill height / lights on the twin (USD; same code in Isaac Sim and tests) |
| `render_plan.py` | Renders every plan row with Isaac Sim → `images/` + `labels.csv` |
| `preview.py` | Plan preview picture, and render-vs-real-camera comparison |
| `tests.py` | 30 checks (math, plan, twin edits, compatibility with Lydia's script) |
| `demo.py` | Runs everything above to prove it works |
| `REVIEW_NOTES.md` | Comments on `train_cosmos_lora.py` (no code changed) |
| `data/real_vial_colors.csv` | Measured real colors (calibration data) |
| `data/real_test_labels.csv` | The 30 real vials in the training label format (for testing, not training) |

## What the real data showed
- The equations reproduce all 10 real recipe colors (worst error 18/255).
- **Carryover is strong**: real first vials hold **26-100%** old liquid (median 53%), not 5-30%. The plan uses the real range.
- Orange → blue carryover comes out **green**, like the Dataset 2 page. Dark → dark (8→9) isn't visible, so it isn't labeled contaminated.
- The twin's fill animation is Recipe 7 (50 mL blue + 20 mL clear), and height is linear in mL. That is how mL → fill height works.

## On the Spark (Isaac Sim)
```bash
python3 make_plan.py --runs 40 --out plan.csv                     # ~500 vials
<isaac-sim>/python.sh render_plan.py --twin <csi_digital_twin_v1> --plan plan.csv --out <out> --limit 5
python3 preview.py --compare <out> --out compare.png             # renders next to real photos
<isaac-sim>/python.sh render_plan.py --twin <csi_digital_twin_v1> --plan plan.csv --out <out>
python train_cosmos_lora.py --labels <out>/labels.csv --img-root <out>
```
Run `preview.py --compare` from `csi-vial-fill-dataset/` (it uses `key_frames_sharp/`).
If renders look too pale or too strong, change `twin_strength` in `color_model.json`, remake the plan, and render again.

## Not tested yet (needs the GPU)
`render_plan.py` itself. It follows the twin's own `render_example.py`, and its scene edits pass
the tests, but nobody has run Isaac Sim with it yet. Do the `--limit 5` run first.
