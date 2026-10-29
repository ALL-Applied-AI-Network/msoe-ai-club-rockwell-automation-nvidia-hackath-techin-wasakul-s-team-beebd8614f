"""
Render every row of plan.csv with the digital twin (Isaac Sim, headless, GPU).
Writes one 640x480 image per vial plus labels.csv for train_cosmos_lora.py.

Run with Isaac Sim's Python on the Spark (or Brev), from this folder:
    <isaac-sim>/python.sh render_plan.py --twin /workspace/csi_digital_twin_v1 \
        --plan plan.csv --out /workspace/data/twin_run1 --limit 5      # test 5 first
Then train:
    python train_cosmos_lora.py --labels /workspace/data/twin_run1/labels.csv \
        --img-root /workspace/data/twin_run1

Safe to stop and rerun: images that already exist are skipped.
Based on the twin's own tools/render_example.py (same render settings).
"""
import argparse
import csv
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--twin", required=True, help="folder csi_digital_twin_v1 (unzipped)")
ap.add_argument("--plan", default=os.path.join(HERE, "plan.csv"))
ap.add_argument("--out", required=True, help="output folder (images/ + labels.csv)")
ap.add_argument("--limit", type=int, default=0, help="render only the first N rows (0 = all)")
ap.add_argument("--subframes", type=int, default=32, help="path-tracing quality (twin default 32)")
ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim argument (repeatable)")
a = ap.parse_args()

stage_file = os.path.join(a.twin, "stations", "fill_station", "fill_station.usda")
if not os.path.isfile(stage_file):
    ap.error(f"not found: {stage_file}  (point --twin at the unzipped csi_digital_twin_v1 folder)")
with open(a.plan, newline="") as f:
    plan = list(csv.DictReader(f))
if a.limit:
    plan = plan[: a.limit]
img_dir = os.path.join(a.out, "images")
os.makedirs(img_dir, exist_ok=True)
labels_path = os.path.join(a.out, "labels.csv")
done = set()
if os.path.exists(labels_path):
    with open(labels_path, newline="") as f:
        done = {r["image"] for r in csv.DictReader(f)}

from isaacsim import SimulationApp  # noqa: E402  (must start before other omni imports)
kit = SimulationApp({"headless": True, "renderer": "PathTracing", "width": 640, "height": 480,
                     "extra_args": a.kit_arg})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import cv2
    import numpy as np
    import omni.timeline
    import omni.usd
    import omni.replicator.core as rep

    import recipes as R
    import twin_edit

    ctx = omni.usd.get_context()
    if not ctx.open_stage(stage_file):
        raise RuntimeError(f"could not open {stage_file}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(10):
        kit.update()
    stage = ctx.get_stage()
    h = twin_edit.setup(stage)

    omni.timeline.get_timeline_interface().set_current_time(R.TWIN_FRAME / stage.GetTimeCodesPerSecond())
    kit.update()
    rp = rep.create.render_product(h["camera"], (640, 480))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    kit.update()

    new_file = not os.path.exists(labels_path)
    with open(labels_path, "a", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=list(plan[0].keys()))
        if new_file:
            w.writeheader()
        for i, row in enumerate(plan):
            name = "images/" + row["image"]
            if name in done:
                continue
            twin_edit.apply_sample(h, row)
            kit.update()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
            img = np.asarray(ann.get_data())[..., :3]
            if img.shape[:2] != (480, 640):
                raise RuntimeError(f"render returned shape {img.shape}; check GPU/RTX startup errors")
            if img.mean() < 5:
                print(f"WARNING: {row['image']} looks black", flush=True)
            if not cv2.imwrite(os.path.join(a.out, name), img[..., ::-1]):
                raise OSError(f"could not write {name}")
            w.writerow({**row, "image": name})
            lf.flush()
            print(f"[{i + 1}/{len(plan)}] {name}", flush=True)
    print(f"done: images in {img_dir}, labels in {labels_path}", flush=True)
except BaseException:
    traceback.print_exc()
    kit.close(exit_code=1)
    raise
kit.close()
