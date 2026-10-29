"""
Edits the digital twin for one planned vial: liquid color, fill height, lighting.

Uses only the USD library (pxr), which Isaac Sim includes, so the exact same
code runs inside Isaac Sim (render_plan.py) and in tests.py on any computer
(pip install usd-core). All edits go into the stage's SESSION layer, so the
twin's files on disk are never changed.

Works on stations/fill_station/fill_station.usda (paths below). For the full
line (csi_line.usda) pass prefix="/CSI_Line".
"""
from pxr import Gf, Sdf, Usd, UsdLux

import recipes as R

ROOT = "/FillStation"
LIQUID = ROOT + "/FillStation_LocalRoot/Fill_liquid"
LIQUID_SHADER = ROOT + "/_materials/USD_blue_liquid_approximation/Principled_BSDF"
CAMERA = ROOT + "/FillStation_LocalRoot/FillCam/FillCam"


def setup(stage, prefix=""):
    """Find the parts we change and remember the original light intensities."""
    h = {
        "liquid_scale": stage.GetPrimAtPath(prefix + LIQUID).GetAttribute("xformOp:scale"),
        "liquid_color": stage.GetPrimAtPath(prefix + LIQUID_SHADER).GetAttribute("inputs:diffuseColor"),
        "camera": prefix + CAMERA,
        "lights": [],
    }
    for name, attr in h.items():
        if name in ("liquid_scale", "liquid_color") and not attr.IsValid():
            raise RuntimeError(f"twin part not found: {name} (is this the fill station file?)")
    for p in Usd.PrimRange(stage.GetPrimAtPath(prefix + ROOT)):
        if p.HasAPI(UsdLux.LightAPI) or p.GetTypeName().endswith("Light"):
            a = UsdLux.LightAPI(p).GetIntensityAttr()
            h["lights"].append((a, a.Get()))
    stage.SetEditTarget(stage.GetSessionLayer())
    return h


def apply_sample(h, row, frame=R.TWIN_FRAME):
    """Set color, fill height and lighting for one plan row (a dict from plan.csv)."""
    t = Usd.TimeCode(frame)
    color = Gf.Vec3f(float(row["twin_r"]), float(row["twin_g"]), float(row["twin_b"]))
    h["liquid_color"].Set(color)
    s = float(row["fill_scale"])
    h["liquid_scale"].Set(Gf.Vec3f(1.0, 1.0, s), t)   # stretch the liquid up/down from its bottom
    k = float(row.get("light_scale") or 1.0)
    for attr, original in h["lights"]:
        attr.Set(original * k)


def read_back(h, frame=R.TWIN_FRAME):
    """What the twin currently has (used by tests)."""
    return {"color": tuple(h["liquid_color"].Get()),
            "scale_z": h["liquid_scale"].Get(Usd.TimeCode(frame))[2],
            "lights": [a.Get() for a, _ in h["lights"]]}
