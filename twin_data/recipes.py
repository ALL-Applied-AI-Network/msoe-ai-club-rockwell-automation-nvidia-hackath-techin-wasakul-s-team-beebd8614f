"""
The equations: recipe -> liquid color, carryover mix, mL -> fill height.

Everything here is plain Python + numpy (no GPU, no Isaac Sim), so it can be
tested anywhere. The color numbers come from color_model.json, which
calibrate_colors.py fits from the REAL labeled vials (video 5).

How the color math works (Beer-Lambert, the standard dye-mixing law):
  1. Each dye tank adds "absorbance" in proportion to its concentration:
        c_orange = T200 mL / total mL      c_blue = T400 mL / total mL
        A[channel] = base + k_orange[channel] * c_orange + k_blue[channel] * c_blue
     (T100 is clear water: it only dilutes, which lowers both concentrations.)
  2. Color seen = empty_vial_color * exp(-A)       (in linear light)
  3. Carryover (first vial after a recipe change): leftover liquid from the
     previous recipe mixes in. With alpha = share of old liquid:
        c_first = alpha * c_previous + (1 - alpha) * c_current
     Mixing concentrations (not RGB) is what makes blue + orange look GREEN,
     like the real vial 7 on the Dataset 2 page. RGB averaging would give gray.
"""
import colorsys
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Recipe table (dataset README; identical to train_cosmos_lora.py) ----
RECIPES = {
    5: {100: 10, 200: 10, 400: 10}, 6: {100: 30, 200: 10},
    7: {100: 20, 400: 50},          8: {100: 20, 200: 20, 400: 20},
    9: {100: 10, 200: 35, 400: 35}, 10: {100: 45, 200: 15, 400: 15},
    11: {100: 10, 200: 10, 400: 50}, 12: {100: 10, 200: 50, 400: 10},
    13: {200: 50, 400: 20},         14: {100: 60, 400: 10},
}
TANKS = (100, 200, 400)          # T100 clear, T200 orange, T400 blue (T300 out of service)
DISPENSE_ORDER = (200, 400, 100)

# ---- Station rules (Dataset 2 page) ----
MAX_TOTAL_ML = 80     # more than this can spill
MIN_COMPONENT_ML = 10
MAX_T200_ML = 50      # more can fault the system

# ---- Fill height (measured from the twin's own fill animation) ----
# The twin animates Recipe 7 (50 mL blue then 20 mL clear). Liquid top height
# (mesh units): bottom 0.116, 50 mL -> 3.585 (frame 368), 70 mL -> 4.966.
# 3.469 / 4.850 = 0.715 = 50/70, so height is linear in mL.
TWIN_FRAME = 686            # finished vial, no dispensing stream visible
LIQ_BOTTOM = 0.116
LIQ_UNITS_PER_ML = 4.85 / 70.0
LIQ_TOP_AT_FRAME = 4.954    # liquid top at TWIN_FRAME (= 70 mL)
VIAL_TOP = 7.992            # top of the glass (same units) -> about 113 mL brim-full
MAX_VISUAL_ML = (VIAL_TOP - LIQ_BOTTOM) / LIQ_UNITS_PER_ML - 3   # keep liquid inside the glass


def load_model(path=os.path.join(HERE, "color_model.json")):
    with open(path) as f:
        return json.load(f)


MODEL = load_model() if os.path.exists(os.path.join(HERE, "color_model.json")) else None


def is_valid(ml):
    """True if a recipe obeys the station rules."""
    used = {t: v for t, v in ml.items() if v}
    return (set(used) <= set(TANKS) and 0 < sum(used.values()) <= MAX_TOTAL_ML
            and all(v >= MIN_COMPONENT_ML for v in used.values())
            and used.get(200, 0) <= MAX_T200_ML)


def concentrations(ml):
    """mL per tank -> (c_orange, c_blue), the dye shares of the liquid."""
    total = sum(ml.values())
    return np.array([ml.get(200, 0) / total, ml.get(400, 0) / total])


def carryover(c_current, c_previous, alpha):
    """Dye shares of the first vial after a change; alpha = share of old liquid."""
    return alpha * np.asarray(c_previous) + (1 - alpha) * np.asarray(c_current)


def absorbance(c, model=None):
    m = model or MODEL
    k = np.array(m["k"])                 # rows: base, orange, blue; cols: R, G, B
    return k[0] + c[0] * k[1] + c[1] * k[2]


def photo_rgb(c, model=None):
    """Predicted color (0-255) of the liquid in the REAL vial_fill camera."""
    m = model or MODEL
    lin = np.array(m["empty_vial_linear"]) * np.exp(-absorbance(c, m))
    return np.clip(255 * lin ** (1 / 2.2), 0, 255)


TWIN_RECIPE7_COLOR = np.array([0.006, 0.2, 0.3])   # twin's original liquid diffuseColor


def twin_rgb(c, model=None):
    """Value for the twin liquid material's diffuseColor (0-1).
    The twin's original blue IS Recipe 7 (its animation pours 50 mL blue + 20 mL
    clear), so every other color is Recipe 7's color shifted by the dye
    difference:  twin7 * exp(-strength * (A(c) - A(recipe 7))).
    CHECK ON THE SPARK with preview.py --compare; if renders look too weak or
    too strong, change "twin_strength" in color_model.json (default 1.0)."""
    m = model or MODEL
    k = np.array(m["k"])
    c7 = concentrations(RECIPES[7])
    shift = (np.asarray(c) - c7) @ k[1:]
    return np.clip(TWIN_RECIPE7_COLOR * np.exp(-m.get("twin_strength", 1.0) * shift), 0.0005, 1)


def color_name(rgb):
    """Rough color word for a predicted/real camera color (used as observed_color)."""
    r, g, b = [float(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 20:
        return "dark"
    if (mx - mn) / mx < 0.2:
        return "clear" if mx > 60 else "dark"
    h = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] * 360
    for limit, name in ((45, "orange"), (70, "yellow"), (165, "green"),
                        (190, "teal"), (260, "blue"), (330, "purple")):
        if h < limit:
            return name
    return "red"


def color_distance(c1, c2, model=None):
    """How different two liquids look in the real camera (0-255 RGB distance)."""
    return float(np.linalg.norm(photo_rgb(c1, model) - photo_rgb(c2, model)))


def fill_scale(total_ml):
    """Z-scale for the twin's liquid mesh (at TWIN_FRAME) to show total_ml."""
    return (LIQ_BOTTOM + LIQ_UNITS_PER_ML * total_ml) / LIQ_TOP_AT_FRAME
