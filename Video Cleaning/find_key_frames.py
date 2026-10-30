"""
Find the KEY FRAME for each vial: the last vial_fill photo before the vial
leaves the fill station. Only these frames go to the AI.

Run from inside csi-vial-fill-dataset/:
  python3 find_key_frames.py                        (original photos)
  python3 find_key_frames.py vial_fill_from_video   (pictures from video_to_frames.py)
Only reads the dataset. Writes (with "_from_video" added for the video pictures):
  key_frames/            copies of the key frames (one per vial)
  key_frames.csv         one row per vial: video, frame, time, chronological order
  key_frames_check.jpg   contact sheet of all key frames, to eyeball

How it works (640x480 vial_fill camera, fixed position):
  1. HOLDER box: the black vial holder sits here when a carrier is at the
     fill spot. Dark = carrier present, bright white = spot is empty.
  2. FILL box: the vial under the nozzle. A big sudden change here between
     two frames means the carrier moved.
  3. A vial "leaves" when a carrier was there in the frame before AND either
       - the FILL box changes a lot (BIG_CHANGE: carrier caught moving), or
       - it changes a medium amount right after a still frame (QUIET_SWAP:
         the swap happened between two photos, so no motion blur),
     AND the color did not jump UP (that is a fill starting, not a vial leaving)
     AND the black holder itself moved (a pour changes the vial, not the holder).
     The frame before the change is the key frame.
     (If the spot was empty before, the change is just a carrier arriving.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CAM = sys.argv[1] if len(sys.argv) > 1 else "vial_fill"   # folder of pictures
TAG = "" if CAM == "vial_fill" else "_from_video"
FILL_BOX = (280, 300, 345, 400)      # x1, y1, x2, y2: vial under the nozzle
HOLDER_BOX = (270, 205, 370, 250)    # black holder top
BIG_CHANGE = 20   # avg pixel change (0-255) in FILL box = carrier moving
QUIET_SWAP = 12   # smaller change that still counts if the frame before was still
STILL = 5         # change below this = nothing moving
FILL_START = 12   # color (saturation) jumping up by more than this = fill starting
EMPTY = 100       # HOLDER box brighter than this = no carrier at the spot
MOVE_BOX = (250, 195, 390, 260)      # holder + surroundings, to see if the holder moved
HOLDER_MOVED = 6  # avg pixel change in MOVE_BOX above this = holder moved


def crop(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def format_capture_timestamp(seconds):
    total_ms = int(round(float(seconds) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


idx = pd.read_csv(f"{CAM}/image_index.csv")
idx = idx[idx["video_id"].astype(str) != "unassigned"].copy()
idx["video_id"] = idx["video_id"].astype(int)
if "video_s" not in idx.columns:   # original photos: seconds since the first photo
    t = pd.to_datetime(idx["timestamp"])
    idx["video_s"] = (t - t.groupby(idx["video_id"]).transform("min")).dt.total_seconds()
idx = idx.sort_values(["video_id", "video_s"]).reset_index(drop=True)

# measure every frame
change, holder, color, moved_box = [], [], [], []
prev, prev_box, prev_vid = None, None, None
for f, vid in zip(idx["filename"], idx["video_id"]):
    im = Image.open(f"{CAM}/images/{f}")
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    fill = crop(rgb, FILL_BOX)
    box = crop(rgb, MOVE_BOX)
    same_video = prev is not None and vid == prev_vid
    moved_box.append(np.abs(box - prev_box).mean() if same_video else 0.0)
    holder.append(crop(rgb, HOLDER_BOX).mean())
    color.append(crop(np.asarray(im.convert("HSV"), dtype=np.float32), FILL_BOX)[..., 1].mean())
    change.append(np.abs(fill - prev).mean() if same_video else 0.0)
    prev, prev_box, prev_vid = fill, box, vid
idx["change"] = change
idx["holder"] = holder
idx["color"] = color
idx["holder_move"] = moved_box
idx["carrier_here"] = idx["holder"] < EMPTY

# find departures
keys = []
for vid, g in idx.groupby("video_id"):
    g = g.reset_index(drop=True)
    last_hit = -10
    for i in range(1, len(g)):
        c, c_prev = g["change"][i], g["change"][i - 1]
        moved = c > BIG_CHANGE or (c > QUIET_SWAP and c_prev <= STILL)
        fill_starting = g["color"][i] - g["color"][i - 1] > FILL_START
        nxt = g["holder_move"][i + 1] if i + 1 < len(g) else 0.0
        holder_moved = max(g["holder_move"][i], nxt) > HOLDER_MOVED
        if moved and holder_moved and not fill_starting and g["carrier_here"][i - 1]:
            hit, last_hit = i - last_hit > 3, i   # one departure = one key frame
            if not hit:
                continue
            k = g.iloc[i - 1]
            keys.append({
                "video_id": vid,
                "vial_in_video": 0,
                "filename": k["filename"],
                "video_s": round(k["video_s"], 1),
            })

keys = pd.DataFrame(keys)
keys = keys.sort_values(["video_id", "video_s"]).reset_index(drop=True)
keys["vial_in_video"] = keys.groupby("video_id").cumcount() + 1
keys["chronological_order"] = keys.index + 1
keys["Sequence number"] = keys["chronological_order"].astype(int)
keys["Image"] = keys["Sequence number"].apply(lambda n: f"vial_{n:06d}.png")
keys["Capture timestamp"] = keys["video_s"].apply(format_capture_timestamp)
keys["Previous image"] = keys["Image"].shift(1).fillna("")
keys.to_csv(f"key_frames{TAG}.csv", index=False)

out = Path(f"key_frames{TAG}")
out.mkdir(exist_ok=True)
for _, r in keys.iterrows():
    Image.open(f"{CAM}/images/{r['filename']}").save(out / r["Image"], format="PNG")

# contact sheet to eyeball the results
cols, w, h = 8, 160, 120
rows = (len(keys) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), "white")
for n, (_, r) in enumerate(keys.iterrows()):
    im = Image.open(f"{CAM}/images/{r['filename']}").resize((w, h))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 80, 12], fill="black")
    d.text((2, 1), f"#{r['Sequence number']} v{r['video_id']}", fill="yellow")
    sheet.paste(im, ((n % cols) * w, (n // cols) * h))
sheet.save(f"key_frames{TAG}_check.jpg")

print(f"Found {len(keys)} key frames (one per vial that left the station)\n")
print(keys.groupby("video_id").size().rename("vials found").to_string())
print(f"\nSaved: key_frames{TAG}/ , key_frames{TAG}.csv , key_frames{TAG}_check.jpg")
