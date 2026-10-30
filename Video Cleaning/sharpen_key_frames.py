"""
Make each key frame CLEAR: look at every video frame (30 per second) in the
moments before the vial leaves, take the still frames (nothing moving),
average up to 15 of them (removes video noise), then sharpen a little.
Colors stay the same (changes average < 1 out of 255).

Fixes: frames caught mid-move, and blur/noise from the camera and video
compression. (The camera itself is low quality, so this improves it but
can't add detail the camera never captured.)

Run from inside csi-vial-fill-dataset/, AFTER video_to_frames.py and
find_key_frames.py vial_fill_from_video:
    python3 sharpen_key_frames.py
Writes:
  key_frames_sharp/            one clear picture per vial (full frame)
  key_frames_zoom/             same, cropped to the vial and enlarged 1.5x
  key_frames_sharp.csv         video, vial, time, sharpness, chronological order
  key_frames_sharp_check.jpg   contact sheet, to eyeball
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAM = "vial_fill"
MOVE_BOX = (250, 195, 390, 260)   # x1, y1, x2, y2: holder area (moves when vial leaves)
VIAL_BOX = (250, 150, 380, 420)   # the vial, used to measure sharpness
MOVING = 2.0      # holder change per frame above this = moving
STILL = 0.5       # holder change per frame below this = still
LOOK_BACK = 1.5   # pick from the last 1.5 s before the vial starts to leave
GAP = 0.2         # skip the last 0.2 s (motion can start slightly before we see it)
AVERAGE = 15      # average up to this many still frames (0.5 s)
SHARPEN = 1.0     # unsharp-mask strength (0 = off)
ZOOM_BOX = (190, 60, 400, 440)    # x1, y1, x2, y2: crop around the vial


def crop(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


keys = pd.read_csv("key_frames_from_video.csv").sort_values(["video_id", "video_s"]).reset_index(drop=True)
keys["chronological_order"] = keys.index + 1
out = Path("key_frames_sharp")
out.mkdir(exist_ok=True)
zoom = Path("key_frames_zoom")
zoom.mkdir(exist_ok=True)
results = []

for vid, group in keys.groupby("video_id"):
    cap = cv2.VideoCapture(f"{CAM}/videos/{CAM}_video_{vid}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    for _, k in group.iterrows():
        # read every frame from 3 s before to 3 s after the rough key frame
        start = max(0, int((k["video_s"] - 3) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, motion, sharp = [], [], []
        prev = None
        for n in range(int(6 * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            box = crop(g, MOVE_BOX)
            motion.append(np.abs(box - prev).mean() if prev is not None else 0.0)
            sharp.append(cv2.Laplacian(crop(g, VIAL_BOX), cv2.CV_32F).var())
            frames.append(fr)
            prev = box
        times = [(start + n) / fps for n in range(len(frames))]

        # when does the vial start to leave? first of 3 moving frames in a row,
        # at or after 1 s before the rough key frame
        onset = len(frames)
        for n in range(len(frames) - 2):
            if times[n] >= k["video_s"] - 1.0 and all(m > MOVING for m in motion[n:n + 3]):
                onset = n
                break

        # candidates: still frames in the last LOOK_BACK seconds before leaving
        t_end = times[onset - 1] - GAP if onset < len(frames) else times[-1]
        cands = [n for n in range(1, onset)
                 if t_end - LOOK_BACK <= times[n] <= t_end and motion[n] < STILL]
        if not cands:   # fall back to anything before leaving
            cands = list(range(max(1, onset - int(LOOK_BACK * fps)), onset)) or [0]
        # average the last still frames before leaving (they show the same scene)
        use = cands[-AVERAGE:]
        best = use[-1]
        avg = np.mean([frames[n].astype(np.float32) for n in use], axis=0)
        img = avg + SHARPEN * (avg - cv2.GaussianBlur(avg, (0, 0), 2.0))
        img = np.clip(img, 0, 255).astype(np.uint8)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sharp_after = cv2.Laplacian(crop(g, VIAL_BOX), cv2.CV_32F).var()

        name = f"{int(k['chronological_order']):03d}_v{vid}_vial{int(k['vial_in_video']):02d}_{times[best]:07.2f}s.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        z = cv2.resize(crop(img, ZOOM_BOX), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(zoom / name), z, [cv2.IMWRITE_JPEG_QUALITY, 98])
        results.append({
            "video_id": vid, "vial_in_video": int(k["vial_in_video"]), "filename": name,
            "chronological_order": int(k["chronological_order"]),
            "video_s": round(times[best], 2),
            "sharpness_before": round(sharp[min(range(len(times)), key=lambda n: abs(times[n] - k["video_s"]))]),
            "sharpness_after": round(sharp_after),
            "frames_averaged": len(use),
        })
    cap.release()
    print(f"video {vid}: {len(group)} key frames sharpened")

res = pd.DataFrame(results)
res.to_csv("key_frames_sharp.csv", index=False)

# contact sheet, zoomed on the vial
cols, w, h = 8, 200, 250
rows = (len(res) + cols - 1) // cols
sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
for n, r in res.iterrows():
    im = cv2.imread(str(out / r["filename"]))[40:440, 170:490]
    im = cv2.resize(im, (w, h))
    cv2.rectangle(im, (0, 0), (90, 16), (0, 0, 0), -1)
    cv2.putText(im, f"v{r['video_id']} #{r['vial_in_video']}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    sheet[(n // cols) * h:(n // cols + 1) * h, (n % cols) * w:(n % cols + 1) * w] = im
cv2.imwrite("key_frames_sharp_check.jpg", sheet)

print(f"\nSharpness (higher = clearer): median {res['sharpness_before'].median():.0f} -> "
      f"{res['sharpness_after'].median():.0f}, lowest {res['sharpness_before'].min():.0f} -> "
      f"{res['sharpness_after'].min():.0f}")
print("Saved: key_frames_sharp/ , key_frames_zoom/ , key_frames_sharp.csv , key_frames_sharp_check.jpg")
