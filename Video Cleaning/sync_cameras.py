"""
Steps 4 + 5: line up the 3 cameras and add the video 5 labels.
Run from inside csi-vial-fill-dataset/:   python3 sync_cameras.py
Only reads files. Output: synced.csv (one row per moment in time).
"""
import pandas as pd

TOLERANCE_S = 1.0  # cap_seal frame must be within 1 s of the vial_fill frame


def load(cam):
    df = pd.read_csv(f"{cam}/image_index.csv", parse_dates=["timestamp"])
    df = df[df["video_id"].astype(str) != "unassigned"].copy()  # step 2: drop
    df["video_id"] = df["video_id"].astype(int)
    return df.sort_values("timestamp")


vial = load("vial_fill").rename(columns={"filename": "vial_fill_img"})
tank = load("tank_level").rename(columns={"filename": "tank_level_img"})
cap = load("cap_seal").rename(columns={"filename": "cap_seal_img"})

# vial_fill and tank_level share exact timestamps -> exact match
synced = vial.merge(tank[["timestamp", "tank_level_img"]], on="timestamp", how="left")

# cap_seal -> closest frame in the same video, within the tolerance
cap["cap_seal_time"] = cap["timestamp"]
synced = pd.merge_asof(
    synced.sort_values("timestamp"), cap[["timestamp", "video_id", "cap_seal_img", "cap_seal_time"]],
    on="timestamp", by="video_id", direction="nearest",
    tolerance=pd.Timedelta(seconds=TOLERANCE_S),
)
synced["cap_seal_gap_s"] = (synced["cap_seal_time"] - synced["timestamp"]).dt.total_seconds().round(3)
synced = synced.drop(columns="cap_seal_time")

# step 5: video 5 labels (matched by vial_fill filename)
labels = pd.read_csv("vial_fill/labels/video5_frame_labels.csv").drop(columns="timestamp")
labels = labels.rename(columns={"filename": "vial_fill_img"})
synced = synced.merge(labels, on="vial_fill_img", how="left")

synced.to_csv("synced.csv", index=False)

# summary
print(f"synced.csv: {len(synced)} rows (one per vial_fill frame)\n")
print("Rows per video, and how many have each camera:")
summary = synced.groupby("video_id").agg(
    rows=("vial_fill_img", "size"),
    tank_level=("tank_level_img", "count"),
    cap_seal=("cap_seal_img", "count"),
    labeled=("in_fill", "count"),
)
print(summary.to_string())
print(f"\nLargest cap_seal time gap: {synced['cap_seal_gap_s'].abs().max():.3f} s")
v5 = synced[synced["video_id"] == 5]
print(f"Video 5: {int((v5['in_fill'] == 1).sum())} frames during a fill, "
      f"{int((v5['in_fill'] == 0).sum())} between fills, "
      f"{v5['vial'].nunique()} vials")
