"""Assign red / yellow / green zones to BONES-SEED clips for the g1d retarget.

red     not usable on g1d: non-upright, airborne/terrain, sitting or lying on the floor, or mostly near it
yellow  usable but partly outside the g1d workspace (lift stroke, lean-back, base speed)
green   inside the workspace on (almost) every frame

Lift speed is reported but does not decide the zone.

    python scripts/bones_seed/classify.py --survey survey_raw.parquet --out_dir DIR
"""

import argparse
import os

import pandas as pd

from dataset import BONES, BVH_PREFIX, METADATA

NON_UPRIGHT = (
    r"\b(?:crawl|all fours|hands and knees|lying|plank|handstand|standing on hands|flip|rolling|cartwheel"
    r"|somersault|push.?up|swim|prone|superman|flying with wings"
    r"|faint|fall(?:s|ing)? down|falls? (?:to|on|onto) the (?:ground|floor)|rise up|rising up)"
)
TERRAIN = r"climb|ladder|stairs|on a box|onto .*box|from (?:a|the) box|obstacle|vault|parkour"
# Jumps stay in when they are kinematically inside the workspace (in-place jumps, dance hops);
# their vertical speed is handled by the root height filter, not by zoning.
JUMP = r"jump"

FLOOR_FRAC = 0.5
# A deep squat keeps the Hips above ~0.2 m; lower means sitting or lying on the floor.
SEATED_HIPS_Z = 0.15
OUT_WS_FRAC = 0.01


def assign(df):
    text = df.content_body_position.fillna("") + " | " + df.content_type_of_movement.fillna("")
    terrain_text = text + " | " + df.content_short_description.fillna("")
    # Body position / movement type miss some clips (handstand loops, cartwheels tagged "standing");
    # names and short descriptions catch them. Long natural descriptions are left out: "flips a lever".
    upright_text = (
        terrain_text + " | " + df.content_short_description_2.fillna("") + " | "
        + df.content_name.fillna("").str.replace("_", " ") + " | " + df.filename.str.replace("_", " ")
    )
    zone = pd.Series("green", index=df.index)
    reason = pd.Series("", index=df.index)

    out_ws = df.frac_out_ws > OUT_WS_FRAC
    crit = df[["frac_lift_low", "frac_lift_high", "frac_pitch_back", "frac_pitch_fwd", "frac_base_fast"]]
    names = {
        "frac_lift_low": "升降到底",
        "frac_lift_high": "升降到顶",
        "frac_pitch_back": "后仰超限",
        "frac_pitch_fwd": "前俯超限",
        "frac_base_fast": "底盘超速",
    }
    zone[out_ws] = "yellow"
    reason[out_ws] = crit[out_ws].idxmax(axis=1).map(names)

    floor = (df.frac_floor > FLOOR_FRAC) | (df.hips_z_min < SEATED_HIPS_Z)
    zone[floor], reason[floor] = "red", "贴地"
    jump = terrain_text.str.contains(JUMP, case=False, regex=True) & out_ws
    zone[jump], reason[jump] = "red", "跳跃（出工作空间）"
    terrain = terrain_text.str.contains(TERRAIN, case=False, regex=True)
    zone[terrain], reason[terrain] = "red", "地形"
    upright = upright_text.str.contains(NON_UPRIGHT, case=False, regex=True)
    zone[upright], reason[upright] = "red", "非直立"
    return zone, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    meta = pd.read_parquet(METADATA)
    meta["path"] = BONES + "/" + meta.move_soma_uniform_path
    sv = pd.read_parquet(args.survey)
    df = meta.merge(sv, on="path", how="left")
    assert df.frames_30.notna().all(), "survey is missing clips"
    assert (df.error == "").all(), df[df.error != ""].path.head()

    df["zone"], df["reason"] = assign(df)
    df["frames_120"] = df.move_duration_frames
    df["seconds"] = (df.frames_120 / 120).round(2)

    cols = [
        "filename", "zone", "reason", "frames_120", "frames_30", "seconds", "package", "category",
        "content_short_description", "content_body_position", "content_type_of_movement", "is_mirror",
        "hips_z_min", "hips_z_p50", "hips_z_max", "frac_lift_low", "frac_lift_high", "frac_floor",
        "pitch_min", "pitch_max", "frac_pitch_back", "frac_pitch_fwd", "base_speed_max", "frac_base_fast", "frac_out_ws",
        "lift_speed_p95", "frac_lift_fast", "hand_z_min", "hand_z_max", "base_path_m", "move_soma_uniform_path",
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    out = df[cols].sort_values(["zone", "reason", "category", "filename"])
    out.to_csv(f"{args.out_dir}/bones_seed_zones.csv", index=False, float_format="%.4g")
    out.to_parquet(f"{args.out_dir}/bones_seed_zones.parquet")
    # Paths relative to the BVH root, so the lists can be passed as loader.config.file_list.
    bvh_rel = df.move_soma_uniform_path.str.removeprefix(BVH_PREFIX)
    for z in ["red", "yellow", "green"]:
        with open(f"{args.out_dir}/{z}.txt", "w") as f:
            f.write("\n".join(bvh_rel[df.zone == z]) + "\n")
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
