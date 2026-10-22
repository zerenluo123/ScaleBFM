"""Assign red / yellow / green zones to HiPHI clips for the g1d retarget (rules: scaleretarget/pipeline/workspace.py).

Covers the unpacked clips in the survey. HiPHI describes each motion by its Frame-LU (e.g. Self_motion-crawl)
and one long natural sentence. Terrain and non-upright motion are matched on the Frame-LU; the sentence only
adds non-upright words after the ones it uses in other senses are removed ("rolling shoulders", "the box
lying on the floor", "rise up tall"). Jumps are matched on both. Sitting or lying on the floor is caught
by the hips height. Clips the loader cannot read (BVHs of a reduced skeleton with only Spine and Spine1, so
no Spine2 for the torso target) are red with the reason 骨架不支持. Clips whose reference arm pops on any
step (a twist flip or a hand flip faster than a human arm turns, see survey.py) are red with the reason
参考数据跳变: the pop is in the data, so no retarget can follow it.

    python scripts/hiphi/classify.py --survey survey_raw.parquet --out_dir DIR
"""

import argparse
import os

import pandas as pd

from dataset import MIRROR, read_metadata
from scaleretarget.pipeline.workspace import assign_zones, write_lists

AMBIGUOUS = r"\b(?:rolling|lying|ris(?:e|ing) up)\b"


def assign(df):
    frame_lu = df.frame_lu.str.replace(r"[-_]", " ", regex=True)
    text = df.text_annotation.str.replace(AMBIGUOUS, "", case=False, regex=True)
    return assign_zones(df, frame_lu + " | " + text, frame_lu, frame_lu + " | " + df.text_annotation, frame_lu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    meta = read_metadata()
    sv = pd.read_parquet(args.survey).rename(columns={"path": "package_bvh"})
    df = meta.merge(sv, on="package_bvh", how="inner")
    print(f"{len(df)} of {len(meta)} motions surveyed")

    failed = df.error != ""
    df["zone"], df["reason"] = "red", "骨架不支持"
    df.loc[~failed, "zone"], df.loc[~failed, "reason"] = assign(df[~failed])
    pops = ~failed & (df.ref_pops > 0)
    df.loc[pops, "zone"], df.loc[pops, "reason"] = "red", "参考数据跳变"
    # A mirrored clip has its original's skeleton (offsets mirrored); both count as one capture session.
    original = df.motion_id.str.removesuffix(MIRROR)
    df["session"] = df.actor_id + "/" + original.map(dict(zip(df.motion_id, df.skeleton))).fillna(df.skeleton)

    cols = [
        "motion_id", "rel", "zone", "reason", "frame", "lu", "frame_lu", "actor_id", "mirrored", "is_hoi",
        "object_categories", "text_annotation", "duration_sec", "frame_count", "frame_time", "frames_30", "skeleton", "session",
        "stands", "ground_shift", "hips_z_min", "hips_z_p50", "hips_z_max", "frac_lift_low", "frac_lift_high", "lift_high_run_s",
        "frac_floor", "pitch_min", "pitch_max", "frac_pitch_back", "frac_pitch_fwd", "base_speed_max",
        "frac_base_fast", "frac_out_ws", "lift_speed_p95", "frac_lift_fast", "hand_z_min", "hand_z_max",
        "base_path_m", "ref_twist_flips", "ref_hand_flips", "ref_pops", "ref_pop_peak", "turned_frames", "error",
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    out = df[cols].sort_values(["zone", "reason", "frame_lu", "motion_id"])
    out.to_csv(f"{args.out_dir}/hiphi_zones.csv", index=False, float_format="%.4g")
    out.to_parquet(f"{args.out_dir}/hiphi_zones.parquet")
    write_lists(df.zone, df.rel + ".bvh", args.out_dir)
    print(df.groupby(["zone", "reason"]).size().to_string())
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
