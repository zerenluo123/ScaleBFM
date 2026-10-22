"""Assign red / yellow / green zones to OMOMO clips for the g1d retarget (rules: scaleretarget/pipeline/workspace.py).

OMOMO is carrying, pushing and dragging large objects (boxes, tables, chairs, mop, vacuum). It has no text
annotation and the file names give only the object, so no text rule applies; picking things up off the
floor is caught by the hips height. Clips whose reference arm pops on any step are red with the reason
参考数据跳变 (as in HiPHI).

    python scripts/omomo/classify.py --survey survey_raw.parquet --out_dir DIR
"""

import argparse
import os

import pandas as pd

from dataset import clip_names
from scaleretarget.pipeline.workspace import assign_zones, write_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.survey)
    df = pd.concat([df, pd.DataFrame([clip_names(r) for r in df.rel], index=df.index)], axis=1)
    df["duration_sec"] = df.frame_count * df.frame_time
    print(f"{len(df)} clips surveyed")

    failed = df.error != ""
    no_text = pd.Series("", index=df.index)
    df["zone"], df["reason"] = "red", "骨架不支持"
    df.loc[~failed, "zone"], df.loc[~failed, "reason"] = assign_zones(df[~failed], no_text[~failed], no_text[~failed])
    pops = ~failed & (df.ref_pops > 0)
    df.loc[pops, "zone"], df.loc[pops, "reason"] = "red", "参考数据跳变"

    cols = [
        "rel", "zone", "reason", "subject", "object", "take", "gender", "beta0", "duration_sec",
        "frame_count", "frame_time", "frames_30", "hips_z_min", "hips_z_p50", "hips_z_max", "frac_lift_low",
        "frac_lift_high", "lift_high_run_s", "frac_floor", "pitch_min", "pitch_max", "frac_pitch_back", "frac_pitch_fwd",
        "base_speed_max", "frac_base_fast", "frac_out_ws", "lift_speed_p95", "frac_lift_fast", "hand_z_min",
        "hand_z_max", "base_path_m", "ref_twist_flips", "ref_hand_flips", "ref_pops", "ref_pop_peak", "error",
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    out = df[cols].sort_values(["zone", "reason", "object", "rel"])
    out.to_csv(f"{args.out_dir}/omomo_zones.csv", index=False, float_format="%.4g")
    out.to_parquet(f"{args.out_dir}/omomo_zones.parquet")
    write_lists(df.zone, df.rel + ".pkl", args.out_dir)
    print(df.groupby(["zone", "reason"]).agg(clips=("rel", "size"), hours=("duration_sec", lambda s: s.sum() / 3600)).to_string())
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
