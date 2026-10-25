"""Assign red / yellow / green zones to Embody3D clips for the g1d retarget (rules: scaleretarget/pipeline/workspace.py).

Embody3D's text is the task in the capture name (crouched sneaking hiding, ascending stairs, ...), the
same for every participant of a capture; conversation tasks are numbered scenes. Falls, floor work and
sitting are caught by the hips height everywhere, and so is standing on something (the hips held above
the lift top). Clips whose reference arm pops on any step are red with the reason 参考数据跳变 (as in
HiPHI), and so are clips with any frame the capture marks as untracked (参考数据缺帧): its pose is not
the person's.

With --torso (torso_survey.py) the zones are for a retarget that raises the body to the lowest lift
(robot.root_height_filter.min_height, workspace.assign_raised_zones): near-floor and seated clips that do
not lie on the floor are kept, and clips holding a lean g1d cannot follow go red.

    python scripts/embody3d/classify.py --survey survey_raw.parquet --out_dir DIR [--torso torso_raw.parquet]
"""

import argparse
import os

import pandas as pd

from dataset import clip_names
from scaleretarget.pipeline.workspace import assign_raised_zones, assign_zones, write_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--torso", default=None, help="torso_survey.py output: zone for the raised retarget")
    args = ap.parse_args()

    df = pd.read_parquet(args.survey)
    df = pd.concat([df, pd.DataFrame([clip_names(r) for r in df.rel], index=df.index)], axis=1)
    df["duration_sec"] = df.frame_count * df.frame_time
    print(f"{len(df)} clips surveyed")

    failed = df.error != ""
    df["zone"], df["reason"] = "red", "骨架不支持"
    words = df.words[~failed]
    df.loc[~failed, "zone"], df.loc[~failed, "reason"] = assign_zones(
        df[~failed], words, words, jump_text=words, seated_text=words)
    pops = ~failed & (df.ref_pops > 0)
    df.loc[pops, "zone"], df.loc[pops, "reason"] = "red", "参考数据跳变"
    lost = ~failed & (df.frac_untracked > 0)
    df.loc[lost, "zone"], df.loc[lost, "reason"] = "red", "参考数据缺帧"
    torso_cols = []
    if args.torso:
        torso = pd.read_parquet(args.torso)
        torso = torso[torso.error == ""].drop(columns="error")
        torso_cols = [c for c in torso.columns if c != "rel"]
        df = df.merge(torso, on="rel", how="left")
        df["zone"], df["reason"], df["raised"] = assign_raised_zones(df)
        torso_cols = ["raised", *torso_cols]

    cols = [
        "rel", "zone", "reason", "subset", "capture", "person", "session_id", "task", "words", "beta0", "duration_sec",
        "frame_count", "frame_time", "frames_30", "frac_untracked", "untracked_run_s", "hips_z_min", "hips_z_p50",
        "hips_z_max", "frac_lift_low", "frac_lift_high", "lift_high_run_s", "frac_floor", "pitch_min", "pitch_max",
        "frac_pitch_back", "frac_pitch_fwd", "base_speed_max", "frac_base_fast", "frac_out_ws", "lift_speed_p95",
        "frac_lift_fast", "hand_z_min", "hand_z_max", "base_path_m", "ref_twist_flips", "ref_hand_flips", "ref_pops",
        "ref_pop_peak", *torso_cols, "error",
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    out = df[cols].sort_values(["zone", "reason", "subset", "rel"])
    out.to_csv(f"{args.out_dir}/embody3d_zones.csv", index=False, float_format="%.4g")
    out.to_parquet(f"{args.out_dir}/embody3d_zones.parquet")
    write_lists(df.zone, df.rel + ".pkl", args.out_dir)
    print(df.groupby(["zone", "reason"]).agg(clips=("rel", "size"), hours=("duration_sec", lambda s: s.sum() / 3600)).to_string())
    print(df.groupby(["subset", "zone"]).duration_sec.sum().div(3600).unstack(fill_value=0).round(2).to_string())
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
