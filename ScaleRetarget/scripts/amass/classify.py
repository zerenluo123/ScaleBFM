"""Assign red / yellow / green zones to AMASS clips for the g1d retarget (rules: scaleretarget/pipeline/workspace.py).

AMASS has no text annotation; the text rules read the subject folder and file name in words, which
describe the motion in some subsets (ACCAD, KIT, EyesJapanDataset, ...) and are numbers in others (CMU).
Falls, floor work and sitting are caught by the hips height everywhere, and so is standing on something
(the hips held above the lift top) where the name does not say it. Clips whose reference arm pops on
any step are red with the reason 参考数据跳变 (as in HiPHI).

    python scripts/amass/classify.py --survey survey_raw.parquet --out_dir DIR
"""

import argparse
import os

import pandas as pd

from dataset import clip_names
from scaleretarget.pipeline.workspace import assign_zones, write_lists

# EKUT names its takes by code: beam / beamn a raised balance beam, slp / slpd stepping onto a platform,
# wsu / wsd (b, f) stairs up / down. Its push and walking takes stay on the floor.
EKUT_TERRAIN = r"^(?:beamn?|slpd?|ws[ud][bf]?)$"


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
    df["zone"], df["reason"] = "red", "骨架不支持"
    words = df.words[~failed]
    ekut = (df.subset[~failed] == "EKUT") & words.str.match(EKUT_TERRAIN)
    terrain = words.mask(ekut, words + " stairs")
    df.loc[~failed, "zone"], df.loc[~failed, "reason"] = assign_zones(
        df[~failed], words, terrain, jump_text=words, seated_text=words)
    pops = ~failed & (df.ref_pops > 0)
    df.loc[pops, "zone"], df.loc[pops, "reason"] = "red", "参考数据跳变"

    cols = [
        "rel", "zone", "reason", "subset", "subject", "sequence", "words", "gender", "beta0", "duration_sec",
        "frame_count", "frame_time", "frames_30", "hips_z_min", "hips_z_p50", "hips_z_max", "frac_lift_low",
        "frac_lift_high", "lift_high_run_s", "frac_floor", "pitch_min", "pitch_max", "frac_pitch_back", "frac_pitch_fwd",
        "base_speed_max", "frac_base_fast", "frac_out_ws", "lift_speed_p95", "frac_lift_fast", "hand_z_min",
        "hand_z_max", "base_path_m", "ref_twist_flips", "ref_hand_flips", "ref_pops", "ref_pop_peak", "error",
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    out = df[cols].sort_values(["zone", "reason", "subset", "rel"])
    out.to_csv(f"{args.out_dir}/amass_zones.csv", index=False, float_format="%.4g")
    out.to_parquet(f"{args.out_dir}/amass_zones.parquet")
    write_lists(df.zone, df.rel + ".npz", args.out_dir)
    print(df.groupby(["zone", "reason"]).agg(clips=("rel", "size"), hours=("duration_sec", lambda s: s.sum() / 3600)).to_string())
    print(df.groupby(["subset", "zone"]).duration_sec.sum().div(3600).unstack(fill_value=0).round(2).to_string())
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
