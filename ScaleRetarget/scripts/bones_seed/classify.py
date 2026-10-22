"""Assign red / yellow / green zones to BONES-SEED clips for the g1d retarget (rules: scaleretarget/pipeline/workspace.py).

    python scripts/bones_seed/classify.py --survey survey_raw.parquet --out_dir DIR
"""

import argparse
import os

import pandas as pd

from dataset import BVH_PREFIX, METADATA
from scaleretarget.pipeline.workspace import assign_zones, write_lists


def assign(df):
    text = df.content_body_position.fillna("") + " | " + df.content_type_of_movement.fillna("")
    terrain_text = text + " | " + df.content_short_description.fillna("")
    # Body position / movement type miss some clips (handstand loops, cartwheels tagged "standing");
    # names and short descriptions catch them. Long natural descriptions are left out: "flips a lever".
    upright_text = (
        terrain_text + " | " + df.content_short_description_2.fillna("") + " | "
        + df.content_name.fillna("").str.replace("_", " ") + " | " + df.filename.str.replace("_", " ")
    )
    # Only the body position tells sitting apart from a description of a seated person or a chair.
    return assign_zones(df, upright_text, terrain_text, seated_text=df.content_body_position.fillna(""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    meta = pd.read_parquet(METADATA)
    sv = pd.read_parquet(args.survey)
    # Matched below the BONES root, so a survey of another machine's copy still applies.
    sv["move_soma_uniform_path"] = sv.pop("path").str.extract(r"(soma_uniform/.*)$", expand=False)
    df = meta.merge(sv, on="move_soma_uniform_path", how="left")
    assert df.frames_30.notna().all(), "survey is missing clips"
    assert (df.error == "").all(), df[df.error != ""].move_soma_uniform_path.head()

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
    write_lists(df.zone, df.move_soma_uniform_path.str.removeprefix(BVH_PREFIX), args.out_dir)
    print("written", args.out_dir)


if __name__ == "__main__":
    main()
