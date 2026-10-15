"""BONES-SEED paths and zone table, shared by the scripts in this folder.

The source dataset under BONES is read-only. Everything derived from it goes under OUT: survey/
(kinematic survey, zone table, green / yellow / red file lists, reference twist flips) and one folder
per batch-retarget version, vN/ (bvh/, ood/, run.json, config snapshot).
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BONES = "/home/zerenluo/Space/bfm_datasets/bones-seed"
OUT = "/home/zerenluo/Space/bfm_datasets/bone-seed_retarget"
BVH_PREFIX = "soma_uniform/bvh/"
BVH_ROOT = f"{BONES}/{BVH_PREFIX}"
OFFICIAL_CSV_ROOT = f"{BONES}/g1/csv"  # official BONES G1 retarget
METADATA = f"{BONES}/metadata/seed_metadata_v004.parquet"
ZONES = f"{OUT}/survey/bones_seed_zones.parquet"
TWIST_FLIPS = f"{OUT}/survey/ref_twist_flips.csv"
# "Frame Time" in the SOMA BVH headers; slightly off 1/120, which decides the resampled frame count.
BVH_FRAME_TIME = 0.008333


def load_zones(path=ZONES):
    """The zone table, with rel = <date>/<clip>: the path under BVH_ROOT and under vN/bvh/ without suffix."""
    df = pd.read_parquet(path)
    df["rel"] = df.move_soma_uniform_path.str.removeprefix(BVH_PREFIX).str.removesuffix(".bvh")
    return df


def actor(rel):
    """Capture session and actor of a clip, e.g. ('230112', 'A117'); mirrored clips (_M) share them."""
    name = rel.split("/")[-1].removesuffix("_M")
    return rel.split("/")[0], name.rsplit("__", 1)[1] if "__" in name else ""
