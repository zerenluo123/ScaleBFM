"""BONES-SEED paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

The source dataset under BONES is read-only. Everything derived from it goes under OUT: survey/
(kinematic survey, zone table, green / yellow / red file lists, reference twist flips) and one folder
per batch-retarget version, vN/ (bvh/, ood/, run.json, config snapshot).

    python scripts/bones_seed/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

BONES = os.environ.get("BONES_SEED", "/home/zerenluo/Space/bfm_datasets/bones-seed")
OUT = os.environ.get("BONES_SEED_OUT", "/home/zerenluo/Space/bfm_datasets/bone-seed_retarget")
BVH_PREFIX = "soma_uniform/bvh/"
BVH_ROOT = f"{BONES}/{BVH_PREFIX}"
OFFICIAL_CSV_ROOT = f"{BONES}/g1/csv"  # official BONES G1 retarget
METADATA = f"{BONES}/metadata/seed_metadata_v004.parquet"
ZONES = f"{OUT}/survey/bones_seed_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/soma_to_unitree_g1d.yaml"
# "Frame Time" in the SOMA BVH headers; slightly off 1/120, which decides the resampled frame count.
BVH_FRAME_TIME = 0.008333


class BonesSeed(Dataset):
    name = "BONES-SEED"
    key = "bones_seed"
    loader = "soma"
    source = BONES
    bvh_root = BVH_ROOT
    out = OUT
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/soma_to_unitree_g1d.yaml",
                "config/loader/soma.yaml", "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml")
    official_csv_root = OFFICIAL_CSV_ROOT
    mirror_suffix = "_M"
    group_label = "日期"

    def load_zones(self, path=ZONES):
        """The zone table, with rel = <date>/<clip>: the path under BVH_ROOT and under vN/bvh/ without suffix."""
        df = pd.read_parquet(path)
        df["rel"] = df.move_soma_uniform_path.str.removeprefix(BVH_PREFIX).str.removesuffix(".bvh")
        df["group"] = df.rel.str.split("/").str[0]
        df["description"] = df.content_short_description
        df["span_expected"] = (df.frames_120 - 1) * BVH_FRAME_TIME
        session_actor = df.rel.map(actor)
        df["session"], df["actor"] = session_actor.str[0], session_actor.str[1]
        return df


DATASET = BonesSeed()


def load_zones(path=ZONES):
    return DATASET.load_zones(path)


def actor(rel):
    """Capture session and actor of a clip, e.g. ('230112', 'A117'); mirrored clips (_M) share them."""
    name = rel.split("/")[-1].removesuffix("_M")
    return rel.split("/")[0], name.rsplit("__", 1)[1] if "__" in name else ""


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
