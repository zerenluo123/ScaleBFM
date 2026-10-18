"""HiPHI paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

HIPHI holds the download (data/*.tar.zst, metadata/) and UNPACKED the plain packages
data/{frame}/{lu}/{motion_id}/motion_actor.bvh (unpack.py); both are read-only. UNPACKED/bvh/{frame}/
{motion_id}.bvh are relative symlinks to those BVHs, so a clip is rel = {frame}/{motion_id} and the batch
run writes vN/bvh/{frame}/{motion_id}.pkl. Everything derived goes under OUT. On another machine, set
HIPHI / HIPHI_UNPACKED / HIPHI_OUT.

    python scripts/hiphi/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

HIPHI = os.environ.get("HIPHI", "/home/zerenluo/Space/bfm_datasets/HiPHI")
UNPACKED = os.environ.get("HIPHI_UNPACKED", "/home/zerenluo/Space/bfm_datasets/HiPHI_unpacked")
OUT = os.environ.get("HIPHI_OUT", "/home/zerenluo/Space/bfm_datasets/HiPHI_retarget")
BVH_ROOT = f"{UNPACKED}/bvh"
METADATA = f"{HIPHI}/metadata/hiphi_metadata.csv"
MANIFEST = f"{HIPHI}/data/archive_manifest.csv"
MOTION_TO_PART = f"{HIPHI}/data/motion_to_part.csv"
ZONES = f"{OUT}/survey/hiphi_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/hiphi_to_unitree_g1d.yaml"
# "Frame Time" in most HiPHI BVH headers (some say 0.01111; the survey records each), which decides the
# resampled frame count.
BVH_FRAME_TIME = 0.011111
MIRROR = "__mirror"


class Hiphi(Dataset):
    name = "HiPHI"
    key = "hiphi"
    loader = "hiphi"
    source = UNPACKED
    bvh_root = BVH_ROOT
    out = OUT
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/hiphi_to_unitree_g1d.yaml",
                "config/loader/hiphi.yaml", "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml")
    mirror_suffix = MIRROR
    group_label = "演员"

    def load_zones(self, path=ZONES):
        """The zone table (classify.py); rel = {frame}/{motion_id}, session = skeleton calibration."""
        df = pd.read_parquet(path)
        df["filename"] = df.motion_id
        df["group"] = df["actor"] = df.actor_id
        df["category"] = df.frame_lu
        df["description"] = df.text_annotation
        df["seconds"] = df.duration_sec.round(2)
        df["span_expected"] = (df.frame_count - 1) * df.frame_time.fillna(BVH_FRAME_TIME)
        return df


DATASET = Hiphi()


def read_metadata():
    """hiphi_metadata.csv with rel and the BVH path of each motion's package under UNPACKED/data."""
    meta = pd.read_csv(METADATA, keep_default_na=False)
    meta["rel"] = meta.frame + "/" + meta.motion_id
    meta["package_bvh"] = UNPACKED + "/data/" + meta.frame + "/" + meta.lu + "/" + meta.motion_id + "/motion_actor.bvh"
    return meta


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
