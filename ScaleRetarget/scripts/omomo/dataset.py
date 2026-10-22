"""OMOMO paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

OMOMO holds the downloaded data.tar.gz; it is read-only. Only its two whole-sequence files are used,
data/train_diffusion_manip_seq_joints24.p and data/test_diffusion_manip_seq_joints24.p (SMPL-X body
poses at 30 fps of 17 subjects moving 15 objects; the rest of the archive is object geometry and
training windows). They are extracted into OMOMO_UNPACKED/data/, and convert.py writes every sequence in
the loader's format to OUT/bvh/{seq_name}.pkl (seq_name = sub10_clothesstand_000); these are the clips:
rel = seq_name, and the batch run writes vN/bvh/{rel}.pkl. Everything derived goes under OUT. On another
machine, set OMOMO / OMOMO_UNPACKED / OMOMO_OUT.

    python scripts/omomo/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import re
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

OMOMO = os.environ.get("OMOMO", "/home/zerenluo/Space/bfm_datasets/omomo")
OMOMO_UNPACKED = os.environ.get("OMOMO_UNPACKED", "/home/zerenluo/Space/bfm_datasets/omomo_unpacked")
OUT = os.environ.get("OMOMO_OUT", "/home/zerenluo/Space/bfm_datasets/omomo_retarget")
BVH_ROOT = f"{OUT}/bvh"
ZONES = f"{OUT}/survey/omomo_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/omomo_to_unitree_g1d.yaml"
SEQ_FILES = ("data/train_diffusion_manip_seq_joints24.p", "data/test_diffusion_manip_seq_joints24.p")
CLIP_RE = re.compile(r"^(?P<subject>sub\d+)_(?P<object>[a-z]+)_(?P<take>\d+)$")


class Omomo(Dataset):
    name = "OMOMO"
    key = "omomo"
    loader = "omomo"
    source = OMOMO
    bvh_root = BVH_ROOT
    suffix = ".pkl"
    out = OUT
    correspondence = CORRESPONDENCE
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/omomo_to_unitree_g1d.yaml",
                "config/correspondence/optim_shape/omomo_to_unitree_g1.yaml", "config/loader/omomo.yaml",
                "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml", "assets/unitree_g1/g1_mocap_29dof.xml")
    group_label = "受试者"

    def load_zones(self, path=ZONES):
        """The zone table (classify.py); rel = sub10_clothesstand_000, one session per subject."""
        df = pd.read_parquet(path)
        df["filename"] = df.rel
        df["group"] = df["actor"] = df["session"] = df.subject
        df["category"] = df.object
        df["description"] = df.object + " #" + df["take"]
        df["seconds"] = df.duration_sec.round(2)
        df["span_expected"] = (df.frame_count - 1) * df.frame_time
        return df


DATASET = Omomo()


def clip_names(rel):
    """Subject, object and take number of a clip name (sub10_clothesstand_000 -> sub10, clothesstand, 000)."""
    m = CLIP_RE.match(rel)
    return {"subject": m["subject"], "object": m["object"], "take": m["take"]}


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
