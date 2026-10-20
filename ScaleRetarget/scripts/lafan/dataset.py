"""LAFAN1 paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

LAFAN holds the 77 BVHs unzipped from lafan1.zip, one take each, {take}_{subject}.bvh (e.g.
walk1_subject1.bvh, 30 fps); it is read-only. split.py cuts every take into parts of about PART_SEC
seconds, OUT/bvh/{take}_{subject}_pNN.bvh, and the parts are the clips: rel = {take}_{subject}_pNN, and the
batch run writes vN/bvh/{rel}.pkl. Everything derived goes under OUT. On another machine, set
LAFAN / LAFAN_OUT.

    python scripts/lafan/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import re
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

LAFAN = os.environ.get("LAFAN", "/home/zerenluo/Space/bfm_datasets/lafan1")
OUT = os.environ.get("LAFAN_OUT", "/home/zerenluo/Space/bfm_datasets/lafan1_retarget")
BVH_ROOT = f"{OUT}/bvh"
ZONES = f"{OUT}/survey/lafan_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/lafan_to_unitree_g1d.yaml"
PART_SEC = 30.0
CLIP_RE = re.compile(r"^(?P<source>(?P<take>(?P<action>[A-Za-z]+)\d+)_(?P<subject>subject\d+))_p(?P<part>\d+)$")


class Lafan(Dataset):
    name = "LAFAN1"
    key = "lafan"
    loader = "lafan"
    source = LAFAN
    bvh_root = BVH_ROOT
    out = OUT
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/lafan_to_unitree_g1d.yaml",
                "config/loader/lafan.yaml", "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml")
    group_label = "演员"

    def load_zones(self, path=ZONES):
        """The zone table (classify.py); rel = {take}_{subject}_pNN, one session per subject."""
        df = pd.read_parquet(path)
        df["filename"] = df.rel
        df["group"] = df["actor"] = df["session"] = df.subject
        df["category"] = df.action
        df["description"] = df.words + "（" + df.source + " 第 " + df.part.astype(str) + " 段）"
        df["seconds"] = df.duration_sec.round(2)
        df["span_expected"] = (df.frame_count - 1) * df.frame_time
        return df


DATASET = Lafan()


def clip_names(rel):
    """source take, take, action, subject and part number of a clip name, with the action spelled out in
    words (fallAndGetUp -> fall and get up) for the text rules."""
    m = CLIP_RE.match(rel)
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m["action"]).lower()
    return {"source": m["source"], "take": m["take"], "action": m["action"], "subject": m["subject"],
            "part": int(m["part"]), "words": words}


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
