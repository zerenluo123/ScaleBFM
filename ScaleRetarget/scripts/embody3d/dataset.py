"""Embody3D paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

EMBODY3D holds the seven downloaded SMPL-X archives, embody_3d__20251016__{CATEGORY}_smplx.zip; it is
read-only. They hold 16 subsets (multiperson: emotions, location, polyadic, icebreakers; ...), each a folder
of captures, c--{date}--{time}--{participants}--pilot--MotionPrior{N}--{TASK}_{name}--{start}-{end}, with
one folder per participant of SMPL-X arrays at 30 fps (Y up). convert.py writes each participant of each
capture in the loader's format to OUT/bvh/{subset}/{capture}/{person}.pkl, and these are the clips:
rel = {subset}/{capture}/{person}, and the batch run writes vN/bvh/{rel}.pkl. Everything derived goes under
OUT. On another machine, set EMBODY3D / EMBODY3D_OUT.

    python scripts/embody3d/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import re
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

EMBODY3D = os.environ.get("EMBODY3D", "/home/zerenluo/Space/bfm_datasets/embody3d")
OUT = os.environ.get("EMBODY3D_OUT", "/home/zerenluo/Space/bfm_datasets/embody3d_retarget")
BVH_ROOT = f"{OUT}/bvh"
ZONES = f"{OUT}/survey/embody3d_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/embody3d_to_unitree_g1d.yaml"
CAPTURE_RE = re.compile(r"^(?P<session>c--\d{8}--\d{4})--.*--MotionPrior\d*--(?P<task>.+)--\d+-\d+$")


class Embody3d(Dataset):
    name = "Embody3D"
    key = "embody3d"
    loader = "embody3d"
    source = EMBODY3D
    bvh_root = BVH_ROOT
    suffix = ".pkl"
    out = OUT
    correspondence = CORRESPONDENCE
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/embody3d_to_unitree_g1d.yaml",
                "config/correspondence/optim_shape/embody3d_to_unitree_g1.yaml", "config/loader/embody3d.yaml",
                "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml", "assets/unitree_g1/g1_mocap_29dof.xml")
    group_label = "子数据集"

    def load_zones(self, path=ZONES):
        """The zone table (classify.py); rel = {subset}/{capture}/{person}, one session per capture day and
        time (all the captures of one group of participants), one actor per person in it."""
        df = pd.read_parquet(path)
        df["filename"] = df.rel
        df["group"] = df["category"] = df.subset
        df["session"] = df.session_id
        df["actor"] = df.session_id + "/" + df.person
        df["description"] = df.words
        df["seconds"] = df.duration_sec.fillna(0).round(2)
        df["span_expected"] = ((df.frame_count - 1) * df.frame_time).fillna(0)
        return df


DATASET = Embody3d()


def clip_names(rel):
    """Subset, capture, person, session and task of a clip, with the task name in words for the text rules
    (locomotion/c--20250508--0947--...--LOCOMOTION_crouched_sneaking_hiding--012-034/YQW182 ->
    LOCOMOTION_crouched_sneaking_hiding, crouched sneaking hiding). The task's leading tag (LOCOMOTION,
    CHARADES, EMO, ...) names the subset and is left out of the words, and so are numbers."""
    subset, capture, person = rel.split("/")
    m = CAPTURE_RE.match(capture)
    task = m["task"]
    name = task.split("_", 1)[1] if "_" in task else ""
    words = " ".join(w for w in re.sub(r"[^A-Za-z]+", " ", name).lower().split())
    return {"subset": subset, "capture": capture, "person": person, "session_id": m["session"], "task": task,
            "words": words}


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
