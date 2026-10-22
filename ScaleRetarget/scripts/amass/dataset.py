"""AMASS paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

AMASS holds the downloaded SMPL-X N archives, one per subset ({subset}.tar.bz2); it is read-only. The
GRAB subset is the GRAB capture fitted once more and is left out (scripts/grab/ converts the original).
The other archives are unpacked as they are into OUT/bvh/{subset}/{subject}/{name}_stageii.npz (the
loader's format; the _stagei.npz next to them hold only the body shape and are skipped), and these are
the clips: rel = {subset}/{subject}/{name}_stageii, and the batch run writes vN/bvh/{rel}.pkl. Everything
derived goes under OUT. On another machine, set AMASS / AMASS_OUT.

    python scripts/amass/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import re
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

AMASS = os.environ.get("AMASS", "/home/zerenluo/Space/bfm_datasets/amass")
OUT = os.environ.get("AMASS_OUT", "/home/zerenluo/Space/bfm_datasets/amass_retarget")
BVH_ROOT = f"{OUT}/bvh"
ZONES = f"{OUT}/survey/amass_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/amass_to_unitree_g1d.yaml"
SKIP_ARCHIVES = ("GRAB.tar.bz2",)


class Amass(Dataset):
    name = "AMASS"
    key = "amass"
    loader = "amass"
    source = AMASS
    bvh_root = BVH_ROOT
    suffix = ".npz"
    out = OUT
    correspondence = CORRESPONDENCE
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/amass_to_unitree_g1d.yaml",
                "config/correspondence/optim_shape/amass_to_unitree_g1.yaml", "config/loader/amass.yaml",
                "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml", "assets/unitree_g1/g1_mocap_29dof.xml")
    group_label = "子数据集"

    def load_zones(self, path=ZONES):
        """The zone table (classify.py); rel = {subset}/{subject}/{name}_stageii, one session per subject."""
        df = pd.read_parquet(path)
        df["filename"] = df.rel
        df["group"] = df["category"] = df.subset
        df["actor"] = df["session"] = df.subset + "/" + df.subject
        df["description"] = df.words
        # A clip whose npz cannot be read has no header; it is red (骨架不支持) and never converted.
        df["seconds"] = df.duration_sec.fillna(0).round(2)
        df["span_expected"] = ((df.frame_count - 1) * df.frame_time).fillna(0)
        return df


DATASET = Amass()


def words(text):
    """Lower-case words of a folder or file name: camelCase, underscores, dashes and digits split, single
    letters (take labels like B3, c3d) dropped."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return " ".join(w for w in re.sub(r"[^A-Za-z]+", " ", text).lower().split() if len(w) > 1)


def clip_names(rel):
    """Subset, subject and sequence of a clip, with the subject folder and sequence name in words for
    the text rules (ACCAD/Female1Walking_c3d/B3 - walk1_stageii -> female walking walk)."""
    subset, subject, name = rel.split("/", 2)
    name = name.removesuffix("_stageii")
    return {"subset": subset, "subject": subject, "sequence": name, "words": words(f"{subject} {name}")}


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
