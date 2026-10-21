"""GRAB paths and zone table, shared by the scripts in this folder (layout: scaleretarget/pipeline/dataset.py).

GRAB holds the ten downloaded subject zips, zips/grab__s1.zip ... grab__s10.zip; it is read-only.
They are unzipped as they are into GRAB_UNPACKED/s1 ... s10/{object}_{action}[_N][_Retake].npz (120 fps;
body, hands, object and table). convert.py writes the body and hand poses of every sequence in the
loader's format to OUT/bvh/s1/{name}.pkl, and these are the clips: rel = s1/{name}, and the batch run
writes vN/bvh/{rel}.pkl. Everything derived goes under OUT. On another machine, set GRAB / GRAB_UNPACKED /
GRAB_OUT.

The GRAB subset of AMASS is the same capture fitted once more; it is left out of AMASS and this one is
used. Both play on the body shape fitted to the G1, so the betas they differ in are not used.

    python scripts/grab/dataset.py --shell    # variables for scripts/retarget_version.sh
"""

import os
import re
import sys

import pandas as pd

from scaleretarget.pipeline.dataset import ROOT, Dataset

GRAB = os.environ.get("GRAB", "/home/zerenluo/Space/bfm_datasets/grab")
GRAB_UNPACKED = os.environ.get("GRAB_UNPACKED", "/home/zerenluo/Space/bfm_datasets/grab_unpacked")
OUT = os.environ.get("GRAB_OUT", "/home/zerenluo/Space/bfm_datasets/grab_retarget")
BVH_ROOT = f"{OUT}/bvh"
ZONES = f"{OUT}/survey/grab_zones.parquet"
CORRESPONDENCE = f"{ROOT}/config/correspondence/gmr/grab_to_unitree_g1d.yaml"
CLIP_RE = re.compile(r"^(?P<subject>s\d+)/(?P<object>[a-z0-9]+)_(?P<rest>.+)$")


class Grab(Dataset):
    name = "GRAB"
    key = "grab"
    loader = "grab"
    source = GRAB
    bvh_root = BVH_ROOT
    suffix = ".pkl"
    out = OUT
    correspondence = CORRESPONDENCE
    zones_file = os.path.basename(ZONES)
    snapshot = ("config/robot/unitree_g1d.yaml", "config/correspondence/gmr/grab_to_unitree_g1d.yaml",
                "config/correspondence/optim_shape/grab_to_unitree_g1.yaml", "config/loader/grab.yaml",
                "config/loader/base.yaml", "assets/unitree_g1d/g1d_mocap.xml", "assets/unitree_g1/g1_mocap_29dof.xml")
    group_label = "受试者"

    def load_zones(self, path=ZONES):
        """The zone table (classify.py); rel = s1/{name}, one session per subject."""
        df = pd.read_parquet(path)
        df["filename"] = df.rel
        df["group"] = df["actor"] = df["session"] = df.subject
        df["category"] = df.action
        df["description"] = df.object + " " + df.words
        df["seconds"] = df.duration_sec.round(2)
        df["span_expected"] = (df.frame_count - 1) * df.frame_time
        return df


DATASET = Grab()


def clip_names(rel):
    """Subject, object and action of a clip name (s1/cup_drink_1_Retake -> s1, cup, drink), with the
    action in words for the text rules (takepicture -> take picture)."""
    m = CLIP_RE.match(rel)
    action = " ".join(t for t in m["rest"].split("_") if not t.isdigit() and t != "Retake")
    words = action.replace("takepicture", "take picture")
    return {"subject": m["subject"], "object": m["object"], "action": action, "words": words}


if __name__ == "__main__" and sys.argv[1:] == ["--shell"]:
    print(DATASET.shell())
