"""Per-clip kinematic survey of the LAFAN1 parts (split.py) against the g1d workspace.

Loads each part with LafanLoader at 30 fps, as the batch run does, and writes one row of workspace metrics
(scaleretarget/pipeline/workspace.py) per clip, plus the skeleton (hash of the BVH offsets), the frame
count and time, and the steps where the reference arm pops (scaleretarget/pipeline/ref_pops.py; LAFAN is
captured at 30 fps, so a step is one native frame). Zones are assigned afterwards from these metrics
(classify.py).

    python scripts/lafan/survey.py --out survey_raw.parquet [--workers 20]
"""

import argparse
import glob
import hashlib
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from dataset import BVH_ROOT, CORRESPONDENCE, ROOT
from scaleretarget.pipeline.ref_pops import bvh_rotations, reference_pops, table_arm_bones
from scaleretarget.pipeline.workspace import clip_metrics, rotation_offsets

FPS = 30
OFFSETS = rotation_offsets(CORRESPONDENCE)
_TABLE = yaml.safe_load(open(CORRESPONDENCE))["ik_match_table"]
HIPS, CHEST = _TABLE["pelvis_proxy"][0], _TABLE["torso_link"][0]
HANDS = ("LeftHand", "RightHand")
ARM_BONES = table_arm_bones(_TABLE)
_ctx = {}


def read_header(path):
    """(short hash of the OFFSET lines but the root's, which is where the take starts; frame count; frame time)."""
    offsets, frames = [], None
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok and tok[0] == "OFFSET":
                offsets.append(" ".join(tok[1:]))
            elif tok[:1] == ["Frames:"]:
                frames = int(tok[1])
            elif tok[:2] == ["Frame", "Time:"]:
                return hashlib.md5(" ".join(offsets[1:]).encode()).hexdigest()[:8], frames, float(tok[2])
    raise ValueError(f"no Frame Time in {path}")


def _loader():
    if "loader" not in _ctx:
        from hydra.utils import instantiate
        from loguru import logger
        from scaleretarget.viewer.clip_player import load_config
        logger.remove()
        loader = instantiate(load_config("unitree_g1d", "lafan").loader)
        loader.target_fps = FPS
        _ctx["loader"] = loader
    return _ctx["loader"]


def survey_clip(path):
    row = {"rel": os.path.basename(path)[:-4]}
    try:
        row["skeleton"], row["frame_count"], row["frame_time"] = read_header(path)
        frames, extras = _loader()._load_sample(path)
        pos = lambda b: np.array([f[b][0] for f in frames])
        rot = lambda b: R.from_quat(np.array([f[b][1] for f in frames]), scalar_first=True)
        hands_z = np.stack([pos(h)[:, 2] for h in HANDS], 1)
        row |= clip_metrics(pos(HIPS), rot(HIPS), rot(CHEST), hands_z, extras["fps"], OFFSETS)
        row |= reference_pops(lambda: bvh_rotations(path), frames, extras["fps"], row["frame_time"], ARM_BONES)
        row["error"] = ""
    except Exception as e:  # noqa: BLE001 - one bad file should not kill the sweep
        row["error"] = repr(e)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.chdir(ROOT)

    paths = sorted(glob.glob(f"{BVH_ROOT}/*.bvh"))
    print(f"{len(paths)} parts in {BVH_ROOT}")
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = list(pool.imap_unordered(survey_clip, paths))
    df = pd.DataFrame(rows).sort_values("rel")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    print(f"{(time.time() - t0) / 60:.1f} min, errors: {int((df.error != '').sum())}, written {args.out}")


if __name__ == "__main__":
    main()
