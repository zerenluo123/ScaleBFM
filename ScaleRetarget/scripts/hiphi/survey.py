"""Per-clip kinematic survey of HiPHI BVHs against the g1d workspace.

Loads each clip with HiphiLoader at 30 fps, as the batch run does (rescaled to SOMA Uniform, forearm twist
in the hand, ground shift, bone frames turned to the common convention), and writes one row of workspace
metrics (scaleretarget/pipeline/workspace.py) per clip, plus the skeleton calibration (hash of the BVH
offsets), the frame time, the loader's ground shift and frame turns, and the steps where the reference arm
pops. Zones are assigned afterwards from these metrics (classify.py).

Pops are checked at the capture's own rate, 90 fps (scaleretarget/pipeline/ref_pops.py).

With --torso it writes workspace.torso_metrics instead, for the zones of a retarget that raises the body
to the lowest lift (classify.py --torso).

    python scripts/hiphi/survey.py --out survey_raw.parquet [--limit N] [--workers 20]
    python scripts/hiphi/survey.py --torso --out torso_raw.parquet [--workers 64]
"""

import argparse
import hashlib
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from dataset import CORRESPONDENCE, ROOT, read_metadata
from scaleretarget.pipeline.ref_pops import bvh_rotations, reference_pops, table_arm_bones
from scaleretarget.pipeline.workspace import clip_metrics, rotation_offsets, torso_metrics

FPS = 30
OFFSETS = rotation_offsets(CORRESPONDENCE)
_TABLE = yaml.safe_load(open(CORRESPONDENCE))["ik_match_table"]
HIPS, CHEST = _TABLE["pelvis_proxy"][0], _TABLE["torso_link"][0]
HANDS = ("LeftHand", "RightHand")
ARM_BONES = table_arm_bones(_TABLE)
_ctx = {}


def read_header(path):
    """(short hash of the OFFSET lines, frame time): one hash per capture calibration of an actor; the frame
    time is 0.011111 or 0.01111 and decides the resampled frame count."""
    offsets = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok and tok[0] == "OFFSET":
                offsets.append(" ".join(tok[1:]))
            elif tok[:2] == ["Frame", "Time:"]:
                return hashlib.md5(" ".join(offsets).encode()).hexdigest()[:8], float(tok[2])
    raise ValueError(f"no Frame Time in {path}")


def _loader():
    if "loader" not in _ctx:
        from hydra.utils import instantiate
        from loguru import logger
        from scaleretarget.viewer.clip_player import load_config
        logger.remove()
        loader = instantiate(load_config("unitree_g1d", "hiphi").loader)
        loader.target_fps = FPS
        _ctx["loader"] = loader
    return _ctx["loader"]


def load_clip(path):
    """Loader frames and extras of a clip, and the inputs of clip_metrics / torso_metrics."""
    frames, extras = _loader()._load_sample(path)
    pos = lambda b: np.array([f[b][0] for f in frames])
    rot = lambda b: R.from_quat(np.array([f[b][1] for f in frames]), scalar_first=True)
    hands_z = np.stack([pos(h)[:, 2] for h in HANDS], 1)
    return frames, extras, (pos(HIPS), rot(HIPS), rot(CHEST), hands_z, extras["fps"], OFFSETS)


def torso_clip(path):
    row = {"path": path}
    try:
        row |= torso_metrics(*load_clip(path)[2])
        row["error"] = ""
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)
    return row


def survey_clip(path):
    row = {"path": path}
    try:
        row["skeleton"], row["frame_time"] = read_header(path)
        frames, extras, inputs = load_clip(path)
        row |= clip_metrics(*inputs)
        row |= reference_pops(lambda: bvh_rotations(path), frames, extras["fps"], row["frame_time"], ARM_BONES)
        row |= {
            "turned_frames": ",".join(f"{b}:{k}" for b, k in extras["turned_frames"].items()),
            "ground_shift": float(extras["ground_shift"]), "stands": bool(extras["stands"]), "error": "",
        }
    except Exception as e:  # noqa: BLE001 - one bad file should not kill the sweep
        row["error"] = repr(e)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--paths", nargs="*", default=None)
    ap.add_argument("--torso", action="store_true", help="torso metrics for the raised zones (classify.py --torso)")
    args = ap.parse_args()
    os.chdir(ROOT)
    clip = torso_clip if args.torso else survey_clip

    if args.paths:
        paths = args.paths
    else:
        meta = read_metadata()
        present = meta[meta.package_bvh.map(os.path.isfile)]
        print(f"{len(present)} of {len(meta)} motions unpacked")
        paths = list(present.package_bvh)[: args.limit]

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(clip, paths, chunksize=2), 1):
            rows.append(row)
            if i % 500 == 0 or i == len(paths):
                el = time.time() - t0
                print(f"{i}/{len(paths)}  {el/60:.1f} min  eta {el/i*(len(paths)-i)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    print("errors:", int((df.error != "").sum()), "written", args.out)


if __name__ == "__main__":
    main()
