"""Per-clip kinematic survey of HiPHI BVHs against the g1d workspace.

Loads each clip with HiphiLoader at 30 fps, as the batch run does (rescaled to SOMA Uniform, forearm twist
in the hand, ground shift, bone frames turned to the common convention), and writes one row of workspace
metrics (scaleretarget/pipeline/workspace.py) per clip, plus the skeleton calibration (hash of the BVH
offsets), the frame time, the loader's ground shift and frame turns, and the steps where the reference arm
pops. Zones are assigned afterwards from these metrics (classify.py).

A pop is a twist flip or a hand flip (scaleretarget/utils/jump_audit.py) at 30 fps whose spinning bone (upper
arm, hand) turns faster than POP_SPIN at the capture's own rate (90 fps) within the step. Real fast hands
(shaking off water, flapping, wiggling) stay under ~80 rad/s there and turn smoothly over the native frames;
capture pops turn the hand 120-180 deg within one to three native frames (100-300 rad/s, the arm still).

    python scripts/hiphi/survey.py --out survey_raw.parquet [--limit N] [--workers 20]
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
from scaleretarget.pipeline.workspace import clip_metrics, rotation_offsets
from scaleretarget.utils.jump_audit import ARM_LINKS, SIDES, bone_speeds, hand_flips, twist_flips
from scaleretarget.utils.lafan_vendor.extract import read_bvh_multi_channel
from scaleretarget.utils.lafan_vendor.utils import quat_fk

FPS = 30
OFFSETS = rotation_offsets(CORRESPONDENCE)
_TABLE = yaml.safe_load(open(CORRESPONDENCE))["ik_match_table"]
HIPS, CHEST = _TABLE["pelvis_proxy"][0], _TABLE["torso_link"][0]
HANDS = ("LeftHand", "RightHand")
ARM_BONES = [[_TABLE[f"{s}_{link}"][0] for link in ARM_LINKS] for s in SIDES]
POP_SPIN = 100.0  # rad/s at the native frame rate
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


def native_peaks(path, candidates, fps, frame_time):
    """Fastest native-rate turn (rad/s) of each candidate's bone within its 30 fps step.
    candidates: (bone, step) pairs."""
    bvh = read_bvh_multi_channel(path)
    quats = quat_fk(bvh.quats, bvh.pos, bvh.parents)[0]
    turn = {}
    for bone in {b for b, _ in candidates}:
        q = R.from_quat(quats[:, bvh.bones.index(bone)], scalar_first=True)
        turn[bone] = (q[:-1].inv() * q[1:]).magnitude() / frame_time
    peaks = []
    for bone, s in candidates:
        n0 = int(np.floor(s / fps / frame_time))
        n1 = max(min(int(np.ceil((s + 1) / fps / frame_time)), len(turn[bone])), n0 + 1)
        peaks.append(float(turn[bone][n0:n1].max()))
    return np.array(peaks)


def reference_pops(path, frames, fps, frame_time):
    """Counts of 30 fps twist flips and hand flips, and of those that are capture pops (see the module doc),
    with the fastest native turn among them (rad/s, 0 without flips)."""
    candidates = []
    counts = {"ref_twist_flips": 0, "ref_hand_flips": 0}
    for bones in ARM_BONES:
        speeds = bone_speeds(frames, bones, fps)
        for key, flips, bone in (("ref_twist_flips", twist_flips, bones[0]), ("ref_hand_flips", hand_flips, bones[2])):
            steps = np.where(flips(speeds))[0]
            counts[key] += len(steps)
            candidates += [(bone, s) for s in steps]
    peaks = native_peaks(path, candidates, fps, frame_time) if candidates else np.zeros(0)
    return counts | {"ref_pops": int((peaks > POP_SPIN).sum()), "ref_pop_peak": float(peaks.max(initial=0.0))}


def survey_clip(path):
    row = {"path": path}
    try:
        row["skeleton"], row["frame_time"] = read_header(path)
        frames, extras = _loader()._load_sample(path)
        pos = lambda b: np.array([f[b][0] for f in frames])
        rot = lambda b: R.from_quat(np.array([f[b][1] for f in frames]), scalar_first=True)
        hands_z = np.stack([pos(h)[:, 2] for h in HANDS], 1)
        row |= clip_metrics(pos(HIPS), rot(HIPS), rot(CHEST), hands_z, extras["fps"], OFFSETS)
        row |= reference_pops(path, frames, extras["fps"], row["frame_time"])
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
    args = ap.parse_args()
    os.chdir(ROOT)

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
        for i, row in enumerate(pool.imap_unordered(survey_clip, paths, chunksize=2), 1):
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
