"""Per-clip kinematic survey of BONES-SEED SOMA Uniform BVHs against the g1d workspace.

Parses only the joints needed (hips, spine chain, arms), downsamples to 30 fps, and writes one row
of metrics per clip. Zones are assigned afterwards from these metrics, so thresholds can be changed
without re-reading the 277 GB of BVH text.

    python scripts/bones_seed/survey.py --out /path/to/survey.parquet [--limit N] [--workers 20]
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from dataset import BONES, METADATA

SRC_FPS = 120
STRIDE = 4  # 120 fps -> 30 fps, same rate the retarget runs at
FPS = SRC_FPS / STRIDE

JOINTS = [
    "Root", "Hips", "Spine1", "Spine2", "Chest",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]

# g1d: pelvis_proxy height with the lift at 0, and the total lift stroke. With Hips scale 1.0 the
# IK puts the lift at (human Hips z - LIFT_BASE).
LIFT_BASE = 0.801
LIFT_STROKE = 0.42
LIFT_TOL = 0.02
FLOOR_HIPS_Z = 0.55
PITCH_MIN_DEG = -2.5  # Yaw_Joint (forward pitch) range is -0.04363..2.3562 rad
PITCH_MAX_DEG = 135.0
LIFT_SPEED = 0.06
BASE_SPEED = 1.5

# SOMA Y-up centimeters -> Z-up meters, identical to SomaLoader.
Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def _wxyz(q):
    return R.from_quat([q[1], q[2], q[3], q[0]])


# Rotation offsets of the Hips -> pelvis_proxy and Chest -> torso_link entries in soma_to_unitree_g1d.yaml.
HIPS_OFFSET = _wxyz([-0.4992, -0.4992, -0.5008, -0.5008])
CHEST_OFFSET = _wxyz([-0.4991, -0.4991, -0.5009, -0.5009])


def parse_header(path):
    names, parents, offsets, channels = [], [], [], []
    stack = []
    with open(path) as f:
        for n_lines, line in enumerate(f, 1):
            tok = line.split()
            if not tok:
                continue
            if tok[0] in ("ROOT", "JOINT"):
                names.append(tok[1])
                parents.append(stack[-1] if stack else -1)
            elif tok[0] == "End":
                names.append(None)
                parents.append(stack[-1])
            elif tok[0] == "{":
                stack.append(len(names) - 1)
            elif tok[0] == "}":
                stack.pop()
            elif tok[0] == "OFFSET":
                offsets.append([float(v) for v in tok[1:4]])
            elif tok[0] == "CHANNELS":
                channels.append(tok[2:])
            elif tok[0] == "Frame" and tok[1] == "Time:":
                return names, parents, offsets, channels, n_lines


def load_joints(path):
    names, parents, offsets, channels, n_header = parse_header(path)
    col, cols = 0, {}
    ch_iter = iter(channels)
    for name in names:
        if name is None:
            continue
        ch = next(ch_iter)
        cols[name] = (col, ch)
        col += len(ch)

    usecols, layout = [], {}
    for j in JOINTS:
        start, ch = cols[j]
        layout[j] = (len(usecols), ch)
        usecols.extend(range(start, start + len(ch)))
    # pandas returns usecols in file order; JOINTS is listed in file order so the layout holds.
    assert np.all(np.diff(usecols) > 0)
    data = pd.read_csv(
        path, skiprows=n_header, sep=" ", header=None, usecols=usecols, dtype=np.float64, engine="c"
    ).to_numpy()[::STRIDE]

    T = data.shape[0]
    glob_rot, glob_pos = {}, {}
    name_to_idx = {n: i for i, n in enumerate(names) if n is not None}
    for j in JOINTS:
        start, ch = layout[j]
        vals = data[:, start:start + len(ch)]
        pos_ch = [k for k, c in enumerate(ch) if c.endswith("position")]
        rot_ch = [k for k, c in enumerate(ch) if c.endswith("rotation")]
        local_pos = vals[:, pos_ch] if pos_ch else np.tile(offsets[name_to_idx[j]], (T, 1))
        axes = "".join(ch[k][0] for k in rot_ch)
        local_rot = R.from_euler(axes.upper(), vals[:, rot_ch], degrees=True)
        p = parents[name_to_idx[j]]
        if p < 0:
            glob_rot[j], glob_pos[j] = local_rot, local_pos
        else:
            pn = names[p]
            glob_rot[j] = glob_rot[pn] * local_rot
            glob_pos[j] = glob_pos[pn] + glob_rot[pn].apply(local_pos)
    y_to_z = R.from_matrix(Y_TO_Z)
    pos = {j: glob_pos[j] @ Y_TO_Z.T / 100.0 for j in JOINTS}
    rot = {j: y_to_z * glob_rot[j] for j in ("Hips", "Chest")}
    return pos, rot


def torso_target_pitch(rot):
    """Forward pitch (deg) the torso_link target asks of Yaw_Joint, measured in the heading frame of
    the pelvis_proxy target; matches the IK pitch closely when the joint is not at a limit."""
    heading = (rot["Hips"] * HIPS_OFFSET).apply([1.0, 0.0, 0.0])
    heading[:, 2] = 0
    heading /= np.linalg.norm(heading, axis=1, keepdims=True) + 1e-9
    z = (rot["Chest"] * CHEST_OFFSET).apply([0.0, 0.0, 1.0])
    return np.degrees(np.arctan2((z * heading).sum(1), z[:, 2]))


def frac(mask):
    return float(np.mean(mask)) if mask.size else 0.0


def survey_clip(path):
    try:
        P, rot = load_joints(path)
    except Exception as e:  # noqa: BLE001 - one bad file should not kill a 142k-file sweep
        return {"path": path, "error": repr(e)}

    hips = P["Hips"]
    hz = hips[:, 2]
    pitch = torso_target_pitch(rot)

    v_lift = np.abs(np.diff(hz)) * FPS
    k = 5  # base speed over ~0.17 s windows, so mocap jitter does not dominate
    v_base = np.linalg.norm(hips[k:, :2] - hips[:-k, :2], axis=1) * FPS / k if len(hips) > k else np.zeros(0)
    hands_z = np.stack([P["LeftHand"][:, 2], P["RightHand"][:, 2]], 1)

    lift_low = hz < LIFT_BASE - LIFT_TOL
    lift_high = hz > LIFT_BASE + LIFT_STROKE + LIFT_TOL
    pitch_back = pitch < PITCH_MIN_DEG
    pitch_fwd = pitch > PITCH_MAX_DEG
    fast = np.zeros_like(hz, dtype=bool)
    fast[k:] = v_base > BASE_SPEED
    out_ws = lift_low | lift_high | pitch_back | pitch_fwd | fast

    return {
        "path": path,
        "frames_30": int(len(hz)),
        "hips_z_min": float(hz.min()),
        "hips_z_p50": float(np.median(hz)),
        "hips_z_max": float(hz.max()),
        "frac_lift_low": frac(lift_low),
        "frac_lift_high": frac(lift_high),
        "frac_floor": frac(hz < FLOOR_HIPS_Z),
        "pitch_min": float(pitch.min()),
        "pitch_max": float(pitch.max()),
        "frac_pitch_back": frac(pitch_back),
        "frac_pitch_fwd": frac(pitch_fwd),
        "lift_speed_p95": float(np.percentile(v_lift, 95)) if v_lift.size else 0.0,
        "frac_lift_fast": frac(v_lift > LIFT_SPEED),
        "base_speed_max": float(v_base.max()) if v_base.size else 0.0,
        "frac_base_fast": frac(fast),
        "frac_out_ws": frac(out_ws),
        "hand_z_min": float(hands_z.min()),
        "hand_z_max": float(hands_z.max()),
        "base_path_m": float(np.linalg.norm(np.diff(hips[:, :2], axis=0), axis=1).sum()),
        "error": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--paths", nargs="*", default=None)
    args = ap.parse_args()

    meta = pd.read_parquet(METADATA)
    if args.paths:
        paths = args.paths
    else:
        paths = [f"{BONES}/{p}" for p in meta.move_soma_uniform_path]
        if args.limit:
            paths = paths[: args.limit]

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(survey_clip, paths, chunksize=8), 1):
            rows.append(row)
            if i % 2000 == 0 or i == len(paths):
                el = time.time() - t0
                print(f"{i}/{len(paths)}  {el/60:.1f} min  eta {el/i*(len(paths)-i)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    print("errors:", int((df.error != "").sum()), "written", args.out)


if __name__ == "__main__":
    main()
