"""Per-clip kinematic survey of BONES-SEED SOMA Uniform BVHs against the g1d workspace.

Parses only the joints needed (hips, spine chain, arms), downsamples to 30 fps, and writes one row
of metrics (scaleretarget/pipeline/workspace.py) per clip. Zones are assigned afterwards from these
metrics, so thresholds can be changed without re-reading the 277 GB of BVH text.

    python scripts/bones_seed/survey.py --out /path/to/survey.parquet [--limit N] [--workers 20]
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from dataset import BONES, CORRESPONDENCE, METADATA
from scaleretarget.pipeline.workspace import clip_metrics, rotation_offsets

SRC_FPS = 120
STRIDE = 4  # 120 fps -> 30 fps, same rate the retarget runs at
FPS = SRC_FPS / STRIDE

JOINTS = [
    "Root", "Hips", "Spine1", "Spine2", "Chest",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]

# SOMA Y-up centimeters -> Z-up meters, identical to SomaLoader.
Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

OFFSETS = rotation_offsets(CORRESPONDENCE)


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


def survey_clip(path):
    try:
        P, rot = load_joints(path)
    except Exception as e:  # noqa: BLE001 - one bad file should not kill a 142k-file sweep
        return {"path": path, "error": repr(e)}
    hands_z = np.stack([P["LeftHand"][:, 2], P["RightHand"][:, 2]], 1)
    return {"path": path, **clip_metrics(P["Hips"], rot["Hips"], rot["Chest"], hands_z, FPS, OFFSETS), "error": ""}


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
