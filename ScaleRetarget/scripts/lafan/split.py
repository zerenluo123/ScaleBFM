"""Cut each LAFAN1 take into parts of about PART_SEC seconds, the clips of the pipeline.

A take runs 1.7-5 minutes and mixes upright motion with falls and floor work, so zoning whole takes would
drop every take with a few seconds on the floor. Each take {take}_{subject}.bvh is cut into
round(length / PART_SEC) equal parts (at least one) that do not overlap, written to
OUT/bvh/{take}_{subject}_pNN.bvh: the same header with the part's frame count, and the part's motion lines
unchanged (positions stay in the take's world frame). Rerunning rewrites the same files.

    python scripts/lafan/split.py
"""

import glob
import os

import numpy as np

from dataset import BVH_ROOT, LAFAN, PART_SEC


def split(path):
    with open(path, "rb") as f:
        lines = f.read().splitlines(keepends=True)
    i = next(k for k, line in enumerate(lines) if line.startswith(b"Frames:"))
    frame_time = float(lines[i + 1].split()[2])
    motion = lines[i + 2:]
    n = int(lines[i].split()[1])
    assert len(motion) == n, f"{path}: {len(motion)} motion lines, header says {n}"
    parts = max(1, round(n * frame_time / PART_SEC))
    bounds = np.linspace(0, n, parts + 1).round().astype(int)
    eol = lines[i][len(lines[i].rstrip()):]
    take = os.path.basename(path)[:-4]
    for p, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        out = os.path.join(BVH_ROOT, f"{take}_p{p:02d}.bvh")
        with open(out + ".tmp", "wb") as f:
            f.writelines([*lines[:i], f"Frames: {b - a}".encode() + eol, lines[i + 1], *motion[a:b]])
        os.replace(out + ".tmp", out)
    return parts


if __name__ == "__main__":
    os.makedirs(BVH_ROOT, exist_ok=True)
    takes = sorted(glob.glob(f"{LAFAN}/*.bvh"))
    total = sum(split(path) for path in takes)
    print(f"{len(takes)} takes cut into {total} parts in {BVH_ROOT}")
