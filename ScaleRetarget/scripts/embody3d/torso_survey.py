"""Per-clip torso survey of the Embody3D clips for the raised zones (workspace.torso_metrics): frames
lying on the floor, the longest held lean-back and side lean, and the workspace metrics of the motion
raised to the lowest lift. classify.py --torso reads it.

    python scripts/embody3d/torso_survey.py --out torso_raw.parquet [--workers 64]
"""

import argparse
import glob
import os

from dataset import BVH_ROOT, DATASET
from scaleretarget.pipeline.smplx_survey import run, torso_clip

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    run(DATASET, sorted(glob.glob(f"{BVH_ROOT}/*/*/*.pkl")), os.path.abspath(args.out), args.workers, clip=torso_clip)
