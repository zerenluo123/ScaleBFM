"""Per-clip kinematic survey of the unpacked AMASS clips against the g1d workspace
(scaleretarget/pipeline/smplx_survey.py). Capture rates differ by subset (60-250 fps).

    python scripts/amass/survey.py --out survey_raw.parquet [--workers 64]
"""

import argparse
import glob
import os

from dataset import BVH_ROOT, DATASET
from scaleretarget.pipeline.smplx_survey import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    run(DATASET, sorted(glob.glob(f"{BVH_ROOT}/**/*_stageii.npz", recursive=True)), os.path.abspath(args.out), args.workers)
