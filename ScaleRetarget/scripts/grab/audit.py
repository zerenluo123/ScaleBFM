"""Quality audit of the GRAB g1d retarget (scaleretarget/pipeline/audit.py).

full    every pkl: NaN, frame count vs the loader's resampling, joint range, lift / joint / base speed
sample  stratified sample (zone x reason x category): recompute the IK targets from the source and
        compare them with the robot links from FK; self-collision depth between robot bodies

    python scripts/grab/audit.py --run_dir RUN_DIR --out_dir DIR [--frac 0.02] [--workers 18]
"""

from dataset import DATASET
from scaleretarget.pipeline.audit import main

if __name__ == "__main__":
    main(DATASET)
