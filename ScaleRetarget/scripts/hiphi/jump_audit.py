"""Causes of the arm jumps in a HiPHI retarget, and twist flips in the reference
(scaleretarget/pipeline/jump_audit.py). The comparison is the standard G1 retarget of each clip with a jump,
computed on the spot.

segments  -> <version>/audit/jump_segments.csv, from the clips with jumps in the panel's frame cache
flips     green / yellow clips, optionally of some actors -> survey/ref_twist_flips.csv

    python scripts/hiphi/jump_audit.py segments --version v1
    python scripts/hiphi/jump_audit.py flips [--actors A001 A003]
"""

from dataset import DATASET
from scaleretarget.pipeline.jump_audit import main

if __name__ == "__main__":
    main(DATASET, groups_flag="--actors")
