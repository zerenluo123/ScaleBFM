"""Causes of the arm jumps in an Embody3D retarget, and twist flips in the reference
(scaleretarget/pipeline/jump_audit.py). The comparison is the standard G1 retarget of each clip with a jump
(embody3d_to_unitree_g1.yaml), computed on the spot.

segments  -> <version>/audit/jump_segments.csv, from the clips with jumps in the panel's frame cache
flips     green / yellow clips, optionally of some subsets -> survey/ref_twist_flips.csv

    python scripts/embody3d/jump_audit.py segments --version v1
    python scripts/embody3d/jump_audit.py flips [--subsets locomotion charades]
"""

from dataset import DATASET
from scaleretarget.pipeline.jump_audit import main

if __name__ == "__main__":
    main(DATASET, groups_flag="--subsets")
