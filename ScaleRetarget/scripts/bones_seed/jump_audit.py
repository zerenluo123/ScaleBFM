"""Causes of the arm jumps in a BONES-SEED retarget, next to the official BONES G1 retarget, and twist
flips in the SOMA reference (scaleretarget/pipeline/jump_audit.py).

segments  -> <version>/audit/jump_segments.csv, from the clips with jumps in the panel's frame cache
flips     green / yellow clips, optionally of some capture dates -> survey/ref_twist_flips.csv

    python scripts/bones_seed/jump_audit.py segments --version v3
    python scripts/bones_seed/jump_audit.py flips [--dates 230112 230113]
"""

from dataset import DATASET
from scaleretarget.pipeline.jump_audit import main

if __name__ == "__main__":
    main(DATASET, groups_flag="--dates")
