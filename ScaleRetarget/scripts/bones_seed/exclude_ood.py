"""Move the BONES-SEED clips that the jump audit blames on the data or on our joint limits into a
version's OOD zone (scaleretarget/pipeline/exclude.py): actors of a capture date whose clips mostly have
twist flips, and clips with a jump of an excluded cause (plus the mirrored twin _M for causes in the
reference motion). Prints the plan; --apply moves the pkl. Run jump_audit.py segments / flips first.

    python scripts/bones_seed/exclude_ood.py --version v3 [--apply]
"""

from dataset import DATASET
from scaleretarget.pipeline.exclude import main

if __name__ == "__main__":
    main(DATASET)
