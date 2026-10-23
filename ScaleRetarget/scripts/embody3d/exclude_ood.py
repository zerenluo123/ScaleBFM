"""Move the Embody3D clips that the jump audit blames on the data or on our joint limits into a version's
OOD zone (scaleretarget/pipeline/exclude.py): a session's clips when most of them have twist flips, and
clips with a jump of an excluded cause. Prints the plan; --apply moves the pkl. Run jump_audit.py
segments / flips first.

    python scripts/embody3d/exclude_ood.py --version v1 [--apply]
"""

from dataset import DATASET
from scaleretarget.pipeline.exclude import main

if __name__ == "__main__":
    main(DATASET)
