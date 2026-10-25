"""Hard-link the BONES-SEED clips the raise does not change from an earlier version into a raised one
(scaleretarget/pipeline/link_unraised.py).

    python scripts/bones_seed/link_unraised.py --src v1 --dst v2 [--apply]
"""

from dataset import DATASET
from scaleretarget.pipeline.link_unraised import main

if __name__ == "__main__":
    main(DATASET)
