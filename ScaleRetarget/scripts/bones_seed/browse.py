"""Browse BONES-SEED clips by g1d zone, follow the batch retarget progress, and play clips in MuJoCo
(scaleretarget/pipeline/panel.py). "对比官方 G1" opens our retarget next to the official BONES G1 retarget.

    python scripts/bones_seed/browse.py [--version v2] [--cache_only]
    python scripts/bones_seed/browse.py --build_cache v1 [--workers 18] [--loop 60]    # fill / keep a version's frame cache
"""

from dataset import DATASET
from scaleretarget.pipeline.panel import main

if __name__ == "__main__":
    main(DATASET)
