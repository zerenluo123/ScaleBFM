"""Browse LAFAN1 clips by g1d zone, follow the batch retarget progress, and play clips in MuJoCo
(scaleretarget/pipeline/panel.py). "对比标准 G1" retargets the clip to the standard G1 on the spot with the
original ScaleRetarget table (lafan_to_unitree_g1.yaml), next to ours.

    python scripts/lafan/browse.py [--version v1] [--cache_only]
    python scripts/lafan/browse.py --build_cache v1 [--workers 18] [--loop 60]    # fill / keep a version's frame cache
"""

from dataset import DATASET
from scaleretarget.pipeline.panel import main

if __name__ == "__main__":
    main(DATASET)
