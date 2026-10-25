"""Hard-link the HiPHI clips the raise does not change from an earlier version into a raised one
(scaleretarget/pipeline/link_unraised.py).

    python scripts/hiphi/link_unraised.py --src v2 --dst v3 [--apply]
"""

from dataset import DATASET
from scaleretarget.pipeline.link_unraised import main

if __name__ == "__main__":
    main(DATASET)
