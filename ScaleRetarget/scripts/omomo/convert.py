"""Write every OMOMO sequence in the OmomoLoader format: OMOMO_UNPACKED/data/{train,test}_diffusion_manip_seq_joints24.p
-> OUT/bvh/{seq_name}.pkl, as scaleretarget/utils/convert_omomo_to_smplx.py writes them (the sequence's
fields, poses = body pose padded with zeros to 165, mocap_frame_rate 30). Files already written are skipped.

    python scripts/omomo/convert.py
"""

import os
import pickle

import joblib
import numpy as np

from dataset import BVH_ROOT, OMOMO_UNPACKED, SEQ_FILES

FPS = 30


def main():
    os.makedirs(BVH_ROOT, exist_ok=True)
    total = written = 0
    for name in SEQ_FILES:
        for seq in joblib.load(f"{OMOMO_UNPACKED}/{name}").values():
            total += 1
            dst = f"{BVH_ROOT}/{seq['seq_name']}.pkl"
            if os.path.exists(dst):
                continue
            seq["poses"] = np.concatenate([seq["pose_body"], np.zeros((len(seq["pose_body"]), 102))], axis=1)
            seq["mocap_frame_rate"] = np.array(FPS)
            with open(dst + ".tmp", "wb") as f:
                pickle.dump(seq, f)
            os.replace(dst + ".tmp", dst)
            written += 1
    print(f"{total} sequences in {OMOMO_UNPACKED}, {written} written to {BVH_ROOT}")


if __name__ == "__main__":
    main()
