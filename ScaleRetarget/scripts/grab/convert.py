"""Write every unzipped GRAB sequence in the GrabLoader format: GRAB_UNPACKED/s1/{name}.npz ->
OUT/bvh/s1/{name}.pkl (scaleretarget/utils/convert_grab_to_smplx.py: body and hand poses, frame rate,
gender, zero betas). Files already written are skipped.

    python scripts/grab/convert.py [--workers 32]
"""

import argparse
import glob
import os
from multiprocessing import Pool

import joblib

from dataset import BVH_ROOT, GRAB_UNPACKED
from scaleretarget.utils.convert_grab_to_smplx import construct_smplx_data


def convert(src):
    rel = os.path.relpath(src, GRAB_UNPACKED)[:-4]
    dst = f"{BVH_ROOT}/{rel}.pkl"
    if os.path.exists(dst):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    joblib.dump(construct_smplx_data(src), dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    srcs = sorted(glob.glob(f"{GRAB_UNPACKED}/s*/*.npz"))
    with Pool(args.workers) as pool:
        written = sum(pool.imap_unordered(convert, srcs, chunksize=8))
    print(f"{len(srcs)} sequences in {GRAB_UNPACKED}, {written} written to {BVH_ROOT}")


if __name__ == "__main__":
    main()
