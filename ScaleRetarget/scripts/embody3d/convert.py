"""Write every participant of every Embody3D capture in the Embody3dLoader format, read straight from the
SMPL-X archives: {subset}/{capture}/{person}/smplx_mesh_*/{capture}.npy -> OUT/bvh/{subset}/{capture}/{person}.pkl.

The pkl holds what scaleretarget/utils/gather_embody3d.py gathers for the body (pose_body, root_orient,
trans), and for the survey and the grounding the betas (one row; the archive repeats it on every frame),
the gender of the SMPL-X model they fit (neutral), the frame rate (30) and the tracked frames (the archive's
missing/ array, 1 where the tracking is good). Hand poses are left out: the loader plays the hands flat.
Files already written are skipped.

    python scripts/embody3d/convert.py [--workers 16]
"""

import argparse
import glob
import io
import os
import zipfile
from collections import defaultdict
from multiprocessing import Pool

import joblib
import numpy as np

from dataset import BVH_ROOT, EMBODY3D

FPS = 30.0
PARTS = {"pose_body": "smplx_mesh_body_pose", "root_orient": "smplx_mesh_global_orient", "trans": "smplx_mesh_transl",
         "betas": "smplx_mesh_betas", "tracked": "missing"}
_zips = {}


def clips(archive):
    """(archive, rel, {part: member}) of every participant folder in the archive, and the folders that lack a part."""
    found = defaultdict(dict)
    for name in zipfile.ZipFile(archive).namelist():
        parts = name.split("/")
        if len(parts) == 5 and parts[4].endswith(".npy") and parts[3] in PARTS.values():
            found["/".join(parts[:3])].setdefault(parts[3], name)
    full = [(archive, rel, m) for rel, m in sorted(found.items()) if len(m) == len(PARTS)]
    return full, sorted(set(found) - {rel for _, rel, _ in full})


def convert(job):
    archive, rel, members = job
    dst = f"{BVH_ROOT}/{rel}.pkl"
    if os.path.exists(dst):
        return False
    zf = _zips.setdefault(archive, zipfile.ZipFile(archive))
    arr = {part: np.load(io.BytesIO(zf.read(members[d]))) for part, d in PARTS.items()}
    n = len(arr["pose_body"])
    assert all(len(a) == n for a in arr.values()), f"{rel}: parts differ in frame count"
    data = {"pose_body": arr["pose_body"], "root_orient": arr["root_orient"], "trans": arr["trans"],
            "betas": arr["betas"][0], "gender": "neutral", "mocap_frame_rate": FPS,
            "tracked": arr["tracked"].reshape(n) > 0.5}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    joblib.dump(data, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    jobs = []
    for archive in sorted(glob.glob(f"{EMBODY3D}/embody_3d__*_smplx.zip")):
        full, partial = clips(archive)
        jobs += full
        print(f"{os.path.basename(archive)}: {len(full)} participants" + (f", {len(partial)} lacking a part, e.g. {partial[0]}" if partial else ""))
    with Pool(args.workers) as pool:
        written = sum(pool.imap_unordered(convert, jobs, chunksize=8))
    print(f"{len(jobs)} participants, {written} written to {BVH_ROOT}")


if __name__ == "__main__":
    main()
