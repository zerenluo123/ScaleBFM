"""Unpack HiPHI archives into UNPACKED and link every unpacked BVH under UNPACKED/bvh.

A part counts as unpacked when every motion it holds (data/motion_to_part.csv) has its BVH on disk;
those are skipped. A part is only extracted when the disk has room for it (archive_manifest.csv) plus
--reserve_gb. The archives are only read. Afterwards UNPACKED/bvh/{frame}/{motion_id}.bvh are rebuilt
as relative symlinks to ../../data/{frame}/{lu}/{motion_id}/motion_actor.bvh for every BVH present.

    python scripts/hiphi/unpack.py --parts 1 2 12      # or --all; --link_only only rebuilds the links
"""

import argparse
import os
import shutil
import subprocess
import time

import pandas as pd

from dataset import HIPHI, MANIFEST, MOTION_TO_PART, UNPACKED, read_metadata


def part_motions(meta):
    """archive name -> package BVH paths of its motions and their mirrors."""
    parts = pd.read_csv(MOTION_TO_PART)
    archive = dict(zip(parts.motion_id, parts.archive_name)) | dict(zip(parts.mirrored_motion_id, parts.archive_name))
    return meta.assign(archive=meta.motion_id.map(archive)).groupby("archive").package_bvh.apply(list).to_dict()


def unpack(archives, motions, reserve_gb):
    if shutil.which("zstd") is None:
        raise SystemExit("[unpack] tar needs the zstd program: conda install -c conda-forge zstd")
    manifest = pd.read_csv(MANIFEST).set_index("archive_name")
    os.makedirs(UNPACKED, exist_ok=True)
    for name in archives:
        bvhs = motions[name]
        if all(os.path.isfile(p) for p in bvhs):
            print(f"[unpack] {name}: done ({len(bvhs)} motions)")
            continue
        need, free = manifest.at[name, "uncompressed_bytes"], shutil.disk_usage(UNPACKED).free
        if free < need + reserve_gb * 1e9:
            raise SystemExit(f"[unpack] {name} needs {need / 1e9:.1f} GB (+{reserve_gb} GB reserve), "
                             f"{free / 1e9:.1f} GB free on {UNPACKED}")
        t0 = time.time()
        # Members are HiPHI/data/{frame}/{lu}/{motion_id}/...
        subprocess.run(["tar", "-I", "zstd", "-xf", f"{HIPHI}/data/{name}", "-C", UNPACKED, "--strip-components=1"],
                       check=True)
        print(f"[unpack] {name}: {len(bvhs)} motions, {need / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min")


def link(meta):
    """Rebuild UNPACKED/bvh from the package BVHs on disk; returns how many are linked."""
    root = os.path.join(UNPACKED, "bvh")
    present = meta[meta.package_bvh.map(os.path.isfile)]
    for frame, rows in present.groupby("frame"):
        os.makedirs(os.path.join(root, frame), exist_ok=True)
        for r in rows.itertuples():
            path = os.path.join(root, r.rel + ".bvh")
            target = os.path.join("..", "..", "data", r.frame, r.lu, r.motion_id, "motion_actor.bvh")
            if os.path.islink(path) and os.readlink(path) == target:
                continue
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(target, path)
    return len(present)


def main():
    ap = argparse.ArgumentParser()
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--parts", type=int, nargs="+", help="part numbers, e.g. 1 2 12")
    which.add_argument("--all", action="store_true")
    which.add_argument("--link_only", action="store_true")
    ap.add_argument("--reserve_gb", type=float, default=50, help="disk space to leave free")
    args = ap.parse_args()

    meta = read_metadata()
    if not args.link_only:
        motions = part_motions(meta)
        archives = sorted(motions) if args.all else [f"HiPHI_data_part_{p:04d}_of_0032.tar.zst" for p in args.parts]
        unpack(archives, motions, args.reserve_gb)
    n = link(meta)
    print(f"[link] {n} of {len(meta)} motions under {UNPACKED}/bvh")


if __name__ == "__main__":
    main()
