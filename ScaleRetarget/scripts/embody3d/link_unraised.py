"""Start a raised version (retarget with ++robot.root_height_filter.min_height=MIN_HIPS) from an earlier
one: clips kept in the current zones whose hips never go below MIN_HIPS retarget the same with the raise,
so their pkl are hard-linked from the earlier version (its OOD ones into the new OOD folder, with their
log rows) instead of converted again. The earlier version is not touched. Run before the new version's
first batch run, which then converts only the rest; the linked clips are listed in <new>/linked.csv.

    python scripts/embody3d/link_unraised.py --src v1 --dst v2 [--apply]
"""

import argparse
import os

import pandas as pd

from dataset import DATASET
from scaleretarget.pipeline.workspace import LIFT_BASE
from scaleretarget.utils.ood import OOD_DIR, _write_ood_log, dataset_pkl, ood_pkl, read_ood_log

MIN_HIPS = LIFT_BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--apply", action="store_true", help="link; without it only count")
    args = ap.parse_args()
    src, dst = DATASET.version_dir(args.src), DATASET.version_dir(args.dst)
    if os.path.exists(os.path.join(dst, "run.json")):
        raise SystemExit(f"{dst} has already been run; link only into a version that has not started")

    zones = DATASET.load_zones()
    keep = zones[zones.zone.isin(["green", "yellow"]) & (zones.hips_z_min >= MIN_HIPS)]
    rows = []
    for rel in keep.rel:
        for where, path in (("bvh", dataset_pkl(src, rel)), ("ood", ood_pkl(src, rel))):
            if os.path.isfile(path):
                rows.append({"rel": rel, "folder": where, "src": path})
                break
    found = pd.DataFrame(rows, columns=["rel", "folder", "src"])
    hours = keep.set_index("rel").loc[found.rel].duration_sec.sum() / 3600
    in_ood = int((found.folder == "ood").sum())
    print(f"{len(keep)} kept clips never below {MIN_HIPS} m; {len(found)} converted in {args.src} "
          f"({hours:.2f} h): {len(found) - in_ood} in bvh/, {in_ood} in {OOD_DIR}/")
    if not args.apply:
        print("dry run; --apply to link")
        return

    for r in found.itertuples():
        target = dataset_pkl(dst, r.rel) if r.folder == "bvh" else ood_pkl(dst, r.rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.exists(target):
            os.link(r.src, target)
    ood = found[found.folder == "ood"]
    if len(ood):
        log = read_ood_log(src)
        _write_ood_log(dst, pd.concat([read_ood_log(dst), log[log.rel.isin(ood.rel)]]).drop_duplicates("rel", keep="last"))
    found.assign(src_version=args.src).drop(columns="src").to_csv(os.path.join(dst, "linked.csv"), index=False)
    print(f"linked into {dst}")


if __name__ == "__main__":
    main()
