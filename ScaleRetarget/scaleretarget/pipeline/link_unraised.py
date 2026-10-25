"""Start a raised version (retarget with ++robot.root_height_filter.min_height=MIN_HIPS) from an earlier
one: clips kept in the current zones whose hips never go below MIN_HIPS retarget the same with the raise,
so their pkl are hard-linked from the earlier version instead of converted again. An OOD pkl of the
earlier version goes into the new OOD folder with its log row, unless it was moved there only for its old
zone (exclude.py's 红区：reason): the clip is kept now, so it goes into bvh/. The earlier version is not
touched. Run before the new version's first batch run, which then converts only the rest; the linked
clips are listed in <new>/linked.csv.

    python scripts/<dataset>/link_unraised.py --src v1 --dst v2 [--apply]
"""

import argparse
import os

import pandas as pd

from scaleretarget.pipeline.workspace import LIFT_BASE
from scaleretarget.utils.ood import OOD_DIR, _write_ood_log, dataset_pkl, ood_pkl, read_ood_log

MIN_HIPS = LIFT_BASE
ZONE_MOVE = "红区："


def main(dataset):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--apply", action="store_true", help="link; without it only count")
    args = ap.parse_args()
    src, dst = dataset.version_dir(args.src), dataset.version_dir(args.dst)
    if os.path.exists(os.path.join(dst, "run.json")):
        raise SystemExit(f"{dst} has already been run; link only into a version that has not started")

    zones = dataset.load_zones()
    keep = zones[zones.zone.isin(["green", "yellow"]) & (zones.hips_z_min >= MIN_HIPS)]
    log = read_ood_log(src)
    zone_moved = set(log.rel[log.reason.str.startswith(ZONE_MOVE)])
    rows = []
    for rel in keep.rel:
        for where, path in (("bvh", dataset_pkl(src, rel)), ("ood", ood_pkl(src, rel))):
            if os.path.isfile(path):
                rows.append({"rel": rel, "folder": "bvh" if rel in zone_moved else where, "src": path})
                break
    found = pd.DataFrame(rows, columns=["rel", "folder", "src"])
    hours = keep.set_index("rel").loc[found.rel].seconds.sum() / 3600
    in_ood = int((found.folder == "ood").sum())
    back = int(found.rel.isin(zone_moved).sum())
    print(f"{len(keep)} kept clips never below {MIN_HIPS} m; {len(found)} converted in {args.src} ({hours:.2f} h): "
          f"{len(found) - in_ood} into bvh/ ({back} of them from {OOD_DIR}/, moved there for their old zone), "
          f"{in_ood} into {OOD_DIR}/")
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
        _write_ood_log(dst, pd.concat([read_ood_log(dst), log[log.rel.isin(ood.rel)]]).drop_duplicates("rel", keep="last"))
    found.assign(src_version=args.src).drop(columns="src").to_csv(os.path.join(dst, "linked.csv"), index=False)
    print(f"linked into {dst}")
