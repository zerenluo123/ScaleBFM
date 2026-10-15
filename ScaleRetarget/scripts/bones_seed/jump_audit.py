"""Causes of the arm jumps in a BONES-SEED retarget, and twist flips in the SOMA reference.

segments  every arm jump (> 20 rad/s) of the converted clips of a version, in the dataset or in its OOD
          zone, with its cause (scaleretarget/utils/jump_audit.py) next to the official BONES G1 retarget
          -> <version>/audit/jump_segments.csv. The clip list comes from the panel's frame cache.
flips     steps per clip where the reference upper arm and forearm spin while the hand stays still, for
          the green and yellow clips (optionally of some capture dates only; rows of other dates are kept)
          -> survey/ref_twist_flips.csv

    python scripts/bones_seed/jump_audit.py segments --version v3
    python scripts/bones_seed/jump_audit.py flips [--dates 230112 230113]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import pandas as pd

from dataset import BVH_ROOT, OFFICIAL_CSV_ROOT, OUT, ROOT, TWIST_FLIPS, load_zones
from scaleretarget.utils.jump_audit import CAUSES, arm_bones, bone_speeds, jump_segments, twist_flips
from scaleretarget.utils.ood import OOD_DIR, find_pkl

_ctx = {}


def _init(robot, snapshot, reference):
    from hydra.utils import instantiate
    from loguru import logger
    from scaleretarget.viewer.clip_player import load_config
    logger.remove()
    config = load_config(robot, "soma", snapshot=snapshot)
    _ctx["loader"] = instantiate(config.loader)
    _ctx["retargeter"] = instantiate(config.retargeter)
    if reference:
        _ctx["reference"] = instantiate(load_config("unitree_g1", "soma").retargeter)


def _load(rel, fps):
    loader = _ctx["loader"]
    loader.target_fps = fps
    return loader._load_sample(os.path.join(BVH_ROOT, rel + ".bvh"))


def _segments(args):
    rel, pkl = args
    from scaleretarget.viewer.clip_player import load_official_csv
    motion = joblib.load(pkl)
    frames, extras = _load(rel, int(round(motion["fps"])))
    _ctx["retargeter"].update(extras)
    reference = None
    csv = os.path.join(OFFICIAL_CSV_ROOT, rel + ".csv")
    if os.path.isfile(csv):
        _ctx["reference"].update(extras)
        reference = (_ctx["reference"], load_official_csv(csv))
    return [{"rel": rel, **row} for row in jump_segments(_ctx["retargeter"], frames, motion, reference)]


def _flips(rel):
    frames, _ = _load(rel, 30)
    bones = arm_bones(_ctx["retargeter"]).values()
    return rel, int(sum(twist_flips(bone_speeds(frames, b, 30)).sum() for b in bones))


def segments(args):
    version_dir = os.path.join(OUT, args.version)
    caches = [os.path.join(version_dir, "frame_counts.csv"), os.path.join(version_dir, OOD_DIR, "frame_counts.csv")]
    found = [pd.read_csv(c, keep_default_na=False) for c in caches if os.path.isfile(c)]
    if not found:
        raise SystemExit(f"no frame cache in {version_dir}; open the panel on {args.version} or run "
                         f"python scripts/bones_seed/browse.py --build_cache {args.version}")
    jumps = pd.concat(found)
    rels = sorted(jumps.rel[jumps.gt20 != ""].unique())
    pkls = [find_pkl(version_dir, r) for r in rels]
    robot = json.load(open(os.path.join(version_dir, "run.json"))).get("robot", "unitree_g1d")
    print(f"{len(rels)} clips with arm jumps in {args.version}")
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(robot, version_dir, True)) as ex:
        rows = [r for rs in ex.map(_segments, zip(rels, pkls)) for r in rs]
    zones = load_zones()[["rel", "zone"]]
    df = pd.DataFrame(rows).merge(zones, on="rel", how="left")
    os.makedirs(os.path.join(version_dir, "audit"), exist_ok=True)
    out = os.path.join(version_dir, "audit", "jump_segments.csv")
    df.to_csv(out, index=False, float_format="%.4g")
    counts = df.groupby("cause").rel.agg(["size", "nunique"]).reindex(CAUSES).fillna(0).astype(int)
    print(counts.rename(columns={"size": "segments", "nunique": "clips"}).to_string())
    print("written", out)


def flips(args):
    zones = load_zones()
    zones = zones[zones.zone != "red"]
    if args.dates:
        zones = zones[zones.rel.str.split("/").str[0].isin(args.dates)]
    print(f"scanning {len(zones)} clips")
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=("unitree_g1d", None, False)) as ex:
        counts = dict(ex.map(_flips, zones.rel, chunksize=4))
    df = zones[["rel", "zone"]].assign(flip_steps=zones.rel.map(counts))
    if os.path.isfile(TWIST_FLIPS):
        old = pd.read_csv(TWIST_FLIPS)
        df = pd.concat([old[~old.rel.isin(df.rel)], df], ignore_index=True)
    df.sort_values("rel").to_csv(TWIST_FLIPS, index=False)
    print(f"{int((df.flip_steps > 0).sum())} of {len(df)} scanned clips have twist flips; written {TWIST_FLIPS}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    seg = sub.add_parser("segments")
    seg.add_argument("--version", required=True)
    flp = sub.add_parser("flips")
    flp.add_argument("--dates", nargs="*", default=None, help="capture dates, e.g. 230112; all by default")
    for p in (seg, flp):
        p.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    os.chdir(ROOT)  # robot configs name their MJCF relative to the ScaleRetarget folder
    segments(args) if args.mode == "segments" else flips(args)


if __name__ == "__main__":
    main()
