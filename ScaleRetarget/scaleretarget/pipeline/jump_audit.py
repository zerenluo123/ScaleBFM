"""Causes of the arm jumps in a dataset's retarget version, and twist flips in its reference motion.

segments  every arm jump (> 20 rad/s) of the converted clips of a version, in the dataset or in its OOD
          zone, with its cause (scaleretarget/utils/jump_audit.py) next to a standard G1 retarget: the
          dataset's official one, or ours retargeted now -> <version>/audit/jump_segments.csv.
          The clip list comes from the panel's frame cache.
flips     steps per clip where the reference upper arm and forearm spin while the hand stays still, for
          the green and yellow clips (optionally of some groups only, e.g. capture dates; rows of other
          groups are kept) -> survey/ref_twist_flips.csv
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import pandas as pd

from scaleretarget.pipeline.dataset import ROOT
from scaleretarget.pipeline.frames import run_info
from scaleretarget.utils.jump_audit import CAUSES, arm_bones, bone_speeds, jump_segments, twist_flips
from scaleretarget.utils.ood import OOD_DIR, find_pkl

REFERENCE_ROBOT = "unitree_g1"

_ctx = {}


def _init(dataset, robot, snapshot, reference):
    from hydra.utils import instantiate
    from loguru import logger
    from scaleretarget.viewer.clip_player import load_config
    logger.remove()
    config = load_config(robot, dataset.loader, snapshot=snapshot)
    _ctx["dataset"] = dataset
    _ctx["loader"] = instantiate(config.loader)
    _ctx["retargeter"] = instantiate(config.retargeter)
    if reference:
        _ctx["reference"] = instantiate(load_config(REFERENCE_ROBOT, dataset.loader).retargeter)
        _ctx["reference"].show_progress = False


def _load(rel, fps):
    loader = _ctx["loader"]
    loader.target_fps = fps
    return loader._load_sample(_ctx["dataset"].bvh_path(rel))


def _reference_motion(rel, frames, fps):
    """The standard G1 retarget of the clip as a formatter dict (xyzw root), or None without one."""
    from scaleretarget.viewer.clip_player import load_official_csv
    root = _ctx["dataset"].official_csv_root
    if root is not None:
        csv = os.path.join(root, rel + ".csv")
        return load_official_csv(csv) if os.path.isfile(csv) else None
    reference = _ctx["reference"]
    root_pos, root_rot, dof_pos = reference.base.split_qpos(reference.retarget([dict(f) for f in frames]))
    return {"root_pos": root_pos, "root_rot": root_rot[:, [1, 2, 3, 0]], "dof_pos": dof_pos, "fps": fps}


def _segments(args):
    rel, pkl = args
    motion = joblib.load(pkl)
    fps = int(round(motion["fps"]))
    frames, extras = _load(rel, fps)
    _ctx["retargeter"].update(extras)
    _ctx["reference"].update(extras)
    ref_motion = _reference_motion(rel, frames, fps)
    reference = (_ctx["reference"], ref_motion) if ref_motion is not None else None
    return [{"rel": rel, **row} for row in jump_segments(_ctx["retargeter"], frames, motion, reference)]


def _flips(rel):
    frames, _ = _load(rel, 30)
    bones = arm_bones(_ctx["retargeter"]).values()
    return rel, int(sum(twist_flips(bone_speeds(frames, b, 30)).sum() for b in bones))


def segments(dataset, args):
    version_dir = dataset.version_dir(args.version)
    caches = [os.path.join(version_dir, "frame_counts.csv"), os.path.join(version_dir, OOD_DIR, "frame_counts.csv")]
    found = [pd.read_csv(c, keep_default_na=False) for c in caches if os.path.isfile(c)]
    if not found:
        raise SystemExit(f"no frame cache in {version_dir}; open the panel on {args.version} or run "
                         f"python scripts/{dataset.key}/browse.py --build_cache {args.version}")
    jumps = pd.concat(found)
    rels = sorted(jumps.rel[jumps.gt20 != ""].unique())
    pkls = [find_pkl(version_dir, r) for r in rels]
    robot = run_info(version_dir).get("robot", dataset.robot)
    print(f"{len(rels)} clips with arm jumps in {args.version}")
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(dataset, robot, version_dir, True)) as ex:
        rows = [r for rs in ex.map(_segments, zip(rels, pkls)) for r in rs]
    zones = dataset.load_zones()[["rel", "zone"]]
    df = (pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rel", "cause"])).merge(zones, on="rel", how="left")
    os.makedirs(os.path.join(version_dir, "audit"), exist_ok=True)
    out = os.path.join(version_dir, "audit", "jump_segments.csv")
    df.to_csv(out, index=False, float_format="%.4g")
    if len(df):
        counts = df.groupby("cause").rel.agg(["size", "nunique"]).reindex(CAUSES).fillna(0).astype(int)
        print(counts.rename(columns={"size": "segments", "nunique": "clips"}).to_string())
    print("written", out)


def flips(dataset, args):
    zones = dataset.load_zones()
    zones = zones[zones.zone != "red"]
    if args.groups:
        zones = zones[zones.group.isin(args.groups)]
    print(f"scanning {len(zones)} clips")
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(dataset, dataset.robot, None, False)) as ex:
        counts = dict(ex.map(_flips, zones.rel, chunksize=4))
    df = zones[["rel", "zone"]].assign(flip_steps=zones.rel.map(counts))
    path = dataset.twist_flips
    if os.path.isfile(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.rel.isin(df.rel)], df], ignore_index=True)
    df.sort_values("rel").to_csv(path, index=False)
    print(f"{int((df.flip_steps > 0).sum())} of {len(df)} scanned clips have twist flips; written {path}")


def main(dataset, argv=None, groups_flag="--groups"):
    ap = argparse.ArgumentParser(description=f"arm jump causes and reference twist flips of {dataset.name}")
    sub = ap.add_subparsers(dest="mode", required=True)
    seg = sub.add_parser("segments")
    seg.add_argument("--version", required=True)
    flp = sub.add_parser("flips")
    flp.add_argument(groups_flag, dest="groups", nargs="*", default=None,
                     help=f"only clips of these {dataset.group_label} values; all by default")
    for p in (seg, flp):
        p.add_argument("--workers", type=int, default=18)
    args = ap.parse_args(argv)
    os.chdir(ROOT)  # robot configs name their MJCF relative to the ScaleRetarget folder
    segments(dataset, args) if args.mode == "segments" else flips(dataset, args)
