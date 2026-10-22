"""Move the clips that the jump audit blames on the data, on our joint limits or on motion too fast for the
robot into a version's OOD zone.

- converted clips the current zone table puts in red (the version was converted before a stricter rule)
- an actor whose capture session has twist flips in at least FLIP_SHARE of its scanned clips
  (survey/ref_twist_flips.csv): all of that actor's green and yellow clips of the session
- clips with a jump segment of an excluded cause (<version>/audit/jump_segments.csv); for causes in the
  reference motion itself, the mirrored twin as well

Prints the plan; --apply moves the pkl (scaleretarget/utils/ood.py), the panel's OOD zone shows them and
can put any of them back. Run jump_audit.py segments / flips first.
"""

import argparse
import os

import pandas as pd

from scaleretarget.utils.jump_audit import CAUSES
from scaleretarget.utils.ood import dataset_pkl, move_to_ood, ood_pkl

FLIP_SHARE = 0.5
MIRRORED = ("ref_fails", "ref_bug")


def reasons(dataset):
    reference = "官方 G1" if dataset.official_csv_root else "标准 G1"
    return {
        "ref_fails": f"{reference} 也转不好",
        "ref_bug": "参考数据跳变",
        "joint_limits": "卡在收窄的肩/肘限位",
        "wrist_cap": "转腕被逐帧上限挤到肩上",
        "fast_motion": "动作太快，机器人跟不上",
    }


def plan(dataset, version_dir, zones):
    """rel -> reason, red clips first, then flipping actors."""
    red = zones.zone == "red"
    out = {rel: f"红区：{reason}" for rel, reason in zip(zones.rel[red], zones.reason[red])
           if os.path.isfile(dataset_pkl(version_dir, rel)) or os.path.isfile(ood_pkl(version_dir, rel))}
    zones = zones[~red]
    if os.path.isfile(dataset.twist_flips):
        flips = pd.read_csv(dataset.twist_flips).merge(dataset.load_zones()[["rel", "session", "actor"]], on="rel")
        share = flips.groupby(["session", "actor"]).flip_steps.apply(lambda s: (s > 0).mean())
        for (session, name), frac in share[share >= FLIP_SHARE].items():
            print(f"[flips] {session} {name}: {frac:.0%} of scanned clips flip")
            for rel in zones.rel[(zones.session == session) & (zones.actor == name)]:
                out.setdefault(rel, f"参考数据上臂扭转翻转（{name}）")
    segments = os.path.join(version_dir, "audit", "jump_segments.csv")
    if os.path.isfile(segments):
        why = reasons(dataset)
        seg = pd.read_csv(segments)
        seg = seg[seg.cause.isin(why)].copy()
        seg["rank"] = seg.cause.map(CAUSES.index)
        known = set(zones.rel)
        for rel, cause in seg.sort_values("rank").groupby("rel").cause.first().items():
            out.setdefault(rel, why[cause])
            if cause in MIRRORED and dataset.twin(rel) in known:
                out.setdefault(dataset.twin(rel), why[cause])
    return out


def main(dataset, argv=None):
    ap = argparse.ArgumentParser(description=f"move {dataset.name} clips the jump audit blames into the OOD zone")
    ap.add_argument("--version", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    version_dir = dataset.version_dir(args.version)
    zones = dataset.load_zones()
    zone_of = dict(zip(zones.rel, zones.zone))

    todo, done, missing = {}, 0, 0
    for rel, reason in plan(dataset, version_dir, zones).items():
        if os.path.isfile(ood_pkl(version_dir, rel)):
            done += 1
        elif os.path.isfile(dataset_pkl(version_dir, rel)):
            todo[rel] = reason
        else:
            missing += 1
    print(f"already in OOD {done}, not converted {missing}, to move {len(todo)}")
    for reason, n in pd.Series(todo, dtype=str).value_counts().items():
        print(f"  {n:4d}  {reason}")
    if not args.apply:
        print("dry run; --apply to move")
        return
    for rel, reason in todo.items():
        move_to_ood(version_dir, rel, zone_of[rel], reason)
    print(f"moved {len(todo)} clips to {os.path.join(version_dir, 'ood')}")
