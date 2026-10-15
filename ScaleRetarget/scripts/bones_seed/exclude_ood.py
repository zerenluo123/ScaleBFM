"""Move the clips that the jump audit blames on the data or on our joint limits into a version's OOD zone.

- an actor whose capture session has twist flips in at least FLIP_SHARE of its scanned clips
  (survey/ref_twist_flips.csv): all of that actor's green and yellow clips of the session
- clips with a jump segment of an excluded cause (<version>/audit/jump_segments.csv); for causes in the
  reference motion itself, the mirrored twin (_M) as well

Prints the plan; --apply moves the pkl (scaleretarget/utils/ood.py), the panel's OOD zone shows them and
can put any of them back. Run jump_audit.py segments / flips first.

    python scripts/bones_seed/exclude_ood.py --version v3 [--apply]
"""

import argparse
import os

import pandas as pd

from dataset import OUT, TWIST_FLIPS, actor, load_zones
from scaleretarget.utils.jump_audit import CAUSES
from scaleretarget.utils.ood import dataset_pkl, move_to_ood, ood_pkl

FLIP_SHARE = 0.5
REASONS = {
    "ref_fails": "官方 G1 也转不好",
    "ref_bug": "参考数据跳变",
    "joint_limits": "卡在收窄的肩/肘限位",
    "wrist_cap": "转腕被逐帧上限挤到肩上",
}
MIRRORED = ("ref_fails", "ref_bug")


def twin(rel):
    return rel.removesuffix("_M") if rel.endswith("_M") else rel + "_M"


def plan(version_dir, zones):
    """rel -> reason, flipping actors first."""
    out = {}
    if os.path.isfile(TWIST_FLIPS):
        flips = pd.read_csv(TWIST_FLIPS)
        flips[["session", "actor"]] = flips.rel.map(actor).tolist()
        share = flips.groupby(["session", "actor"]).flip_steps.apply(lambda s: (s > 0).mean())
        groups = zones.assign(key=zones.rel.map(actor))
        for (session, name), frac in share[share >= FLIP_SHARE].items():
            print(f"[flips] {session} {name}: {frac:.0%} of scanned clips flip")
            for rel in groups.rel[groups.key == (session, name)]:
                out.setdefault(rel, f"参考数据上臂扭转翻转（{name}）")
    segments = os.path.join(version_dir, "audit", "jump_segments.csv")
    if os.path.isfile(segments):
        seg = pd.read_csv(segments)
        seg = seg[seg.cause.isin(REASONS)].copy()
        seg["rank"] = seg.cause.map(CAUSES.index)
        known = set(zones.rel)
        for rel, cause in seg.sort_values("rank").groupby("rel").cause.first().items():
            out.setdefault(rel, REASONS[cause])
            if cause in MIRRORED and twin(rel) in known:
                out.setdefault(twin(rel), REASONS[cause])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    version_dir = os.path.join(OUT, args.version)
    zones = load_zones()
    zones = zones[zones.zone != "red"]
    zone_of = dict(zip(zones.rel, zones.zone))

    todo, done, missing = {}, 0, 0
    for rel, reason in plan(version_dir, zones).items():
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


if __name__ == "__main__":
    main()
