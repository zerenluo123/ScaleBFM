"""Quality audit of the BONES-SEED g1d retarget.

full    every pkl: NaN, frame count vs the loader's resampling, joint range, lift / joint / base speed
sample  stratified sample (zone x reason x category): recompute the IK targets from the BVH and
        compare them with the robot links from FK; self-collision depth between robot bodies

Clips moved to the version's OOD zone are not part of the dataset and are left out.

    python scripts/bones_seed/audit.py --run_dir RUN_DIR --out_dir DIR [--frac 0.02] [--workers 18]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import mujoco as mj
import numpy as np
import pandas as pd

from dataset import BVH_FRAME_TIME, BVH_ROOT, ROOT, load_zones

XML = "assets/unitree_g1d/g1d_mocap.xml"
LIFT = ["LZ_mt_Joint", "LZ_it_Joint"]
# Links tracked by the IK, grouped for reporting (robot body -> group).
GROUPS = {
    "pelvis_proxy": "pelvis", "torso_link": "chest",
    "left_shoulder_roll_link": "shoulder", "right_shoulder_roll_link": "shoulder",
    "left_elbow_link": "elbow", "right_elbow_link": "elbow",
    "left_wrist_yaw_link": "wrist", "right_wrist_yaw_link": "wrist",
}
PENETRATION = 0.01

_model = _data = _base = _loader = _retargeter = _rest_pairs = None


def _init(sample, run_dir):
    global _model, _data, _base, _loader, _retargeter, _rest_pairs
    from scaleretarget.utils.robot_base import RobotBase
    _model = mj.MjModel.from_xml_path(XML)
    _data = mj.MjData(_model)
    _base = RobotBase(_model)
    if sample:
        from loguru import logger
        from hydra.utils import instantiate
        from scaleretarget.viewer.clip_player import load_config
        logger.remove()
        config = load_config("unitree_g1d", "soma", snapshot=run_dir)
        _loader = instantiate(config.loader)
        _retargeter = instantiate(config.retargeter)
        _retargeter.show_progress = False
        mj.mj_forward(_model, _data)
        _rest_pairs = {_pair(c) for c in _data.contact[: _data.ncon]}


def _pair(contact):
    a, b = sorted((_model.geom_bodyid[contact.geom1], _model.geom_bodyid[contact.geom2]))
    return (int(a), int(b))


def _load(pkl):
    m = joblib.load(pkl)
    return m, m["root_pos"], m["root_rot"][:, [3, 0, 1, 2]], m["dof_pos"]


def full_check(args):
    rel, pkl = args
    m, root_pos, root_rot, dof = _load(pkl)
    fps = float(m["fps"])
    out = {"rel": rel, "frames": len(dof), "fps": fps,
           "nan": not all(np.isfinite(x).all() for x in (root_pos, root_rot, dof))}
    names = [_model.joint(i).name for i in range(_model.njnt)][_base.num_root_qpos:]
    lim = _model.jnt_limited[_base.num_root_qpos:].astype(bool)
    lo, hi = _model.jnt_range[_base.num_root_qpos:].T
    over = np.maximum(dof - hi, lo - dof)[:, lim]
    out["range_excess_max"] = float(over.max()) if over.size else 0.0
    lift = dof[:, [names.index(n) for n in LIFT]].sum(1)
    out["lift_speed_max"] = float(np.abs(np.diff(lift[1:])).max() * fps) if len(lift) > 2 else 0.0
    hinge = [i for i, n in enumerate(names) if n not in LIFT]
    vel = np.abs(np.diff(dof[1:, hinge], axis=0)) * fps if len(dof) > 2 else np.zeros((1, len(hinge)))
    out["joint_speed_max"] = float(vel.max())
    out["joint_speed_max_name"] = names[hinge[int(vel.max(0).argmax())]]
    out["joint_speed_p99"] = float(np.percentile(vel, 99))
    base_v = np.linalg.norm(np.diff(root_pos[1:, :2], axis=0), axis=1) * fps if len(dof) > 2 else np.zeros(1)
    yaw = 2 * np.arctan2(root_rot[:, 3], root_rot[:, 0])
    yaw_rate = np.abs(np.angle(np.exp(1j * np.diff(yaw[1:])))) * fps if len(dof) > 2 else np.zeros(1)
    out["base_speed_max"] = float(base_v.max())
    out["yaw_rate_max"] = float(yaw_rate.max())
    return out


def sample_check(args):
    rel, pkl = args
    m, root_pos, root_rot, dof = _load(pkl)
    _loader.target_fps = int(round(m["fps"]))
    frames, extras = _loader._load_sample(os.path.join(BVH_ROOT, rel + ".bvh"))
    _retargeter.update(extras)
    targets = _retargeter.compute_targets(frames)
    n = min(len(targets), len(dof))
    bodies = {b: (_model.body(b).id, entry[0]) for b, entry in _retargeter.ik_match_table.items() if b in GROUPS}
    pos_err = {g: [] for g in set(GROUPS.values())}
    rot_err = []
    pen_frames, pen_max, pairs = 0, 0.0, {}
    for i in range(n):
        _base.merge_qpos(_data.qpos, root_pos[i], root_rot[i], dof[i])
        mj.mj_forward(_model, _data)
        for body, (bid, bone) in bodies.items():
            tp, tq = targets[i][bone]
            pos_err[GROUPS[body]].append(np.linalg.norm(_data.xpos[bid] - tp))
            if GROUPS[body] == "wrist":
                dot = abs(float(np.dot(_data.xquat[bid], tq)))
                rot_err.append(2 * np.degrees(np.arccos(min(1.0, dot))))
        deepest = 0.0
        for c in _data.contact[: _data.ncon]:
            p = _pair(c)
            if p in _rest_pairs or -c.dist < PENETRATION:
                continue
            deepest = max(deepest, -c.dist)
            pairs[p] = pairs.get(p, 0) + 1
        pen_frames += deepest > 0
        pen_max = max(pen_max, deepest)
    out = {"rel": rel, "n": n}
    for g, e in pos_err.items():
        e = np.asarray(e) * 100
        out[f"{g}_p50_cm"], out[f"{g}_p95_cm"], out[f"{g}_max_cm"] = np.percentile(e, 50), np.percentile(e, 95), e.max()
    out["wrist_rot_p50_deg"], out["wrist_rot_p95_deg"] = np.percentile(rot_err, 50), np.percentile(rot_err, 95)
    out["collide_frac"] = pen_frames / n
    out["collide_max_cm"] = pen_max * 100
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:2]
    out["collide_pairs"] = "; ".join(f"{_model.body(a).name}|{_model.body(b).name}:{k}" for (a, b), k in top)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--skip_full", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(ROOT)  # robot configs name their MJCF relative to the ScaleRetarget folder

    z = load_zones()
    z = z[z.zone != "red"].copy()
    z["pkl"] = args.run_dir + "/bvh/" + z.rel + ".pkl"
    z = z[z.pkl.map(os.path.isfile)]
    z["frames_expected"] = np.maximum(2, np.round((z.frames_120 - 1) * BVH_FRAME_TIME * 30).astype(int) + 1)

    if not args.skip_full:
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(False, args.run_dir)) as ex:
            full = pd.DataFrame(ex.map(full_check, zip(z.rel, z.pkl), chunksize=200))
        full = z[["rel", "zone", "reason", "category", "frames_expected", "content_short_description"]].merge(full, on="rel")
        full.to_parquet(f"{args.out_dir}/full.parquet")
        print("full check written", len(full))

    rng = np.random.default_rng(0)
    picks = []
    for _, g in z.groupby(["zone", "reason", "category"]):
        k = max(1, int(round(len(g) * args.frac)))
        picks.append(g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)])
    s = pd.concat(picks)
    print("sample size", len(s))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(True, args.run_dir)) as ex:
        res = pd.DataFrame(ex.map(sample_check, zip(s.rel, s.pkl), chunksize=4))
    res = s[["rel", "zone", "reason", "category", "content_short_description", "seconds"]].merge(res, on="rel")
    res.to_parquet(f"{args.out_dir}/sample.parquet")
    print("sample check written", len(res))


if __name__ == "__main__":
    main()
