"""Why the arm joints of a retargeted clip jump, sorted into causes.

Every jump segment (an arm joint step above JUMP_SPEED, as in arm_jumps.py) gets the global angular
speed of the human upper arm, forearm and hand, how much faster the robot's elbow / wrist link moves
than its IK target, the wrist error, and the arm joints that sat at a limit just before it. With a
reference retarget of the same clip (e.g. an official one) its wrist error is added. Causes:

ref_fails     the reference retarget is far off the targets on this clip as well
ref_bug       the reference motion pops: upper arm and forearm spin together while the hand stays still
              (a twist flip in the capture), or an IK target teleports
joint_limits  the arm sits at a joint limit: the wrist error grows until the arm snaps once the target is
              back in range, or the shoulder flips over while pinned
wrist_cap     only the hand turns fast; the per-frame step threshold of the wrist joints (frame_smooth)
              hands the rest of the turn to the shoulder, so the elbow swings while its target stays
fast_motion   the human arm really moves this fast and the robot links follow their targets
"""

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from scaleretarget.utils.arm_jumps import ARM_JOINTS, JUMP_SKIP_FRAMES, JUMP_SPEED

SIDES = ("left", "right")
# Robot links whose targets are the human upper arm, forearm and hand.
ARM_LINKS = ("shoulder_roll_link", "elbow_link", "wrist_yaw_link")

FLIP_SPIN, FLIP_STILL = 40.0, 20.0  # rad/s: upper arm and forearm above, hand below
TELEPORT = 8.0  # m/s of an elbow / wrist target
SWING = 4.0  # m/s the elbow / wrist link moves faster than its target
STUCK_CM, PINNED = 15.0, 0.03  # wrist error growth over 8 frames; rad from a limit
HAND_SPIN, ARM_STILL = 35.0, 10.0  # rad/s
REF_FAILS_CM = 20.0  # median wrist error of the reference retarget over the clip
CAUSES = ("ref_fails", "ref_bug", "joint_limits", "wrist_cap", "fast_motion")


def arm_bones(retargeter):
    """Human bones targeted by the upper arm, forearm and hand links, per side."""
    return {s: [retargeter.ik_match_table[f"{s}_{link}"][0] for link in ARM_LINKS] for s in SIDES}


def bone_speeds(frames, bones, fps):
    """Global angular speed (rad/s) of each bone per step, (frames - 1, len(bones))."""
    speeds = []
    for bone in bones:
        q = R.from_quat(np.array([f[bone][1] for f in frames]), scalar_first=True)
        speeds.append((q[:-1].inv() * q[1:]).magnitude() * fps)
    return np.stack(speeds, 1)


def twist_flips(speeds):
    """Steps where the upper arm and forearm spin while the hand stays still: a twist flip in the capture."""
    return (speeds[:, 0] > FLIP_SPIN) & (speeds[:, 1] > FLIP_SPIN) & (speeds[:, 2] < FLIP_STILL)


def link_track(retargeter, motion, targets):
    """Wrist error (frames, 2 sides) in m, and per step how much faster the elbow / wrist link moves than
    its target and the target speed (frames - 1, 2 sides) in m/s. motion is a formatter dict (xyzw root)."""
    model, base = retargeter.model, retargeter.base
    data = mj.MjData(model)
    links = [f"{s}_{link}" for s in SIDES for link in ARM_LINKS[1:]]
    bodies = [(model.body(link).id, retargeter.ik_match_table[link][0]) for link in links]
    root_pos, root_rot, dof = motion["root_pos"], np.asarray(motion["root_rot"])[:, [3, 0, 1, 2]], motion["dof_pos"]
    n = min(len(targets), len(dof))
    pos, tpos = np.zeros((n, 4, 3)), np.zeros((n, 4, 3))
    for i in range(n):
        base.merge_qpos(data.qpos, root_pos[i], root_rot[i], dof[i])
        mj.mj_forward(model, data)
        for k, (bid, bone) in enumerate(bodies):
            pos[i, k], tpos[i, k] = data.xpos[bid], targets[i][bone][0]
    err = np.linalg.norm(pos - tpos, axis=2)[:, [1, 3]]
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=2) * motion["fps"]
    target_speed = np.linalg.norm(np.diff(tpos, axis=0), axis=2) * motion["fps"]
    return err, (speed - target_speed).reshape(-1, 2, 2).max(2), target_speed.reshape(-1, 2, 2).max(2)


def jump_segments(retargeter, frames, motion, reference=None):
    """One dict per arm jump segment of a retargeted clip. retargeter must have had update(extras) for
    the clip; frames are the loader's frames at the pkl's rate. reference: optional (retargeter, motion)
    of another retarget of the same clip, compared against its own targets."""
    fps = float(motion["fps"])
    model, base = retargeter.model, retargeter.base
    targets = retargeter.compute_targets(frames)
    err, excess, target_speed = link_track(retargeter, motion, targets)
    ids = [model.joint(f"{s}_{j}_joint").id for s in SIDES for j in ARM_JOINTS]
    names = [f"{s}_{j}" for s in SIDES for j in ARM_JOINTS]
    arm = np.asarray(motion["dof_pos"])[:, model.jnt_qposadr[ids] - base.num_root_qpos]
    lo, hi = model.jnt_range[ids].T
    pinned = np.minimum(arm - lo, hi - arm) < PINNED
    speed = np.abs(np.diff(arm, axis=0)) * fps
    speed[:JUMP_SKIP_FRAMES] = 0.0
    human = {s: bone_speeds(frames, bones, fps) for s, bones in arm_bones(retargeter).items()}
    if reference is not None:
        ref_retargeter, ref_motion = reference
        ref_err = link_track(ref_retargeter, ref_motion, ref_retargeter.compute_targets(frames))[0]

    steps = np.flatnonzero(speed.max(1) > JUMP_SPEED)
    rows = []
    for g in np.split(steps, np.flatnonzero(np.diff(steps) > 2) + 1) if len(steps) else []:
        k0, k1 = g[0], g[-1]
        j = int(speed[k0:k1 + 1].max(0).argmax())
        side, si = SIDES[j // len(ARM_JOINTS)], j // len(ARM_JOINTS)
        w = slice(max(0, k0 - 2), min(len(target_speed), k1 + 3))
        h = human[side]
        pin = pinned[max(0, k0 - 6):k0 + 1, si * len(ARM_JOINTS):(si + 1) * len(ARM_JOINTS)].any(0)
        row = {
            "seg": f"{k0 + 1}→{k1 + 2}", "side": side, "joint": names[j], "speed": speed[k0:k1 + 1].max(),
            "h_arm": h[w, 0].max(), "h_fore": h[w, 1].max(), "h_hand": h[w, 2].max(),
            "flip": bool(twist_flips(h[w]).any()),
            "target_speed": target_speed[w, si].max(), "excess": excess[w, si].max(),
            "err_at_cm": err[k0:k1 + 2, si].max() * 100,
            "err_trend_cm": (err[k0, si] - err[max(0, k0 - 8), si]) * 100,
            "err_med_cm": np.median(err[:, si]) * 100,
            "pinned": ",".join(ARM_JOINTS[i] for i in np.flatnonzero(pin)),
        }
        if reference is not None:
            row["ref_err_at_cm"] = ref_err[k0:k1 + 2, si].max() * 100
            row["ref_err_med_cm"] = np.median(ref_err[:, si]) * 100
        row["cause"] = classify(row)
        rows.append(row)
    return rows


def classify(row):
    if row.get("ref_err_med_cm", 0.0) > REF_FAILS_CM:
        return "ref_fails"
    if row["flip"] or row["target_speed"] > TELEPORT:
        return "ref_bug"
    if row["excess"] >= SWING or (row["err_trend_cm"] >= STUCK_CM and row["pinned"]):
        return "joint_limits"
    if row["h_hand"] >= HAND_SPIN and row["h_arm"] < ARM_STILL:
        return "wrist_cap"
    return "fast_motion"
