"""Pops of the reference arm in a BVH: the steps where the motion data itself jumps.

A pop is a twist flip or a hand flip (scaleretarget/utils/jump_audit.py) at the retarget rate whose spinning
bone (upper arm, hand) turns faster than POP_SPIN at the capture's own rate within the step. Real fast hands
(shaking off water, flapping, wiggling) stay under ~80 rad/s there and turn smoothly over the native frames;
capture pops turn the hand 120-180 deg within one to three native frames (100-300 rad/s, the arm still).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from scaleretarget.utils.jump_audit import SIDES, arm_links, bone_speeds, hand_flips, twist_flips
from scaleretarget.utils.lafan_vendor.extract import read_bvh_multi_channel
from scaleretarget.utils.lafan_vendor.utils import quat_fk

POP_SPIN = 100.0  # rad/s at the native frame rate


def table_arm_bones(table):
    """Upper arm, forearm and hand bones of each side in an ik_match_table."""
    return [[table[link][0] for link in arm_links(table, s)] for s in SIDES]


def native_peaks(path, candidates, fps, frame_time):
    """Fastest native-rate turn (rad/s) of each candidate's bone within its step at fps.
    candidates: (bone, step) pairs."""
    bvh = read_bvh_multi_channel(path)
    quats = quat_fk(bvh.quats, bvh.pos, bvh.parents)[0]
    turn = {}
    for bone in {b for b, _ in candidates}:
        q = R.from_quat(quats[:, bvh.bones.index(bone)], scalar_first=True)
        turn[bone] = (q[:-1].inv() * q[1:]).magnitude() / frame_time
    peaks = []
    for bone, s in candidates:
        n0 = int(np.floor(s / fps / frame_time))
        n1 = max(min(int(np.ceil((s + 1) / fps / frame_time)), len(turn[bone])), n0 + 1)
        peaks.append(float(turn[bone][n0:n1].max()))
    return np.array(peaks)


def reference_pops(path, frames, fps, frame_time, arm_bones):
    """Counts of twist flips and hand flips at fps, and of those that are capture pops (see the module doc),
    with the fastest native turn among them (rad/s, 0 without flips). arm_bones: table_arm_bones."""
    candidates = []
    counts = {"ref_twist_flips": 0, "ref_hand_flips": 0}
    for bones in arm_bones:
        speeds = bone_speeds(frames, bones, fps)
        for key, flips, bone in (("ref_twist_flips", twist_flips, bones[0]), ("ref_hand_flips", hand_flips, bones[2])):
            steps = np.where(flips(speeds))[0]
            counts[key] += len(steps)
            candidates += [(bone, s) for s in steps]
    peaks = native_peaks(path, candidates, fps, frame_time) if candidates else np.zeros(0)
    return counts | {"ref_pops": int((peaks > POP_SPIN).sum()), "ref_pop_peak": float(peaks.max(initial=0.0))}
