import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from scaleretarget.utils.root_height_filter import rate_limit


def lowpass(x, fps, cutoff_hz):
    if len(x) <= 9:
        return x
    b, a = butter(2, cutoff_hz / (fps / 2))
    return filtfilt(b, a, x, axis=0, method="gust")


def heading_yaw(pelvis_rot):
    """(n,) yaw (rad) a pelvis Rotation faces: the ground direction square to its left axis. The forward
    axis points at the floor in a deep forward bend and its ground projection flips by 180 deg there;
    the left axis stays level through forward and backward bends."""
    left = pelvis_rot.apply([0.0, 1.0, 0.0])
    return np.arctan2(-left[:, 0], left[:, 1])


def plan_base_path(xy, pelvis_yaw, fps, cutoff_hz, max_speed, max_turn_rate=None):
    """(n, 2) positions and (n,) headings a differential drive can follow: the heading is the smoothed
    pelvis yaw, turned no faster than max_turn_rate if given, and of the smoothed human motion only the
    part along the heading is kept, at most max_speed; motion sideways to it is dropped."""
    heading = lowpass(np.unwrap(pelvis_yaw), fps, cutoff_hz)
    if max_turn_rate is not None:
        heading = rate_limit(heading, max_turn_rate / fps)
    smooth = lowpass(np.asarray(xy, dtype=np.float64), fps, cutoff_hz)
    mid = 0.5 * (heading[:-1] + heading[1:])
    axis = np.column_stack([np.cos(mid), np.sin(mid)])
    along = np.clip(np.sum(np.diff(smooth, axis=0) * axis, axis=1), -max_speed / fps, max_speed / fps)
    path = np.vstack([smooth[:1], smooth[:1] + np.cumsum(along[:, None] * axis, axis=0)])
    return path, heading


def filter_base_path(frames, root_name, root_rot_offset, fps, cutoff_hz, max_speed, max_turn_rate=None):
    """Move the whole human so the root follows the chassis path and heading: every bone is shifted
    horizontally by the path offset and turned about the vertical through the root by the heading
    offset, the same way the root height filter shifts every bone vertically."""
    xy = np.array([frame[root_name][0][:2] for frame in frames])
    root_rot = R.from_quat(np.array([frame[root_name][1] for frame in frames]), scalar_first=True)
    pelvis_yaw = heading_yaw(root_rot * root_rot_offset)
    path, heading = plan_base_path(xy, pelvis_yaw, fps, cutoff_hz, max_speed, max_turn_rate)
    pivot = np.column_stack([path, np.zeros(len(path))])
    shift = pivot - np.column_stack([xy, np.zeros(len(xy))])
    turn = R.from_rotvec(np.outer(heading - np.unwrap(pelvis_yaw), [0.0, 0.0, 1.0]))
    moved = {}
    for bone in frames[0]:
        pos = np.array([frame[bone][0] for frame in frames], dtype=np.float64)
        rot = R.from_quat(np.array([frame[bone][1] for frame in frames]), scalar_first=True)
        moved[bone] = (pivot + turn.apply(pos + shift - pivot), (turn * rot).as_quat(scalar_first=True))
    return [{bone: (pos[i], quat[i]) for bone, (pos, quat) in moved.items()} for i in range(len(frames))]
