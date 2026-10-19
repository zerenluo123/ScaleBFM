"""g1d workspace check of a human motion, per clip, and the red / yellow / green zones built on it.

red     not usable on g1d: non-upright, airborne/terrain, or down near the floor (sitting, kneeling, deep
        squat) on more than the yellow tolerance of frames
yellow  usable but partly outside the g1d workspace (lift stroke, lean-back, base speed)
green   inside the workspace on (almost) every frame

Lift speed is reported but does not decide the zone. Positions are the loader's (Z-up meters, human scale).
"""

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from scaleretarget.utils.base_path_filter import heading_yaw

# g1d: pelvis_proxy height with the lift at 0, and the total lift stroke. With Hips scale 1.0 the
# IK puts the lift at (human Hips z - LIFT_BASE).
LIFT_BASE = 0.801
LIFT_STROKE = 0.42
LIFT_TOL = 0.02
FLOOR_HIPS_Z = 0.55
PITCH_MIN_DEG = -2.5  # Yaw_Joint (forward pitch) range is -0.04363..2.3562 rad
PITCH_MAX_DEG = 135.0
LIFT_SPEED = 0.06
BASE_SPEED = 1.5
BASE_SPEED_WINDOW = 5  # frames; base speed over ~0.17 s at 30 fps, so mocap jitter does not dominate

NON_UPRIGHT = (
    r"\b(?:crawl|all fours|hands and knees|lying|plank|handstand|standing on hands|flip|rolling|cartwheel"
    r"|somersault|push.?up|swim|prone|superman|flying with wings"
    r"|faint|fall(?:s|ing)? down|falls? (?:to|on|onto) the (?:ground|floor)|rise up|rising up)"
)
TERRAIN = r"climb|ladder|stairs|on a box|onto .*box|from (?:a|the) box|obstacle|vault|parkour"
# Jumps stay in when they are kinematically inside the workspace (in-place jumps, dance hops);
# their vertical speed is handled by the root height filter, not by zoning.
JUMP = r"jump"
OUT_WS_FRAC = 0.01
# Hips below FLOOR_HIPS_Z is over 0.2 m under the lowest lift: the robot cannot follow those frames at all,
# so they get the same tolerance as frames outside the workspace. A brief floor touch and sitting down do not
# separate by duration (longest stretches spread evenly over 0.5-10 s in both datasets).
FLOOR_FRAC = OUT_WS_FRAC
# A deep squat keeps the Hips above ~0.2 m; lower means sitting or lying on the floor.
SEATED_HIPS_Z = 0.15
OUT_WS_REASONS = {
    "frac_lift_low": "升降到底",
    "frac_lift_high": "升降到顶",
    "frac_pitch_back": "后仰超限",
    "frac_pitch_fwd": "前俯超限",
    "frac_base_fast": "底盘超速",
}
REASON_EN = {
    "非直立": "non-upright", "地形": "terrain", "跳跃（出工作空间）": "jump out of workspace", "贴地": "near floor",
    "升降到底": "lift at bottom", "升降到顶": "lift at top", "后仰超限": "lean-back limit",
    "前俯超限": "lean-forward limit", "底盘超速": "base too fast", "骨架不支持": "unsupported skeleton",
    "参考数据跳变": "reference pops",
}


def rotation_offsets(correspondence):
    """Rotation offsets of the pelvis_proxy and torso_link targets in a g1d correspondence yaml."""
    table = yaml.safe_load(open(correspondence))["ik_match_table"]
    return tuple(R.from_quat(table[link][4], scalar_first=True) for link in ("pelvis_proxy", "torso_link"))


def torso_target_pitch(hips_rot, chest_rot, hips_offset, chest_offset):
    """Forward pitch (deg) the torso_link target asks of Yaw_Joint, measured in the heading frame of
    the pelvis_proxy target; matches the IK pitch closely when the joint is not at a limit."""
    yaw = heading_yaw(hips_rot * hips_offset)
    heading = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)])
    z = (chest_rot * chest_offset).apply([0.0, 0.0, 1.0])
    return np.degrees(np.arctan2((z * heading).sum(1), z[:, 2]))


def frac(mask):
    return float(np.mean(mask)) if mask.size else 0.0


def clip_metrics(hips, hips_rot, chest_rot, hands_z, fps, offsets):
    """Workspace metrics of one clip. hips (T, 3) and hands_z (T, 2) in meters, hips_rot / chest_rot
    Rotations of the bones the pelvis_proxy / torso_link targets follow, offsets their rotation offsets."""
    hz = hips[:, 2]
    pitch = torso_target_pitch(hips_rot, chest_rot, *offsets)
    v_lift = np.abs(np.diff(hz)) * fps
    k = BASE_SPEED_WINDOW
    v_base = np.linalg.norm(hips[k:, :2] - hips[:-k, :2], axis=1) * fps / k if len(hips) > k else np.zeros(0)

    lift_low = hz < LIFT_BASE - LIFT_TOL
    lift_high = hz > LIFT_BASE + LIFT_STROKE + LIFT_TOL
    pitch_back = pitch < PITCH_MIN_DEG
    pitch_fwd = pitch > PITCH_MAX_DEG
    fast = np.zeros_like(hz, dtype=bool)
    fast[k:] = v_base > BASE_SPEED
    out_ws = lift_low | lift_high | pitch_back | pitch_fwd | fast

    return {
        "frames_30": int(len(hz)),
        "hips_z_min": float(hz.min()),
        "hips_z_p50": float(np.median(hz)),
        "hips_z_max": float(hz.max()),
        "frac_lift_low": frac(lift_low),
        "frac_lift_high": frac(lift_high),
        "frac_floor": frac(hz < FLOOR_HIPS_Z),
        "pitch_min": float(pitch.min()),
        "pitch_max": float(pitch.max()),
        "frac_pitch_back": frac(pitch_back),
        "frac_pitch_fwd": frac(pitch_fwd),
        "lift_speed_p95": float(np.percentile(v_lift, 95)) if v_lift.size else 0.0,
        "frac_lift_fast": frac(v_lift > LIFT_SPEED),
        "base_speed_max": float(v_base.max()) if v_base.size else 0.0,
        "frac_base_fast": frac(fast),
        "frac_out_ws": frac(out_ws),
        "hand_z_min": float(hands_z.min()),
        "hand_z_max": float(hands_z.max()),
        "base_path_m": float(np.linalg.norm(np.diff(hips[:, :2], axis=0), axis=1).sum()),
    }


def assign_zones(df, upright_text, terrain_text, jump_text=None):
    """(zone, reason) per row of clip_metrics columns. upright_text is searched for non-upright motion,
    terrain_text for terrain, jump_text (default terrain_text) for jumps; later rules override earlier ones."""
    jump_text = terrain_text if jump_text is None else jump_text
    zone = pd.Series("green", index=df.index)
    reason = pd.Series("", index=df.index)

    out_ws = df.frac_out_ws > OUT_WS_FRAC
    zone[out_ws] = "yellow"
    reason[out_ws] = df.loc[out_ws, list(OUT_WS_REASONS)].idxmax(axis=1).map(OUT_WS_REASONS)

    floor = (df.frac_floor > FLOOR_FRAC) | (df.hips_z_min < SEATED_HIPS_Z)
    zone[floor], reason[floor] = "red", "贴地"
    jump = jump_text.str.contains(JUMP, case=False, regex=True) & out_ws
    zone[jump], reason[jump] = "red", "跳跃（出工作空间）"
    terrain = terrain_text.str.contains(TERRAIN, case=False, regex=True)
    zone[terrain], reason[terrain] = "red", "地形"
    upright = upright_text.str.contains(NON_UPRIGHT, case=False, regex=True)
    zone[upright], reason[upright] = "red", "非直立"
    return zone, reason


def write_lists(zones, rel_paths, out_dir):
    """green.txt / yellow.txt / red.txt with the clips' paths relative to the BVH root, as
    loader.config.file_list reads them."""
    for z in ["red", "yellow", "green"]:
        with open(f"{out_dir}/{z}.txt", "w") as f:
            f.write("\n".join(rel_paths[zones == z]) + "\n")
