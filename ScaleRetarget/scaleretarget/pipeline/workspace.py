"""g1d workspace check of a human motion, per clip, and the red / yellow / green zones built on it.

red     not usable on g1d: non-upright (by text or a torso flipped over), airborne, terrain (by text or the
        hips held above the lift top), down near the floor on any frame, or seated / kneeling / crouched
        (by text or the hips low most of the clip)
yellow  usable but partly outside the g1d workspace (lift stroke, lean-back, base speed)
green   inside the workspace on (almost) every frame

assign_raised_zones (opt-in, for a retarget that raises the body to the lowest lift) keeps near-floor
and seated clips that do not lie on the floor, and sends clips that hold a lean g1d cannot follow to red.

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
    r"\b(?:crawl|all fours|hands and knees|lying|plank|handstand|standing on hands|flip|back ?flip|rolling"
    r"|cart ?wheel|handspring"
    r"|somersault|push.?up|swim|prone|superman|flying with wings"
    r"|faint|fall(?:s|ing)? down|falls? (?:to|on|onto) the (?:ground|floor)|rise up|rising up)"
)
TERRAIN = r"climb|ladder|stairs|on a box|onto .*box|from (?:a|the) box|obstacle|vault|parkour"
# Jumps stay in when they are kinematically inside the workspace (in-place jumps, dance hops);
# their vertical speed is handled by the root height filter, not by zoning.
JUMP = r"jump"
SEATED = r"\bsit(?:s|ting)?\b|\bseated\b|kneel"
OUT_WS_FRAC = 0.01
# Hips below FLOOR_HIPS_Z is over 0.2 m under the lowest lift: the robot cannot follow that frame at all.
# Sitting on a chair keeps the Hips at 0.55-0.65 m, so a seated clip has its median below SEATED_P50_Z;
# picking something up from the floor only dips there.
SEATED_P50_Z = 0.70
# A jump keeps the Hips above the lift top for well under a second; longer, the person stands on something.
TERRAIN_RUN_S = 1.0
# Deep forward bends stop near 140 deg; flips, cartwheels and handsprings pass through 150-180.
FLIP_PITCH_DEG = 150.0
BACK_ARCH_DEG = -45.0
# Raised clips (assign_raised_zones): a near-floor or seated clip is kept with the whole body raised so the
# hips stay at or above LIFT_BASE (robot.root_height_filter.min_height), unless it lies on the floor.
# Frames with the torso over 60 deg from upright pile up at hips 0.10-0.35 m (lying) and 0.35-0.70 m
# (kneeling, bent over) in Embody3D, with almost none in between.
PRONE_HIPS_Z = 0.35
PRONE_DEG = 45.0
# g1d has no joint to lean back past PITCH_MIN_DEG or to the side at all; a lean held past these for
# longer than HOLD_S is done on purpose. 5% of the Embody3D green / yellow clips lean back past
# HELD_BACK_DEG on their worst frame.
HELD_BACK_DEG = -19.0
HELD_SIDE_DEG = 25.0
HOLD_S = 2.0
RAISED_FROM = ("贴地", "坐跪")
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
    "参考数据跳变": "reference pops", "坐跪": "seated / kneeling", "参考数据缺帧": "untracked frames",
    "趴躺伏地": "lying on the floor", "后仰过大": "held lean-back", "侧倾过大": "held side lean",
    "抬高救回": "raised to the lift",
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


def torso_target_side_lean(hips_rot, chest_rot, hips_offset, chest_offset):
    """Side lean (deg) of the torso_link target: angle of its up axis out of the plane of the pelvis_proxy
    heading and the vertical, whatever the forward bend."""
    yaw = heading_yaw(hips_rot * hips_offset)
    left = np.column_stack([-np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)])
    z = (chest_rot * chest_offset).apply([0.0, 0.0, 1.0])
    return np.degrees(np.arcsin(np.clip((z * left).sum(1), -1.0, 1.0)))


def torso_metrics(hips, hips_rot, chest_rot, hands_z, fps, offsets):
    """Metrics of assign_raised_zones for one clip (inputs as in clip_metrics): frames lying on the
    floor, the longest held lean-back and side lean, and the clip_metrics of the motion raised so the
    hips never go below LIFT_BASE (prefix raised_)."""
    hz = hips[:, 2]
    pitch = torso_target_pitch(hips_rot, chest_rot, *offsets)
    side = torso_target_side_lean(hips_rot, chest_rot, *offsets)
    raise_m = np.clip(LIFT_BASE - hz, 0.0, None)
    raised = hips.copy()
    raised[:, 2] = hz + raise_m
    m = clip_metrics(raised, hips_rot, chest_rot, hands_z + raise_m[:, None], fps, offsets)
    return {
        "prone_frames": int(((hz < PRONE_HIPS_Z) & (np.abs(pitch) > PRONE_DEG)).sum()),
        "back_hold_s": longest_run(pitch < HELD_BACK_DEG) / fps,
        "side_hold_s": longest_run(np.abs(side) > HELD_SIDE_DEG) / fps,
        "raise_max": float(raise_m.max()),
        "frac_raised": frac(raise_m > 0),
        **{f"raised_{k}": m[k] for k in ("frac_out_ws", *OUT_WS_REASONS)},
    }


def frac(mask):
    return float(np.mean(mask)) if mask.size else 0.0


def longest_run(mask):
    """Length of the longest stretch of consecutive True in a boolean array."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if starts.size else 0


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
        "lift_high_run_s": longest_run(lift_high) / fps,
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


def assign_zones(df, upright_text, terrain_text, jump_text=None, seated_text=None):
    """(zone, reason) per row of clip_metrics columns. upright_text is searched for non-upright motion,
    terrain_text for terrain, jump_text (default terrain_text) for jumps, seated_text (default none) for
    sitting and kneeling; later rules override earlier ones."""
    jump_text = terrain_text if jump_text is None else jump_text
    zone = pd.Series("green", index=df.index)
    reason = pd.Series("", index=df.index)

    out_ws = df.frac_out_ws > OUT_WS_FRAC
    zone[out_ws] = "yellow"
    reason[out_ws] = df.loc[out_ws, list(OUT_WS_REASONS)].idxmax(axis=1).map(OUT_WS_REASONS)

    seated = df.hips_z_p50 < SEATED_P50_Z
    if seated_text is not None:
        seated |= seated_text.str.contains(SEATED, case=False, regex=True)
    zone[seated], reason[seated] = "red", "坐跪"
    floor = df.hips_z_min < FLOOR_HIPS_Z
    zone[floor], reason[floor] = "red", "贴地"
    jump = jump_text.str.contains(JUMP, case=False, regex=True) & out_ws
    zone[jump], reason[jump] = "red", "跳跃（出工作空间）"
    terrain = terrain_text.str.contains(TERRAIN, case=False, regex=True)
    terrain |= df.lift_high_run_s >= TERRAIN_RUN_S
    zone[terrain], reason[terrain] = "red", "地形"
    upright = upright_text.str.contains(NON_UPRIGHT, case=False, regex=True)
    upright |= (df.pitch_max > FLIP_PITCH_DEG) | (df.pitch_min < BACK_ARCH_DEG)
    zone[upright], reason[upright] = "red", "非直立"
    return zone, reason


def assign_raised_zones(df):
    """(zone, reason, raised) per row of assign_zones' zone and reason plus torso_metrics columns, for a
    retarget with robot.root_height_filter.min_height = LIFT_BASE. Every kept clip is zoned by its raised
    motion; near-floor and seated clips are kept too (raised) unless they lie on the floor; any kept clip
    that holds a side lean or a lean-back g1d cannot follow goes red. Rows without torso_metrics are left
    as they are."""
    zone, reason = df.zone.copy(), df.reason.copy()
    measured = df.prone_frames.notna()
    raised = measured & (zone == "red") & reason.isin(RAISED_FROM)
    kept = raised | (measured & zone.isin(["green", "yellow"]))
    out_ws = df.raised_frac_out_ws > OUT_WS_FRAC
    worst = df[[f"raised_{k}" for k in OUT_WS_REASONS]].fillna(0).idxmax(axis=1).str.removeprefix("raised_")
    zone[kept] = np.where(out_ws[kept], "yellow", "green")
    reason[kept] = np.where(out_ws[kept], worst[kept].map(OUT_WS_REASONS), np.where(raised[kept], "抬高救回", ""))
    rules = [
        ("侧倾过大", kept & (df.side_hold_s > HOLD_S)),
        ("后仰过大", kept & (df.back_hold_s > HOLD_S)),
        ("趴躺伏地", raised & (df.prone_frames > 0)),
    ]
    for name, mask in rules:
        zone[mask], reason[mask] = "red", name
    return zone, reason, raised & (zone != "red")


def write_lists(zones, rel_paths, out_dir):
    """green.txt / yellow.txt / red.txt with the clips' paths relative to the BVH root, as
    loader.config.file_list reads them."""
    for z in ["red", "yellow", "green"]:
        with open(f"{out_dir}/{z}.txt", "w") as f:
            f.write("\n".join(rel_paths[zones == z]) + "\n")
