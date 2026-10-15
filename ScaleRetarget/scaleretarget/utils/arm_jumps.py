import numpy as np

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
JUMP_SPEED, JUMP_SKIP_FRAMES = 20.0, 5


def arm_dof_columns(model, num_root_qpos):
    """Columns of the 14 arm joints in dof_pos (qpos without the root)."""
    return [model.jnt_qposadr[model.joint(f"{s}_{j}_joint").id] - num_root_qpos for s in ("left", "right") for j in ARM_JOINTS]


def arm_speed(dof_pos, columns, fps):
    """Per frame step, the fastest arm joint speed (rad/s); the first JUMP_SKIP_FRAMES steps, where the IK
    converges from its initial guess, are zeroed."""
    speed = np.abs(np.diff(np.asarray(dof_pos)[:, columns], axis=0)).max(axis=1) * fps
    speed[:JUMP_SKIP_FRAMES] = 0.0
    return speed


def jump_ranges(speed, threshold=JUMP_SPEED):
    """1-based frame ranges 'a→b' where the arm speed exceeds threshold; steps <= 2 apart merge."""
    steps = np.flatnonzero(speed > threshold)
    if len(steps) == 0:
        return []
    groups = np.split(steps, np.flatnonzero(np.diff(steps) > 2) + 1)
    return [f"{g[0] + 1}→{g[-1] + 2}" for g in groups]


def arm_jump_ranges(model, dof_pos, num_root_qpos, fps, threshold=JUMP_SPEED):
    return jump_ranges(arm_speed(dof_pos, arm_dof_columns(model, num_root_qpos), fps), threshold)
