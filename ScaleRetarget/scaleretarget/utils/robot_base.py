import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R


class RobotBase:
    """Maps between MuJoCo qpos and (root_pos, root_rot wxyz, dof_pos).

    floating: qpos = [root_pos(3), root_quat_wxyz(4), dofs]
    planar:   qpos = [x, y, yaw, dofs]; the root body slides on the ground at a fixed height
              (e.g. a wheeled base), so root_pos z is constant and root_rot is a pure yaw.
    """

    def __init__(self, model: mj.MjModel):
        if model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
            self.type = "floating"
            self.num_root_qpos = 7
        elif (
            model.njnt >= 3
            and model.jnt_type[0] == mj.mjtJoint.mjJNT_SLIDE
            and model.jnt_type[1] == mj.mjtJoint.mjJNT_SLIDE
            and model.jnt_type[2] == mj.mjtJoint.mjJNT_HINGE
            and model.jnt_bodyid[0] == model.jnt_bodyid[1] == model.jnt_bodyid[2]
        ):
            self.type = "planar"
            self.num_root_qpos = 3
            self.base_height = float(model.body_pos[model.jnt_bodyid[0]][2])
        else:
            raise NotImplementedError(
                "Robot model must start with a free joint, or with slide x, slide y, hinge yaw on one body."
            )

    def root_to_qpos(self, qpos, root_pos, root_rot):
        """Writes the root pose into qpos in place."""
        if self.type == "floating":
            qpos[:3] = root_pos
            qpos[3:7] = root_rot
        else:
            qpos[:2] = root_pos[:2]
            qpos[2] = yaw_from_quat(root_rot)

    def split_qpos(self, qpos):
        """qpos (..., nq) -> root_pos (..., 3), root_rot wxyz (..., 4), dof_pos (..., nq - num_root_qpos)."""
        qpos = np.asarray(qpos)
        if self.type == "floating":
            return qpos[..., :3], qpos[..., 3:7], qpos[..., 7:]
        root_pos = np.concatenate(
            [qpos[..., :2], np.full(qpos.shape[:-1] + (1,), self.base_height)], axis=-1
        )
        yaw = qpos[..., 2]
        root_rot = np.stack(
            [np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1
        )
        return root_pos, root_rot, qpos[..., 3:]

    def merge_qpos(self, qpos, root_pos, root_rot, dof_pos):
        """Inverse of split_qpos for a single frame, written into qpos in place."""
        self.root_to_qpos(qpos, root_pos, root_rot)
        qpos[self.num_root_qpos:] = dof_pos


def yaw_from_quat(quat_wxyz):
    """Heading of the body x axis projected on the ground plane."""
    mat = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return np.arctan2(mat[1, 0], mat[0, 0])
