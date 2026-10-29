import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit


class DofLock(Limit):
    """Keeps a group of slide/hinge joints where the configuration already has them: Δq = 0 on their dofs.

    Used for a wheeled chassis whose pose is planned before the IK (base_path_filter): the planned
    x, y, yaw are written into the configuration each frame and the IK solves only the other joints,
    so it cannot slide the chassis sideways or turn it off the planned heading to reach the hands.
    """

    def __init__(self, model, joint_names):
        dofadr = [model.jnt_dofadr[model.joint(n).id] for n in joint_names]
        for n in joint_names:
            assert int(model.jnt_type[model.joint(n).id]) in (int(mj.mjtJoint.mjJNT_SLIDE), int(mj.mjtJoint.mjJNT_HINGE))
        rows = np.zeros((len(dofadr), model.nv))
        rows[np.arange(len(dofadr)), dofadr] = 1.0
        self.G = np.vstack([rows, -rows])
        self.h = np.zeros(2 * len(dofadr))

    def compute_qp_inequalities(self, configuration, dt):
        del configuration, dt
        return Constraint(G=self.G, h=self.h)
