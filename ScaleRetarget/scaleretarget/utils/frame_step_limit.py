import mujoco as mj
import numpy as np
from mink.limits.limit import Constraint, Limit


class FrameStepLimit(Limit):
    """Bounds the summed motion of a group of slide/hinge joints between consecutive motion frames.

    mink's VelocityLimit acts per IK iteration with the model timestep, and the retargeter runs a
    variable number of iterations per frame, so it cannot express a per-frame speed. This limit is
    anchored at the configuration the frame started from: |sum(q - q_anchor)| <= max_step.
    """

    def __init__(self, model, joint_names, max_step):
        self.qposadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
        dofadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in joint_names])
        for n in joint_names:
            assert int(model.jnt_type[model.joint(n).id]) in (int(mj.mjtJoint.mjJNT_SLIDE), int(mj.mjtJoint.mjJNT_HINGE))
        self.row = np.zeros((1, model.nv))
        self.row[0, dofadr] = 1.0
        self.max_step = max_step
        self.anchor = None

    def set_anchor(self, q):
        self.anchor = None if q is None else float(q[self.qposadr].sum())

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        if self.anchor is None:
            return Constraint()
        moved = configuration.q[self.qposadr].sum() - self.anchor
        G = np.vstack([self.row, -self.row])
        h = np.array([self.max_step - moved, self.max_step + moved])
        return Constraint(G=G, h=np.maximum(h, 0.0))
