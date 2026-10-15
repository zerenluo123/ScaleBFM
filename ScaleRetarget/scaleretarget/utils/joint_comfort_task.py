import numpy as np
from mink.tasks.task import Task


class JointComfortTask(Task):
    """Pulls joints back toward the middle of a reference range once they leave its comfort zone.

    Port of soma-retargeter's IKSmoothJointFilter. The comfort zone is the reference range shrunk by
    `offset` rad on both sides; outside it the residual (q - center) is eased in with
    1 - exp(-excess^6.5), so joints inside the zone are untouched. The reference range is given
    explicitly rather than read from the model, so narrowing the hard limits does not move the zone.

    `joints` maps joint name -> (mask, lower, upper); the per-joint cost is weight * mask.
    """

    def __init__(self, model, joints, weight, offset):
        joint_ids = [model.joint(name).id for name in joints]
        self.qposadr = model.jnt_qposadr[joint_ids]
        self.dofadr = model.jnt_dofadr[joint_ids]
        self.nv = model.nv
        mask, lower, upper = np.array(list(joints.values()), dtype=float).T
        self.center = (lower + upper) / 2
        self.half_zone = (upper - lower) / 2 - offset
        super().__init__(cost=weight * mask, gain=1.0, lm_damping=0.0)

    def compute_error(self, configuration):
        err = configuration.q[self.qposadr] - self.center
        excess = np.maximum(np.abs(err) - self.half_zone, 0.0)
        return err * (1.0 - np.exp(-excess**6.5))

    def compute_jacobian(self, configuration):
        del configuration
        jac = np.zeros((len(self.dofadr), self.nv))
        jac[np.arange(len(self.dofadr)), self.dofadr] = 1.0
        return jac
