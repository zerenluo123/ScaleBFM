import numpy as np
from mink.tasks.task import Task


class FrameSmoothTask(Task):
    """Penalises how far hinge joints move from the configuration the motion frame started from.

    With d = q - q_anchor the error is d * (1 - exp(-(|d| / threshold)^power)): steps well below
    threshold (radians per motion frame) are left free, larger ones are pulled back, so the solver
    no longer swaps arm branches within a frame while ordinary motion is tracked without lag.
    threshold 0 gives the plain quadratic penalty on d. Without an anchor (first frame) it is inactive.
    """

    def __init__(self, model, joint_names, weight, threshold=0.0, power=4.0):
        joint_ids = [model.joint(name).id for name in joint_names]
        self.qposadr = model.jnt_qposadr[joint_ids]
        self.dofadr = model.jnt_dofadr[joint_ids]
        self.nv = model.nv
        self.threshold = threshold
        self.power = power
        self.anchor = None
        super().__init__(cost=np.full(len(joint_ids), weight), gain=1.0, lm_damping=0.0)

    def set_anchor(self, q):
        self.anchor = None if q is None else np.array(q)[self.qposadr]

    def _shape(self, d):
        """Error and its derivative w.r.t. d."""
        if self.threshold <= 0:
            return d, np.ones_like(d)
        x = (np.abs(d) / self.threshold) ** self.power
        ease = np.exp(-x)
        return d * (1.0 - ease), 1.0 - ease + self.power * x * ease

    def compute_error(self, configuration):
        if self.anchor is None:
            return np.zeros(len(self.qposadr))
        return self._shape(configuration.q[self.qposadr] - self.anchor)[0]

    def compute_jacobian(self, configuration):
        jac = np.zeros((len(self.dofadr), self.nv))
        if self.anchor is not None:
            slope = self._shape(configuration.q[self.qposadr] - self.anchor)[1]
            jac[np.arange(len(self.dofadr)), self.dofadr] = slope
        return jac
