import numpy as np
from scipy.spatial.transform import Rotation as R

from scaleretarget.loader.base_loader import BaseLoader
from scaleretarget.utils.lafan_vendor.extract import read_bvh_multi_channel
from scaleretarget.utils.lafan_vendor.utils import quat_fk, quat_mul
from scaleretarget.utils.resampling import resample_bvh_motion


class SomaLoader(BaseLoader):
    """BONES-SEED SOMA Uniform BVH: Y-up, centimeters, Root and Hips both carry 6 channels."""

    def _load_sample(self, sample_path):

        bvh_data = read_bvh_multi_channel(sample_path)

        aligned_fps = resample_bvh_motion(bvh_data, self.target_fps)

        global_data = quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)

        # Same Y-up to Z-up rotation as soma-retargeter's "Mujoco" facing direction,
        # so results share the world frame of the official BONES G1 CSVs.
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        frames = []
        for frame in range(bvh_data.pos.shape[0]):
            result = {}
            for i, bone in enumerate(bvh_data.bones):
                orientation = quat_mul(rotation_quat, global_data[0][frame, i])
                position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
                result[bone] = (position, orientation)
            frames.append(result)

        extras = {
            "fps": aligned_fps,
        }

        return frames, extras
