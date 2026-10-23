import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
from loguru import logger

from scaleretarget.loader.base_loader import BaseLoader
from scaleretarget.utils.resampling import resample_smplx_motion
from scaleretarget.utils.shape_optimizer import cached_smplx_shape
from scaleretarget.utils.smplx import SmplxModelCache, lowest_vertex_z, run_smplx_inference

class Embody3dLoader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)
        self.body_models = SmplxModelCache(self.config.body_model_path)
        self._optimize_shape()

    def _load_sample(self, sample_path):
        smplx_data = joblib.load(sample_path)
        num_frames = smplx_data['pose_body'].shape[0]

        rot_offset = R.from_quat([0.5,0.5,0.5,0.5], scalar_first=True)
        smplx_data['root_orient'] = (rot_offset * R.from_rotvec(smplx_data['root_orient'])).as_rotvec()
        smplx_data['trans'] = rot_offset.apply(smplx_data['trans'])
        trans = torch.tensor(smplx_data["trans"]).float()
        ground_shift = 0.0
        if self.config.get("ground", False):
            ground_shift = -self._lowest_vertex_z(smplx_data, trans)
            trans[:, 2] += ground_shift

        if self.use_optimized_shape:
            betas = self.shape
            body_model = self.body_models.get('neutral')
        else:
            logger.error(f"[Loader] Embody3D only supports retargeting based on shape optimization.")
            import ipdb; ipdb.set_trace()
        smplx_output = run_smplx_inference(
            body_model,
            betas=betas, # (16,)
            global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
            body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
            transl=trans, # (N, 3)
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(), # FIXME: use real hand pose when dex hands are available
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            # expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        human_height = 1.66 + 0.1 * betas[0, 0].item()

        global_orient = smplx_output.global_orient.detach().cpu().numpy().reshape(num_frames, 3)
        full_body_pose = smplx_output.full_pose.detach().cpu().numpy().reshape(num_frames, -1, 3)
        joints = smplx_output.joints.detach().cpu().numpy().reshape(num_frames, -1, 3)
        joint_names = JOINT_NAMES[: len(body_model.parents)]
        parents = body_model.parents

        global_orient, full_body_pose, joints, aligned_fps = resample_smplx_motion(
            global_orient,
            full_body_pose,
            joints,
            30,
            self.target_fps,
        )

        frames = []
        for curr_frame in range(len(global_orient)):
            result = {}
            single_global_orient = global_orient[curr_frame]
            single_full_body_pose = full_body_pose[curr_frame]
            single_joints = joints[curr_frame]
            joint_orientations = []
            for i, joint_name in enumerate(joint_names):
                if i == 0:
                    rot = R.from_rotvec(single_global_orient)
                else:
                    rot = joint_orientations[parents[i]] * R.from_rotvec(
                        single_full_body_pose[i].squeeze()
                    )
                joint_orientations.append(rot)
                result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))


            frames.append(result)


        extras = {
            "fps": aligned_fps,
            "disable_scale_table": self.use_optimized_shape,
            "actual_human_height": human_height,
            "ground_shift": ground_shift,
        }

        return frames, extras

    def _lowest_vertex_z(self, smplx_data, trans):
        """Lowest point of the clip's own body (its neutral betas, all of them) over the tracked frames of the
        clip; the untracked ones (smplx_data['tracked'] False) hold no real pose."""
        betas = np.asarray(smplx_data["betas"]).reshape(1, -1)
        if not hasattr(self, "ground_models"):
            self.ground_models = SmplxModelCache(self.config.body_model_path, num_betas=betas.shape[1])
        tracked = np.asarray(smplx_data.get("tracked", np.ones(len(trans), bool)), bool)
        if not tracked.any():
            tracked[:] = True
        num_frames = int(tracked.sum())
        return lowest_vertex_z(
            self.ground_models.get('neutral'),
            betas=torch.tensor(betas).float(),
            global_orient=torch.tensor(smplx_data["root_orient"][tracked]).float(),
            body_pose=torch.tensor(smplx_data["pose_body"][tracked]).float(),
            transl=trans[torch.from_numpy(tracked)],
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
        )
    
    def _optimize_shape(self):
        self.use_optimized_shape = self.config.use_optimized_shape
        if self.use_optimized_shape:
            logger.info(f'[Loader] Using optimized shape for retargeting.')
            self.shape = cached_smplx_shape(
                robot_type=self.config.shape_robot.robot_type,
                robot_xml_path=self.config.shape_robot.robot_xml_path,
                body_model_path=self.config.body_model_path,
                optim_joint_matches=self.config.optim_joint_matches,
                optim_iterations=self.config.optim_iterations,
            )
