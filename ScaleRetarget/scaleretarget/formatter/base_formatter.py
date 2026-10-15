import mujoco as mj
from loguru import logger

from scaleretarget.utils.robot_base import RobotBase

class BaseFormatter:
    def __init__(self, config):
        self.config = config
        self.quat_order = config.get('quat_order', 'xyzw')
        logger.info(f"[Formatter] Formatter has quat order: {self.quat_order}")
        self.base = RobotBase(mj.MjModel.from_xml_path(config.robot.robot_xml_path))


    def format(self, qpos_list, extras):
        root_pos, root_rot, dof_pos = self.base.split_qpos(qpos_list) # wxyz
        root_pos, root_rot, dof_pos = root_pos.copy(), root_rot.copy(), dof_pos.copy()
        if self.quat_order == 'xyzw':
            root_rot[:, [0,1,2,3]] = root_rot[:, [1,2,3,0]]
        # num_frames = root_pos.shape[0]
        
        motion_data = {
            "fps": extras['fps'],
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
        }
        
        return motion_data
