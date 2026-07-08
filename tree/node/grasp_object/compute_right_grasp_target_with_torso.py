"""通过腰部采样计算右臂抓取目标和腰部辅助位姿。"""

import py_trees
from py_trees.common import Status

from .right_grasp_target_utils import RightGraspTargetComputer
from ..base import TimedMockAction


class ComputeRightGraspTargetWithTorso(TimedMockAction):
    """使用腰部采样选择可达右臂抓取目标，并写入腰部目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.computer = RightGraspTargetComputer(
            config_label=config_label,
            ros_node=ros_node,
            blackboard=self.blackboard,
            params=params,
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        try:
            self.ros_node.set_live_runtime(
                self.config_label,
                "TORSO_SAMPLE",
                "Sampling torso-assisted right grasp targets",
            )
            selected = self.computer.compute_torso_sample_target()
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 腰部采样计算右臂抓取目标失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.clear_live_runtime()
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算腰部辅助右臂抓取目标: "
            f"torso={sample['label']}, grasp={selected['grasp_target']}, "
            f"left_shift={selected['left_shift_target']}"
        )
        return Status.SUCCESS

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeRightGraspTargetWithTorso start: "
            f"base_grasp_poses_key={self.computer.base_grasp_poses_key}"
        )
