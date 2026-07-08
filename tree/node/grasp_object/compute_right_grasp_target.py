"""在当前腰部位姿下计算右臂抓取目标。"""

import py_trees
from py_trees.common import Status

from .right_grasp_target_utils import RightGraspTargetComputer
from ..base import TimedMockAction


class ComputeRightGraspTarget(TimedMockAction):
    """只使用当前腰部位姿选择右臂抓取目标，不做腰部采样。"""

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
                "GRASP_COMPUTE",
                "Computing right grasp target with current torso",
            )
            selected = self.computer.compute_current_torso_target()
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前腰部下计算右臂抓取目标失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.clear_live_runtime()
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算当前腰部右臂抓取目标: "
            f"torso={sample['label']}, grasp={selected['grasp_target']}, "
            f"left_shift={selected['left_shift_target']}"
        )
        return Status.SUCCESS

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeRightGraspTarget start: "
            f"base_grasp_poses_key={self.computer.base_grasp_poses_key}"
        )
