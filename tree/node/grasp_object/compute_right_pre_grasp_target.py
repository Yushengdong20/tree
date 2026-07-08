"""在执行预抓取动作前计算右臂预抓取目标。"""

import py_trees
from py_trees.common import Status

from .right_grasp_target_utils import RightGraspTargetComputer
from ..base import TimedMockAction


class ComputeRightPreGraspTarget(TimedMockAction):
    """基于最终抓取目标计算预抓取目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.pregrasp_pose_key = str(
            params.get("pregrasp_pose_key", "grasp_object_pregrasp_pose")
        ).strip()
        if not self.pregrasp_pose_key:
            raise ValueError("pregrasp_pose_key 不能为空")
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(
            key=self.pregrasp_pose_key,
            access=py_trees.common.Access.WRITE,
        )
        self.computer = RightGraspTargetComputer(
            config_label=config_label,
            ros_node=ros_node,
            blackboard=self.blackboard,
            params=params,
        )
        self.blackboard.register_key(
            key=self.computer.selected_grasp_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.computer.selected_base_grasp_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.computer.selected_rotate_z_180_key,
            access=py_trees.common.Access.READ,
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
                "PREGRASP_COMPUTE",
                "Computing right pregrasp target from final grasp target",
            )
            self.computer.refresh_selected_grasp_target()
            self.computer.compute_pregrasp_target(self.pregrasp_pose_key)
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 计算右臂预抓取目标失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.clear_live_runtime()
        return Status.SUCCESS

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeRightPreGraspTarget start: "
            f"pregrasp_pose_key={self.pregrasp_pose_key}"
        )
