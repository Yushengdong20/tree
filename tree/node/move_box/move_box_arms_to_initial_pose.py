"""将双臂移动回 move_box 初始准备位姿。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxArmsToInitialPose(TimedMockAction):
    """复用 arm_controller 的初始位姿逻辑，让双臂回到观察/准备姿态。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        services.arm_controller.move_to_initial_pose()
        return Status.SUCCESS
