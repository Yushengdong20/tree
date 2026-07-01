"""使用共享手臂控制器张开指定夹爪。"""

import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class OpenClaw(TimedMockAction):
    """在真实机器人上执行指定侧夹爪的张开动作。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.side = str(params.get("side", "both")).strip().lower()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_claw_motion():
            self.log_skip_claw_motion()
            return Status.SUCCESS
        if not hasattr(services, "arm_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 中没有 arm_controller: key={self.services_key}"
            )
            return Status.FAILURE
        ok = services.arm_controller.open_claw(self.side)
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        return f"[{self.config_label}] OpenClaw start: key={self.services_key}, side={self.side}"
