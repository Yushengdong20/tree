"""使用共享 move_box 手臂控制器关闭指定夹爪。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxCloseClaw(TimedMockAction):
    """在真实机器人上执行指定侧夹爪的闭合动作。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.side = str(params.get("side", "left")).strip().lower()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    def update(self):
        """读取共享服务并调用 arm_controller.close_claw。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_claw_motion():
            self.log_skip_claw_motion()
            return Status.SUCCESS
        ok = services.arm_controller.close_claw(self.side)
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return f"[{self.config_label}] MoveBoxCloseClaw start: side={self.side}"
