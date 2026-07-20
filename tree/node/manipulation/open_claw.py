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
        self.side_key = str(params.get("side_key", "")).strip()
        self.torque = self._optional_float(params.get("torque", ""))
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.side_key:
            self.blackboard.register_key(key=self.side_key, access=py_trees.common.Access.READ)

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
        try:
            side = self._resolve_side()
        except Exception as exc:
            self.ros_node.get_logger().error(f"[{self.config_label}] 解析夹爪侧别失败: {exc}")
            return Status.FAILURE
        ok = services.arm_controller.open_claw(side, effort=self.torque)
        return Status.SUCCESS if ok else Status.FAILURE

    def _resolve_side(self):
        """读取动态夹爪侧别，side_key 存在时覆盖静态 side。"""
        side = self.side
        if self.side_key:
            if not self.blackboard.exists(self.side_key):
                raise RuntimeError(f"key={self.side_key} 不存在")
            side = str(self.blackboard.get(self.side_key)).strip().lower()
        if side not in ("left", "right", "both"):
            raise ValueError(f"夹爪侧别非法: {side!r}")
        return side

    def describe_start(self):
        return (
            f"[{self.config_label}] OpenClaw start: key={self.services_key}, "
            f"side={self.side}, side_key={self.side_key or '<none>'}, "
            f"torque={self.torque if self.torque is not None else '<default>'}"
        )

    @staticmethod
    def _optional_float(value):
        """空值表示沿用 ArmController 的全局 claw_effort。"""
        if value is None or str(value).strip() == "":
            return None
        return float(value)
