"""使用共享手臂控制器，把指定夹爪移动到 blackboard 中的目标点。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxClawToPoint(TimedMockAction):
    """驱动左手或右手夹爪移动到预先计算好的空间点位。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.point_key = str(params.get("point_key", "")).strip()
        self.side = str(params.get("side", "left")).strip().lower()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.point_key:
            self.blackboard.register_key(key=self.point_key, access=py_trees.common.Access.READ)

    def update(self):
        """读取目标点并调用共享 arm_controller 执行夹爪移动。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if not self.point_key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 params.point_key")
            return Status.FAILURE
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        point = self.blackboard.get(self.point_key) if self.blackboard.exists(self.point_key) else None
        if services is None or point is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services 或目标点缺失: services={services is not None}, point={point is not None}"
            )
            return Status.FAILURE
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        if self.side == "left":
            ok = services.arm_controller.move_left_claw_to_point(point)
        elif self.side == "right":
            ok = services.arm_controller.move_right_claw_to_point(point)
        else:
            self.ros_node.get_logger().error(f"[{self.config_label}] 不支持的 side: {self.side}")
            return Status.FAILURE
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return (
            f"[{self.config_label}] MoveBoxClawToPoint start: "
            f"side={self.side}, point_key={self.point_key}"
        )
