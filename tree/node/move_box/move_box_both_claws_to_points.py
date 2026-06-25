"""使用共享手臂控制器同步移动左右夹爪到 blackboard 中的两个目标点。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxBothClawsToPoints(TimedMockAction):
    """将左右夹爪目标点一次性下发给双臂事件，避免单手先动改变箱体姿态。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.left_point_key = str(params.get("left_point_key", "")).strip()
        self.right_point_key = str(params.get("right_point_key", "")).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.left_point_key:
            self.blackboard.register_key(key=self.left_point_key, access=py_trees.common.Access.READ)
        if self.right_point_key:
            self.blackboard.register_key(key=self.right_point_key, access=py_trees.common.Access.READ)

    def update(self):
        """读取左右目标点并同步下发一次双臂事件。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if not self.left_point_key or not self.right_point_key:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 缺少 left_point_key 或 right_point_key"
            )
            return Status.FAILURE

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        left_point = self.blackboard.get(self.left_point_key) if self.blackboard.exists(self.left_point_key) else None
        right_point = self.blackboard.get(self.right_point_key) if self.blackboard.exists(self.right_point_key) else None
        if services is None or left_point is None or right_point is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services 或双手目标点缺失: "
                f"services={services is not None}, left={left_point is not None}, right={right_point is not None}"
            )
            return Status.FAILURE
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        arm_controller = services.arm_controller
        left_ee_point = arm_controller.claw_point_to_end_effector_point(left_point, "left")
        right_ee_point = arm_controller.claw_point_to_end_effector_point(right_point, "right")
        if left_ee_point is None or right_ee_point is None:
            return Status.FAILURE

        left_ypr = arm_controller.get_initial_left_ypr()
        right_ypr = arm_controller.get_initial_right_ypr()
        left_target = [
            left_ee_point[0],
            left_ee_point[1],
            left_ee_point[2],
            left_ypr[0],
            left_ypr[1],
            left_ypr[2],
        ]
        right_target = [
            right_ee_point[0],
            right_ee_point[1],
            right_ee_point[2],
            right_ypr[0],
            right_ypr[1],
            right_ypr[2],
        ]
        arm_controller.reach_time = 0.0
        ok = arm_controller.execute_arm_event(left_target, right_target)
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return (
            f"[{self.config_label}] MoveBoxBothClawsToPoints start: "
            f"left_point_key={self.left_point_key}, right_point_key={self.right_point_key}"
        )
