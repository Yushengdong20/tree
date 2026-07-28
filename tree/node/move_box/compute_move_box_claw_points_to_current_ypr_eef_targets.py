"""把 base_link 夹爪点目标反算为保持当前姿态的 EEF 完整目标。"""

import py_trees

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class ComputeMoveBoxClawPointsToCurrentYprEefTargets(TimedMockAction):
    """用于下降等只改位置的动作，保留前一轮 FP 已校正的 yaw/pitch/roll。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.left_point_key = str(params.get("left_point_key", "")).strip()
        self.right_point_key = str(params.get("right_point_key", "")).strip()
        self.left_pose_key = str(params.get("left_pose_key", "")).strip()
        self.right_pose_key = str(params.get("right_pose_key", "")).strip()
        for key, access in (
            (self.services_key, py_trees.common.Access.READ),
            (self.left_point_key, py_trees.common.Access.READ),
            (self.right_point_key, py_trees.common.Access.READ),
            (self.left_pose_key, py_trees.common.Access.WRITE),
            (self.right_pose_key, py_trees.common.Access.WRITE),
        ):
            if key:
                self.blackboard.register_key(key=key, access=access)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        arm_controller = getattr(services, "arm_controller", None)
        if arm_controller is None:
            return self._fail("缺少 arm_controller")
        left = self._target_for_side(arm_controller, "left", self.left_point_key)
        right = self._target_for_side(arm_controller, "right", self.right_point_key)
        if left is None or right is None:
            return self._fail("无法用当前 EEF 姿态反算双爪下降目标")
        self.blackboard.set(self.left_pose_key, left, overwrite=True)
        self.blackboard.set(self.right_pose_key, right, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已生成保持当前YPR的双臂目标: "
            f"left=({left[0]:.3f},{left[1]:.3f},{left[2]:.3f}, yaw={left[3]:.1f}), "
            f"right=({right[0]:.3f},{right[1]:.3f},{right[2]:.3f}, yaw={right[3]:.1f})"
        )
        return py_trees.common.Status.SUCCESS

    def _target_for_side(self, arm_controller, side, point_key):
        if not point_key or not self.blackboard.exists(point_key):
            return None
        point = self.blackboard.get(point_key)
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            return None
        current = arm_controller.get_current_end_effector_pose(side)
        if current is None or len(current) != 6:
            return None
        ypr = [float(value) for value in current[3:6]]
        eef = arm_controller.claw_point_to_end_effector_point(point[:3], side, ypr)
        if eef is None:
            return None
        return [float(eef[0]), float(eef[1]), float(eef[2]), *ypr]

    def _fail(self, message):
        self.feedback_message = message
        self.ros_node.get_logger().error(f"[{self.config_label}] {message}")
        return py_trees.common.Status.FAILURE
