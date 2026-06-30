"""通用躯干绝对位姿控制节点。

数据来源优先级：
1. JSON 参数 `pose`，适合固定测试位姿或明确写死的业务位姿。
2. blackboard 参数 `pose_key`，适合前置计算节点动态写入目标位姿。
3. 默认使用 `services.torso_controller.initial_pose`，避免缺参时下发危险零位姿。
"""

import ast

import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class MoveTorsoPose(TimedMockAction):
    """读取 6 维躯干目标位姿并发布到底层控制器。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        raw_pose = params.get("pose", None)
        self.pose = self._parse_pose(raw_pose) if raw_pose not in (None, "") else None
        self.pose_key = str(params.get("pose_key", "")).strip()
        self.wait_done = self._to_bool(params.get("wait_done", True))
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.pose_key:
            self.blackboard.register_key(key=self.pose_key, access=py_trees.common.Access.READ)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    @staticmethod
    def _parse_pose(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError("MoveTorsoPose pose 必须是长度为 6 的列表: [x, y, z, roll, pitch, yaw]")
        return [float(item) for item in value]

    def _resolve_pose(self, services):
        """按 JSON > blackboard > 默认值 的优先级解析目标位姿。"""
        if self.pose is not None:
            return list(self.pose), "json:pose"
        if self.pose_key:
            if not self.blackboard.exists(self.pose_key):
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] blackboard 缺少腰部目标位姿: key={self.pose_key}"
                )
                return None, f"blackboard:{self.pose_key}"
            return self._parse_pose(self.blackboard.get(self.pose_key)), f"blackboard:{self.pose_key}"
        return self._parse_pose(services.torso_controller.initial_pose), "default:torso_controller.initial_pose"

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE
        if not hasattr(services, "torso_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 中没有 torso_controller: key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
            return Status.SUCCESS

        pose, pose_source = self._resolve_pose(services)
        if pose is None:
            return Status.FAILURE
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 发布躯干目标: "
            f"source={pose_source}, "
            f"x={pose[0]:.3f}, y={pose[1]:.3f}, z={pose[2]:.3f}, "
            f"roll={pose[3]:.3f}, pitch={pose[4]:.3f}, yaw={pose[5]:.3f}, "
            f"wait_done={self.wait_done}"
        )
        ok = services.torso_controller.move_to_pose(list(pose), wait_done=self.wait_done)
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        pose_desc = self.pose if self.pose is not None else f"blackboard:{self.pose_key or '<initial_pose>'}"
        return f"[{self.config_label}] MoveTorsoPose start: pose={pose_desc}"
