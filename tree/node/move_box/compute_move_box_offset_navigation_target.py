"""根据已有导航位姿沿其朝向前后偏移，生成新的导航目标。"""

import math

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxOffsetNavigationTarget(TimedMockAction):
    """从 blackboard 读取一个导航 pose，并沿其 yaw 方向做前后偏移。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.source_pose_key = str(
            params.get("source_pose_key", "move_box_selected_approach_pose")
        ).strip()
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_offset_navigation_target")
        ).strip()
        self.offset_distance_m = float(params.get("offset_distance_m", 0.0))
        self.offset_direction = str(params.get("offset_direction", "backward")).strip().lower()

        self.blackboard.register_key(
            key=self.source_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.target_pose_key,
            access=py_trees.common.Access.WRITE,
        )

    def update(self):
        source_pose = self._load_source_pose()
        if source_pose is None:
            return Status.FAILURE

        source_x = float(source_pose["x"])
        source_y = float(source_pose["y"])
        source_yaw_deg = float(source_pose["yaw"])
        yaw_rad = math.radians(source_yaw_deg)

        direction_sign = -1.0 if self.offset_direction == "backward" else 1.0
        target_x = source_x + direction_sign * self.offset_distance_m * math.cos(yaw_rad)
        target_y = source_y + direction_sign * self.offset_distance_m * math.sin(yaw_rad)

        target_pose = dict(source_pose)
        target_pose["x"] = float(target_x)
        target_pose["y"] = float(target_y)
        target_pose["yaw"] = float(source_yaw_deg)
        target_pose["source_pose_key"] = self.source_pose_key
        target_pose["offset_distance_m"] = float(self.offset_distance_m)
        target_pose["offset_direction"] = self.offset_direction

        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已生成偏移导航目标: "
            f"source=({source_x:.3f}, {source_y:.3f}, {source_yaw_deg:.2f}deg), "
            f"target=({target_x:.3f}, {target_y:.3f}, {source_yaw_deg:.2f}deg), "
            f"offset={self.offset_direction} {self.offset_distance_m:.3f}m, "
            f"source_key={self.source_pose_key}, target_key={self.target_pose_key}"
        )
        return Status.SUCCESS

    def _load_source_pose(self):
        if not self.blackboard.exists(self.source_pose_key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 缺少源导航位姿: key={self.source_pose_key}"
            )
            return None

        source_pose = self.blackboard.get(self.source_pose_key)
        if not isinstance(source_pose, dict):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 源导航位姿必须是 dict: "
                f"key={self.source_pose_key}, value={source_pose!r}"
            )
            return None

        try:
            float(source_pose["x"])
            float(source_pose["y"])
            float(source_pose["yaw"])
        except (KeyError, TypeError, ValueError):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 源导航位姿缺少有效 x/y/yaw: "
                f"key={self.source_pose_key}, value={source_pose!r}"
            )
            return None

        return source_pose

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxOffsetNavigationTarget start: "
            f"source_key={self.source_pose_key}, target_key={self.target_pose_key}, "
            f"offset_direction={self.offset_direction}, offset_distance={self.offset_distance_m:.3f}"
        )
