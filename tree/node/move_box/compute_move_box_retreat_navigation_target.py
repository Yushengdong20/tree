"""根据当前 odom 位姿生成一个向后退的导航目标点。"""

import math

import py_trees
from py_trees.common import Status

from tree.utils.geometry import get_odom_pose_transformer

from ..base import TimedMockAction


class ComputeMoveBoxRetreatNavigationTarget(TimedMockAction):
    """把当前底盘位姿沿车体后方投影为一个新的导航目标。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.target_key = str(
            params.get("target_key", "move_box_retreat_navigation_target")
        ).strip()
        self.retreat_distance_m = float(params.get("retreat_distance_m", 1.0))
        self.keep_current_yaw = self._to_bool(params.get("keep_current_yaw", True))
        self.target_yaw_deg = float(params.get("target_yaw_deg", 0.0))

        self.blackboard.register_key(
            key=self.target_key,
            access=py_trees.common.Access.WRITE,
        )
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
        )

    def update(self):
        current_pose = self.odom_transformer.get_current_pose()
        if current_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 odom 数据: topic={self.odom_topic}"
            )
            return Status.RUNNING

        current_x, current_y, _current_z, current_yaw_deg = current_pose
        yaw_rad = math.radians(current_yaw_deg)
        target_x = current_x - self.retreat_distance_m * math.cos(yaw_rad)
        target_y = current_y - self.retreat_distance_m * math.sin(yaw_rad)
        target_yaw_deg = current_yaw_deg if self.keep_current_yaw else self.target_yaw_deg

        target = {
            "x": float(target_x),
            "y": float(target_y),
            "yaw": float(target_yaw_deg),
        }
        self.blackboard.set(self.target_key, target, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算后退导航目标: "
            f"current=({current_x:.3f}, {current_y:.3f}, {current_yaw_deg:.2f}deg), "
            f"target=({target_x:.3f}, {target_y:.3f}, {target_yaw_deg:.2f}deg), "
            f"distance={self.retreat_distance_m:.3f}m, key={self.target_key}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxRetreatNavigationTarget start: "
            f"odom_topic={self.odom_topic}, retreat_distance={self.retreat_distance_m:.3f}, "
            f"target_key={self.target_key}"
        )
