# -*- coding: utf-8 -*-
"""等待机器人接近导航目标的节点。"""

import math
import time

import py_trees
from nav_msgs.msg import Odometry
from py_trees.common import Status

from ..base import TimedMockAction


class WaitUntilNearNavigationGoal(TimedMockAction):
    """持续监听里程计，距离导航目标小于阈值时返回 SUCCESS。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.distance_threshold = float(params.get("distance_threshold", 0.6))
        self.target_key = str(params.get("target_key", "navigation_target")).strip()
        self.target_x = float(params.get("x", 0.0))
        self.target_y = float(params.get("y", 0.0))
        self.use_blackboard_target = self._to_bool(params.get("use_blackboard_target", bool(self.target_key)))
        self.log_interval_sec = float(params.get("log_interval_sec", 1.0))

        self._current_position = None
        self._latest_odom_time = None
        self._last_log_time = 0.0

        if self.target_key:
            self.blackboard.register_key(key=self.target_key, access=py_trees.common.Access.READ)

        # 关键步骤：通过通用消息订阅接口监听 odom，兼容 ROS1/ROS2 的消息类型订阅。
        self._odom_subscriber = self.ros_node.create_message_subscription(
            self.odom_topic,
            Odometry,
            self._on_odom,
            queue_size=10,
        )

    def _on_odom(self, message):
        """缓存最新 odom 位置，update 中只读取缓存做距离判断。"""
        position = message.pose.pose.position
        self._current_position = (
            float(position.x),
            float(position.y),
            float(position.z),
        )
        self._latest_odom_time = time.monotonic()

    def _read_target(self):
        if not self.use_blackboard_target:
            return self.target_x, self.target_y

        if not self.target_key or not self.blackboard.exists(self.target_key):
            return None

        raw_target = self.blackboard.get(self.target_key)
        return self._parse_target_xy(raw_target)

    def _parse_target_xy(self, raw_target):
        """兼容 dict/list/tuple 或带 x/y 属性的导航目标。"""
        if raw_target is None:
            return None

        if isinstance(raw_target, dict):
            if "x" in raw_target and "y" in raw_target:
                return float(raw_target["x"]), float(raw_target["y"])
            if "position" in raw_target:
                return self._parse_target_xy(raw_target["position"])
            if "pose" in raw_target:
                return self._parse_target_xy(raw_target["pose"])

        if isinstance(raw_target, (list, tuple)) and len(raw_target) >= 2:
            return float(raw_target[0]), float(raw_target[1])

        if hasattr(raw_target, "x") and hasattr(raw_target, "y"):
            return float(raw_target.x), float(raw_target.y)

        if hasattr(raw_target, "position"):
            return self._parse_target_xy(raw_target.position)

        if hasattr(raw_target, "pose"):
            return self._parse_target_xy(raw_target.pose)

        return None

    def _log_throttled(self, message):
        now = time.monotonic()
        if now - self._last_log_time < self.log_interval_sec:
            return
        self._last_log_time = now
        self.ros_node.get_logger().info(message)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if self._current_position is None:
            self._log_throttled(f"[{self.config_label}] 等待 odom 数据: topic={self.odom_topic}")
            return Status.RUNNING

        target = self._read_target()
        if target is None:
            self._log_throttled(
                f"[{self.config_label}] 等待导航目标: target_key={self.target_key or '<params>'}"
            )
            return Status.RUNNING

        target_x, target_y = target
        current_x, current_y, _ = self._current_position
        # 关键步骤：导航接近判断只使用 odom 平面距离，避免地形或腰部高度影响判断。
        distance = math.hypot(current_x - target_x, current_y - target_y)
        if distance < self.distance_threshold:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已接近导航目标: "
                f"distance={distance:.3f} < threshold={self.distance_threshold:.3f}, "
                f"current=({current_x:.3f}, {current_y:.3f}), "
                f"target=({target_x:.3f}, {target_y:.3f})"
            )
            return Status.SUCCESS

        self._log_throttled(
            f"[{self.config_label}] 接近导航目标中: "
            f"distance={distance:.3f}, threshold={self.distance_threshold:.3f}, "
            f"current=({current_x:.3f}, {current_y:.3f}), "
            f"target=({target_x:.3f}, {target_y:.3f})"
        )
        return Status.RUNNING

    def describe_start(self):
        return (
            f"[{self.config_label}] WaitUntilNearNavigationGoal start: "
            f"odom_topic={self.odom_topic}, "
            f"target_key={self.target_key or '<params>'}, "
            f"distance_threshold={self.distance_threshold:.3f}"
        )
