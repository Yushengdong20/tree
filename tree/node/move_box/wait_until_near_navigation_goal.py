# -*- coding: utf-8 -*-
"""等待机器人接近导航目标的节点。"""

import math
import time

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.utils.geometry import get_odom_pose_transformer


class WaitUntilNearNavigationGoal(TimedMockAction):
    """持续监听里程计，距离和可选角度满足阈值时返回 SUCCESS。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.distance_threshold = float(params.get("distance_threshold", 0.6))
        self.target_key = str(params.get("target_key", "navigation_target")).strip()
        self.target_x = float(params.get("x", 0.0))
        self.target_y = float(params.get("y", 0.0))
        self.target_yaw = self._optional_float(params.get("yaw", None))
        self.use_blackboard_target = self._to_bool(params.get("use_blackboard_target", bool(self.target_key)))
        self.use_yaw_threshold = self._to_bool(params.get("use_yaw_threshold", False))
        self.yaw_threshold_deg = abs(float(params.get("yaw_threshold_deg", 15.0)))
        self.log_interval_sec = float(params.get("log_interval_sec", 1.0))

        self._last_log_time = 0.0

        if self.target_key:
            self.blackboard.register_key(key=self.target_key, access=py_trees.common.Access.READ)

        # 关键步骤：复用工具类里的 odom 订阅，避免多个 node 各自维护 odom 缓存。
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
        )

    def _read_target(self):
        if not self.use_blackboard_target:
            return self.target_x, self.target_y, self.target_yaw

        if not self.target_key or not self.blackboard.exists(self.target_key):
            return None

        raw_target = self.blackboard.get(self.target_key)
        return self._parse_target_pose(raw_target)

    def _parse_target_pose(self, raw_target):
        """兼容 dict/list/tuple 或带 x/y/yaw 属性的导航目标。"""
        if raw_target is None:
            return None

        if isinstance(raw_target, dict):
            if "x" in raw_target and "y" in raw_target:
                return (
                    float(raw_target["x"]),
                    float(raw_target["y"]),
                    self._optional_float(raw_target.get("yaw", raw_target.get("angle"))),
                )
            if "position" in raw_target:
                return self._parse_target_pose(raw_target["position"])
            if "pose" in raw_target:
                return self._parse_target_pose(raw_target["pose"])

        if isinstance(raw_target, (list, tuple)) and len(raw_target) >= 2:
            target_yaw = self._optional_float(raw_target[2]) if len(raw_target) >= 3 else None
            return float(raw_target[0]), float(raw_target[1]), target_yaw

        if hasattr(raw_target, "x") and hasattr(raw_target, "y"):
            return (
                float(raw_target.x),
                float(raw_target.y),
                self._optional_float(getattr(raw_target, "yaw", None)),
            )

        if hasattr(raw_target, "position"):
            return self._parse_target_pose(raw_target.position)

        if hasattr(raw_target, "pose"):
            return self._parse_target_pose(raw_target.pose)

        return None

    @staticmethod
    def _optional_float(value):
        """把可选配置转换为 float，空值保持 None。"""
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _normalize_angle_deg(angle_deg):
        """归一化角度差到 [-180, 180)，避免跨正负 180 度时误判。"""
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    def _log_throttled(self, message):
        now = time.monotonic()
        if now - self._last_log_time < self.log_interval_sec:
            return
        self._last_log_time = now
        self.ros_node.get_logger().info(message)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        current_pose = self.odom_transformer.get_current_pose()
        if current_pose is None:
            self._log_throttled(f"[{self.config_label}] 等待 odom 数据: topic={self.odom_topic}")
            return Status.RUNNING

        target = self._read_target()
        if target is None:
            self._log_throttled(
                f"[{self.config_label}] 等待导航目标: target_key={self.target_key or '<params>'}"
            )
            return Status.RUNNING

        target_x, target_y, target_yaw = target
        current_x, current_y, _, current_yaw = current_pose
        # 关键步骤：导航接近判断只使用 odom 平面距离，避免地形或腰部高度影响判断。
        distance = math.hypot(current_x - target_x, current_y - target_y)
        yaw_error = None
        if self.use_yaw_threshold:
            if target_yaw is None:
                self._log_throttled(
                    f"[{self.config_label}] 等待带 yaw 的导航目标: "
                    f"target_key={self.target_key or '<params>'}"
                )
                return Status.RUNNING

            # 关键步骤：YOLO 切 FP 时需要等朝向接近，避免 FP 启动时箱子不在视野内。
            yaw_error = abs(self._normalize_angle_deg(current_yaw - target_yaw))

        distance_reached = distance < self.distance_threshold
        yaw_reached = (not self.use_yaw_threshold) or yaw_error <= self.yaw_threshold_deg
        if distance_reached and yaw_reached:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已接近导航目标: "
                f"距离={distance:.3f} < 距离阈值={self.distance_threshold:.3f}, "
                f"当前位置=({current_x:.3f}, {current_y:.3f}), "
                f"目标位置=({target_x:.3f}, {target_y:.3f}), "
                f"检查角度={self.use_yaw_threshold}, "
                f"角度误差={self._format_optional_float(yaw_error)}, "
                f"角度阈值={self.yaw_threshold_deg:.3f}"
            )
            return Status.SUCCESS

        self._log_throttled(
            f"[{self.config_label}] 接近导航目标中: "
            f"距离={distance:.3f}, 距离阈值={self.distance_threshold:.3f}, "
            f"当前位置=({current_x:.3f}, {current_y:.3f}), "
            f"目标位置=({target_x:.3f}, {target_y:.3f}), "
            f"检查角度={self.use_yaw_threshold}, "
            f"角度误差={self._format_optional_float(yaw_error)}, "
            f"角度阈值={self.yaw_threshold_deg:.3f}"
        )
        return Status.RUNNING

    @staticmethod
    def _format_optional_float(value):
        """格式化可选浮点数，便于日志输出。"""
        if value is None:
            return "None"
        return f"{value:.3f}"

    def describe_start(self):
        return (
            f"[{self.config_label}] WaitUntilNearNavigationGoal start: "
            f"odom_topic={self.odom_topic}, "
            f"target_key={self.target_key or '<params>'}, "
            f"距离阈值={self.distance_threshold:.3f}, "
            f"检查角度={self.use_yaw_threshold}, "
            f"角度阈值={self.yaw_threshold_deg:.3f}"
        )
