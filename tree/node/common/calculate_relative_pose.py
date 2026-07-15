# -*- coding: utf-8 -*-
"""计算相对当前底盘位姿的导航目标。"""

import math
import time

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.constants import CHASSIS_FRAME
from tree.utils.geometry import get_odom_pose_transformer


class CalculateRelativePose(TimedMockAction):
    """根据当前 odom 位姿计算相对移动后的绝对导航目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.odom_topic = str(params.get("odom_topic", CHASSIS_FRAME)).strip()
        self.output_key = str(params.get("output_key", "navigation_target")).strip()
        self.dx = float(params.get("dx", 0.0))
        self.dy = float(params.get("dy", 0.0))
        self.dyaw = float(params.get("dyaw", 0.0))
        self.dx_key = str(params.get("dx_key", "")).strip()
        self.dy_key = str(params.get("dy_key", "")).strip()
        self.dyaw_key = str(params.get("dyaw_key", "")).strip()
        self.enabled_key = str(params.get("enabled_key", "")).strip()
        self.relative_frame = str(params.get("relative_frame", "base_link")).strip()
        self.timeout_sec = float(params.get("timeout_sec", 5.0))
        self.log_interval_sec = float(params.get("log_interval_sec", 1.0))

        self._deadline = None
        self._last_log_time = 0.0

        if self.output_key:
            self.blackboard.register_key(key=self.output_key, access=py_trees.common.Access.WRITE)
        for key in (self.dx_key, self.dy_key, self.dyaw_key, self.enabled_key):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

        # 关键步骤：复用共享 odom 订阅器，避免每个行为树节点重复创建订阅。
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
        )

    def initialise(self):
        super().initialise()
        self._deadline = time.monotonic() + self.timeout_sec
        self._last_log_time = 0.0

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if not self.output_key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 params.output_key")
            return Status.FAILURE
        if self.enabled_key and (
            not self.blackboard.exists(self.enabled_key)
            or not self._to_bool(self.blackboard.get(self.enabled_key))
        ):
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 跳过相对导航目标计算: enabled_key={self.enabled_key}"
            )
            return Status.SUCCESS

        current_pose = self.odom_transformer.get_current_pose()
        if current_pose is None:
            if time.monotonic() > self._deadline:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 等待 odom 超时: topic={self.odom_topic}"
                )
                return Status.FAILURE
            self._log_throttled(f"[{self.config_label}] 等待 odom 数据: topic={self.odom_topic}")
            return Status.RUNNING

        try:
            self._refresh_relative_offsets_from_blackboard()
            target_pose = self._calculate_target_pose(current_pose)
        except Exception as exc:
            self.ros_node.get_logger().error(f"[{self.config_label}] 计算相对目标失败: {exc}")
            return Status.FAILURE

        self.blackboard.set(self.output_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算相对导航目标: "
            f"current=({current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[3]:.3f}), "
            f"relative=({self.dx:.3f}, {self.dy:.3f}, {self.dyaw:.3f}, frame={self.relative_frame}), "
            f"target=({target_pose['x']:.3f}, {target_pose['y']:.3f}, {target_pose['yaw']:.3f}), "
            f"output_key={self.output_key}"
        )
        return Status.SUCCESS

    @staticmethod
    def _to_bool(value):
        """兼容字符串形式的布尔配置。"""
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def _refresh_relative_offsets_from_blackboard(self):
        """从 blackboard 覆盖 dx/dy/dyaw，支持前置采样节点动态决定底盘移动量。"""
        if self.dx_key:
            self.dx = self._read_float_key(self.dx_key)
        if self.dy_key:
            self.dy = self._read_float_key(self.dy_key)
        if self.dyaw_key:
            self.dyaw = self._read_float_key(self.dyaw_key)

    def _read_float_key(self, key):
        if not self.blackboard.exists(key):
            raise RuntimeError(f"blackboard key 不存在: {key}")
        return float(self.blackboard.get(key))

    def _calculate_target_pose(self, current_pose):
        """根据 relative_frame 将相对位移转换为地图/odom 下的目标位姿。"""
        current_x, current_y, _current_z, current_yaw = current_pose
        if self.relative_frame in ("base_link", "base", "robot"):
            yaw_rad = math.radians(current_yaw)
            # 关键步骤：base_link 坐标系下的 dx/dy 需要先按当前 yaw 旋转到 map/odom 坐标系。
            target_x = current_x + math.cos(yaw_rad) * self.dx - math.sin(yaw_rad) * self.dy
            target_y = current_y + math.sin(yaw_rad) * self.dx + math.cos(yaw_rad) * self.dy
        elif self.relative_frame in ("map", "odom", "world"):
            target_x = current_x + self.dx
            target_y = current_y + self.dy
        else:
            raise ValueError(f"不支持的 relative_frame: {self.relative_frame}")

        return {
            "x": float(target_x),
            "y": float(target_y),
            "yaw": self._normalize_angle_deg(current_yaw + self.dyaw),
        }

    def _log_throttled(self, message):
        now = time.monotonic()
        if now - self._last_log_time < self.log_interval_sec:
            return
        self._last_log_time = now
        self.ros_node.get_logger().info(message)

    @staticmethod
    def _normalize_angle_deg(angle_deg):
        """把角度归一化到 [-180, 180)，便于底盘导航接口统一处理。"""
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    def describe_start(self):
        return (
            f"[{self.config_label}] CalculateRelativePose start: "
            f"odom_topic={self.odom_topic}, output_key={self.output_key}, "
            f"relative=({self.dx:.3f}, {self.dy:.3f}, {self.dyaw:.3f}), "
            f"relative_frame={self.relative_frame}"
        )
