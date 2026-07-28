"""根据最新 FP 放置误差，生成直接放箱的高位 XY/yaw 小步目标。

本节点只服务 ``direct_place`` 调试方案：箱体仍由双爪夹持时，读取
``RefreshFpAndReportPalletPlaceError`` 输出的 map 系箱心与 yaw 误差，生成：

* 两爪整体平移，消除箱心 XY 误差；
* 两爪绕当前 FP 箱心同步转动，消除箱体 yaw 误差；
* 不修改 z、roll、pitch，所有校正均在高位完成。

计算结果为 base_link 下的左右 EEF 完整位姿，可直接由 ``ArmsToPose`` 执行。
当 XY/yaw 已落入阈值，本节点只置 ``aligned_key=True``，不生成新的动作。
"""

import math

import numpy as np
import py_trees

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction
from .fp_grasp_visualization import _map_from_base_matrix_via_melon_odom, _point_base_to_map


class ComputeMoveBoxFpDirectPlaceCorrectionTargets(TimedMockAction):
    """把 FP 的 map 系 XY/yaw 误差转为一轮受限双爪校正动作。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.error_result_key = str(
            params.get("error_result_key", "move_box_direct_place_fp_error")
        ).strip()
        self.left_pose_key = str(
            params.get("left_pose_key", "move_box_direct_place_fp_left_eef_target")
        ).strip()
        self.right_pose_key = str(
            params.get("right_pose_key", "move_box_direct_place_fp_right_eef_target")
        ).strip()
        self.aligned_key = str(
            params.get("aligned_key", "move_box_direct_place_fp_aligned")
        ).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.xy_tolerance_m = max(float(params.get("xy_tolerance_m", 0.03)), 0.0)
        self.yaw_tolerance_deg = max(float(params.get("yaw_tolerance_deg", 5.0)), 0.0)
        # 目前的周转箱无可区分的前后面：+X/right 与 -X/left 互换 180° 后，
        # 箱体长短边仍完全对齐。因此直接放箱阶段应按“轴”而不是“有向箭头”
        # 比较 yaw，避免 FP/mesh 在等价的 180° 姿态间翻转时，手臂错误地带箱子转半圈。
        # 日后若箱体有必须朝向同一侧的标签/缺口，可在 JSON 设 False。
        self.yaw_axis_symmetric = self._to_bool(params.get("yaw_axis_symmetric", True))
        self.max_step_xy_m = max(float(params.get("max_step_xy_m", 0.02)), 0.001)
        self.max_step_yaw_deg = max(float(params.get("max_step_yaw_deg", 3.0)), 0.1)

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.error_result_key, access=py_trees.common.Access.READ)
        for key in (self.left_pose_key, self.right_pose_key, self.aligned_key):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic, target_frame=MAP_FRAME, base_frame=BASE_LINK_FRAME
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        arm_controller = getattr(services, "arm_controller", None)
        if arm_controller is None:
            return self._fail(f"缺少 arm_controller: services_key={self.services_key}")
        result = self.blackboard.get(self.error_result_key) if self.blackboard.exists(self.error_result_key) else None
        parsed = self._parse_error(result)
        if parsed is None:
            return self._fail(f"FP 误差结果无效: key={self.error_result_key}, value={result!r}")
        actual_center_map, delta_xy, raw_delta_yaw_deg = parsed
        delta_yaw_deg = (
            self._normalize_axis_angle_deg(raw_delta_yaw_deg)
            if self.yaw_axis_symmetric
            else raw_delta_yaw_deg
        )
        planar_error = float(np.linalg.norm(delta_xy))
        aligned = planar_error <= self.xy_tolerance_m and abs(delta_yaw_deg) <= self.yaw_tolerance_deg
        self.blackboard.set(self.aligned_key, bool(aligned), overwrite=True)
        if aligned:
            self.ros_node.get_logger().info(
                f"\033[1;97;42m[{self.config_label}] 高位 FP 对齐已收敛: "
                f"xy={planar_error:.3f}/{self.xy_tolerance_m:.3f}m, "
                f"yaw_effective={delta_yaw_deg:+.2f}/{self.yaw_tolerance_deg:.2f}deg"
                f" (raw={raw_delta_yaw_deg:+.2f}deg, axis_symmetric={self.yaw_axis_symmetric})\033[0m"
            )
            return py_trees.common.Status.SUCCESS

        odom_msg = self.odom_transformer.get_latest_odom() if self.odom_transformer else None
        if odom_msg is None:
            return self._fail("等待 odom，无法计算 map -> base_link 校正")
        map_from_base = _map_from_base_matrix_via_melon_odom(odom_msg)
        base_from_map = np.linalg.inv(map_from_base)
        step_map_xy = self._limit_vector(-delta_xy, self.max_step_xy_m)
        yaw_step_deg = self._clamp(-delta_yaw_deg, -self.max_step_yaw_deg, self.max_step_yaw_deg)
        left_target = self._build_side_target(
            arm_controller, "left", actual_center_map, step_map_xy, yaw_step_deg, map_from_base, base_from_map
        )
        right_target = self._build_side_target(
            arm_controller, "right", actual_center_map, step_map_xy, yaw_step_deg, map_from_base, base_from_map
        )
        if left_target is None or right_target is None:
            return self._fail("无法读取当前双爪/末端 TF 或反算校正目标")
        self.blackboard.set(self.left_pose_key, left_target, overwrite=True)
        self.blackboard.set(self.right_pose_key, right_target, overwrite=True)
        self.ros_node.get_logger().info(
            f"\033[1;97;45m[{self.config_label}] 直接放箱高位 FP 校正: "
            f"planar={planar_error:.3f}m, yaw_effective={delta_yaw_deg:+.2f}deg"
            f" (raw={raw_delta_yaw_deg:+.2f}deg, axis_symmetric={self.yaw_axis_symmetric}); "
            f"step_map=({step_map_xy[0]:+.3f},{step_map_xy[1]:+.3f},+0.000)m, "
            f"yaw_step={yaw_step_deg:+.2f}deg\033[0m"
        )
        return py_trees.common.Status.SUCCESS

    def _build_side_target(self, arm_controller, side, center_map, step_map_xy, yaw_step_deg, map_from_base, base_from_map):
        current_eef = arm_controller.get_current_end_effector_pose(side)
        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if current_eef is None or len(current_eef) != 6 or transform is None:
            return None
        translation, _ = transform
        current_ypr = np.array(current_eef[3:6], dtype=float)
        current_claw_base = np.array(current_eef[:3], dtype=float) + ypr_to_rotation_matrix(current_ypr).dot(
            np.array(translation, dtype=float)
        )
        current_claw_map = _point_base_to_map(current_claw_base, map_from_base)
        yaw_rad = math.radians(yaw_step_deg)
        rotate_z = np.array(
            [[math.cos(yaw_rad), -math.sin(yaw_rad), 0.0], [math.sin(yaw_rad), math.cos(yaw_rad), 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        target_claw_map = np.array(center_map, dtype=float) + rotate_z.dot(
            current_claw_map - np.array(center_map, dtype=float)
        )
        target_claw_map[:2] += step_map_xy
        target_claw_base = self._point_map_to_base(target_claw_map, base_from_map)
        target_ypr = current_ypr.copy()
        target_ypr[0] = self._normalize_angle_deg(target_ypr[0] + yaw_step_deg)
        target_eef = arm_controller.claw_point_to_end_effector_point(target_claw_base, side, target_ypr)
        if target_eef is None:
            return None
        return [float(target_eef[0]), float(target_eef[1]), float(target_eef[2]), *[float(v) for v in target_ypr]]

    @staticmethod
    def _parse_error(result):
        if not isinstance(result, dict):
            return None
        try:
            actual = result["actual"]
            delta = result["delta"]
            orientation = result["orientation"]["delta_rpy_deg"]
            center = np.array([float(actual["x"]), float(actual["y"]), float(actual["z"])], dtype=float)
            delta_xy = np.array([float(delta["x"]), float(delta["y"])], dtype=float)
            yaw = float(orientation["yaw"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(center).all() or not np.isfinite(delta_xy).all() or not math.isfinite(yaw):
            return None
        return center, delta_xy, yaw

    @staticmethod
    def _point_map_to_base(point_map, base_from_map):
        homogeneous = np.ones(4, dtype=float)
        homogeneous[:3] = np.array(point_map, dtype=float)
        return base_from_map.dot(homogeneous)[:3]

    @staticmethod
    def _limit_vector(vector, limit):
        norm = float(np.linalg.norm(vector))
        return vector * (float(limit) / norm) if norm > limit else vector

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

    @staticmethod
    def _normalize_angle_deg(value):
        return (float(value) + 180.0) % 360.0 - 180.0

    @classmethod
    def _normalize_axis_angle_deg(cls, value):
        """把无方向箱体轴的夹角归一化到 [-90, 90)。"""
        return (float(value) + 90.0) % 180.0 - 90.0

    def _fail(self, message):
        self.feedback_message = message
        self.blackboard.set(self.aligned_key, False, overwrite=True)
        self.ros_node.get_logger().error(f"[{self.config_label}] {message}")
        return py_trees.common.Status.FAILURE
