"""放箱后刷新 FoundationPose，并报告实际箱心与码垛目标的 map 偏差。

用途：
- 码垛放箱动作完成后，等待一帧新的 FP 检测；
- 将 FP 箱心从 base_link 转到 map；
- 与本轮规划的最终放置箱心 ``move_box_pallet_place_final_box_pose`` 比较；
- 打印醒目的偏差日志，并发布 RViz 诊断 marker。

这个节点只做诊断，不修改码垛槽位计数，也不改变后续动作目标。
"""

import math
import time

import numpy as np
import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.pallet_place_diagnostics import write_pallet_place_diagnostic
from ..base import TimedMockAction
from .fp_grasp_visualization import (
    _axis_base_to_map,
    _box_edge_indices,
    _fp_box_corners,
    _map_from_base_matrix_via_melon_odom,
    _normalize,
    _point_base_to_map,
)


class RefreshFpAndReportPalletPlaceError(TimedMockAction):
    """刷新一次 FP，并在 map 下可视化/打印放置误差。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.expected_box_pose_key = str(
            params.get("expected_box_pose_key", "move_box_pallet_place_final_box_pose")
        ).strip()
        self.result_key = str(
            params.get("result_key", "move_box_pallet_place_error_result")
        ).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.wait_timeout_sec = float(params.get("wait_timeout_sec", 3.0))
        self.poll_interval_sec = float(params.get("poll_interval_sec", 0.2))
        self.clear_detection_cache_before_wait = self._to_bool(
            params.get("clear_detection_cache_before_wait", True)
        )
        self.fail_on_timeout = self._to_bool(params.get("fail_on_timeout", False))
        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/pallet_place_error_markers")
        ).strip()
        self.text_height_m = float(params.get("text_height_m", 0.35))
        self.expected_box_size_x = float(params.get("expected_box_size_x", 0.60))
        self.expected_box_size_y = float(params.get("expected_box_size_y", 0.40))
        self.expected_box_size_z = float(params.get("expected_box_size_z", 0.34))
        # 无前后特征的矩形箱可绕自身 up 轴翻转 180° 而保持同一几何占位。
        # 用于避免规划轴与 FP 等价反向轴产生虚假的 ±180° yaw 误差。
        self.yaw_axis_symmetric = self._to_bool(params.get("yaw_axis_symmetric", True))

        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )

        self.visualization_pub = None
        if self.visualization_enabled and self.visualization_topic:
            self.visualization_pub = self.ros_node.create_publisher(
                self.visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.expected_box_pose_key, access=py_trees.common.Access.READ)
        if self.result_key:
            self.blackboard.register_key(key=self.result_key, access=py_trees.common.Access.WRITE)

        self.services = None
        self._deadline = 0.0
        self._next_poll_at = 0.0
        self._cache_cleared = False

    def initialise(self):
        super().initialise()
        self.services = None
        self._deadline = time.monotonic() + self.wait_timeout_sec
        self._next_poll_at = 0.0
        self._cache_cleared = False

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        self.services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if self.services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] robot services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE

        expected_pose = self._read_expected_pose()
        if expected_pose is None:
            return Status.FAILURE

        if self.clear_detection_cache_before_wait and not self._cache_cleared:
            clear_cache = getattr(self.services.box_detector, "clear_latest_detection_cache", None)
            if callable(clear_cache):
                clear_cache()
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 已清空旧 FP 检测缓存，等待放箱后的新 FP 帧"
                )
            self._cache_cleared = True

        now = time.monotonic()
        if now > self._deadline:
            message = (
                f"[{self.config_label}] 放箱后等待 FP 新检测超时: "
                f"timeout={self.wait_timeout_sec:.1f}s, topic={self.visualization_topic or '<disabled>'}"
            )
            if self.fail_on_timeout:
                self.ros_node.get_logger().error(message)
                return Status.FAILURE
            self.ros_node.get_logger().warning(message + "，继续流程")
            return Status.SUCCESS

        if now < self._next_poll_at:
            return Status.RUNNING

        self._next_poll_at = now + self.poll_interval_sec
        if not self._refresh_fp_detection():
            return Status.RUNNING

        actual = self._build_actual_map_result()
        if actual is None:
            return Status.RUNNING

        result = self._build_error_result(expected_pose, actual)
        if self.result_key:
            self.blackboard.set(self.result_key, result, overwrite=True)
        self._log_result(result)
        write_pallet_place_diagnostic(
            "post_place_fp_error",
            {
                "label": self.config_label,
                "expected_box_pose_map": expected_pose,
                "actual_box": actual,
                "error": result,
            },
        )
        self._publish_visualization(expected_pose, actual, result)
        return Status.SUCCESS

    def _refresh_fp_detection(self):
        detector = getattr(self.services, "box_detector", None)
        arm_controller = getattr(self.services, "arm_controller", None)
        if detector is None or arm_controller is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] robot services 缺少 box_detector/arm_controller")
            return False

        return bool(
            detector.update_latest_grasp_pose(
                arm_controller.get_initial_left_ypr(),
                arm_controller.get_initial_right_ypr(),
            )
        )

    def _build_actual_map_result(self):
        detector = self.services.box_detector
        box_center = self._as_vector(detector.get_latest_box_center())
        box_axes = detector.get_latest_box_axes()
        # ``right`` 是 FP 新约定 right_x_front_y_up_z 的局部 +X；旧抓取
        # 业务只保存 left 时，以其反向兼容，绝不能把 left 当作新 +X。
        right_axis = self._as_vector(box_axes.get("right") if box_axes else None)
        if right_axis is None:
            left_axis = self._as_vector(box_axes.get("left") if box_axes else None)
            right_axis = -left_axis if left_axis is not None else None
        up_axis = self._as_vector(box_axes.get("up") if box_axes else None)
        front_axis = self._as_vector(getattr(detector, "get_latest_box_front_axis", lambda: None)())
        if box_center is None:
            return None

        odom_msg = self.odom_transformer.get_latest_odom() if self.odom_transformer is not None else None
        if odom_msg is None:
            self.ros_node.get_logger().warning(f"[{self.config_label}] 等待 odom，无法把 FP 箱心转换到 map")
            return None

        map_from_base = _map_from_base_matrix_via_melon_odom(odom_msg)
        center_map = _point_base_to_map(box_center, map_from_base)
        right_axis_map = _axis_base_to_map(box_center, right_axis, map_from_base) if right_axis is not None else None
        up_axis_map = _axis_base_to_map(box_center, up_axis, map_from_base) if up_axis is not None else None
        front_axis_map = _axis_base_to_map(box_center, front_axis, map_from_base) if front_axis is not None else None

        box_size = self._box_size(detector)
        return {
            "center_base": box_center,
            "center_map": center_map,
            "right_axis_map": right_axis_map,
            "front_axis_map": front_axis_map,
            "up_axis_map": up_axis_map,
            "box_size": box_size,
        }

    def _build_error_result(self, expected_pose, actual):
        expected_center = np.array(
            [
                float(expected_pose["x"]),
                float(expected_pose["y"]),
                float(expected_pose.get("z", 0.0)),
            ],
            dtype=float,
        )
        actual_center = np.array(actual["center_map"], dtype=float)
        delta = actual_center - expected_center
        planar = math.hypot(float(delta[0]), float(delta[1]))
        distance = float(np.linalg.norm(delta))
        expected_rpy = {
            "roll": float(expected_pose.get("roll", 0.0)),
            "pitch": float(expected_pose.get("pitch", 0.0)),
            "yaw": float(expected_pose.get("yaw", 0.0)),
        }
        actual_rpy = self._axes_to_rpy_deg(actual)
        comparison_expected_rpy = dict(expected_rpy)
        target_axis_flipped = False
        if actual_rpy is not None and self.yaw_axis_symmetric:
            expected_axes, target_axis_flipped = self._canonical_expected_box_axes(
                expected_pose, actual
            )
            canonical_rpy = self._axes_to_rpy_deg_from_axes(expected_axes)
            if canonical_rpy is not None:
                comparison_expected_rpy = canonical_rpy
        orientation_delta = None
        if actual_rpy is not None:
            orientation_delta = {
                axis: self._normalize_angle_deg(
                    actual_rpy[axis] - comparison_expected_rpy[axis]
                )
                for axis in ("roll", "pitch", "yaw")
            }
        return {
            "expected": {
                "x": float(expected_center[0]),
                "y": float(expected_center[1]),
                "z": float(expected_center[2]),
                **expected_rpy,
            },
            "actual": {
                "x": float(actual_center[0]),
                "y": float(actual_center[1]),
                "z": float(actual_center[2]),
                "rpy": actual_rpy,
            },
            "delta": {
                "x": float(delta[0]),
                "y": float(delta[1]),
                "z": float(delta[2]),
                "planar": float(planar),
                "distance": float(distance),
            },
            "orientation": {
                "expected_rpy_deg": expected_rpy,
                "comparison_expected_rpy_deg": comparison_expected_rpy,
                "actual_rpy_deg": actual_rpy,
                "delta_rpy_deg": orientation_delta,
                "target_axis_flipped_180": target_axis_flipped,
                "yaw_axis_symmetric": self.yaw_axis_symmetric,
            },
        }

    @staticmethod
    def _normalize_angle_deg(angle_deg):
        """把姿态差归一化到 [-180, 180)，避免跨 ±180° 产生假大误差。"""
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _axes_to_rpy_deg(actual):
        """从 FP 的 left/front/up 轴求 map 下箱体 R/P/Y（单位：度）。

        轴定义与新 FP ``right_x_front_y_up_z`` 一致：right 是箱体局部 +X，
        front 是局部 +Y，up 是局部 +Z。输入轴会先正交化，避免 FP 轻微噪声
        造成旋转矩阵不合法；front 的正负由 FP 原始 front 轴保持一致。
        """
        axes = RefreshFpAndReportPalletPlaceError._actual_box_axes(actual)
        if axes is None:
            return None
        return RefreshFpAndReportPalletPlaceError._axes_to_rpy_deg_from_axes(axes)

    @staticmethod
    def _axes_to_rpy_deg_from_axes(axes):
        """从已正交化的 right/front/up 轴求 RPY。"""
        if axes is None:
            return None
        right_axis, front_axis, up_axis = axes

        rotation = np.identity(4, dtype=float)
        rotation[:3, 0] = right_axis
        rotation[:3, 1] = front_axis
        rotation[:3, 2] = up_axis
        try:
            roll_rad, pitch_rad, yaw_rad = tf_trans.euler_from_matrix(rotation)
        except (TypeError, ValueError):
            return None
        return {
            "roll": float(math.degrees(roll_rad)),
            "pitch": float(math.degrees(pitch_rad)),
            "yaw": float(math.degrees(yaw_rad)),
        }

    def _log_result(self, result):
        expected = result["expected"]
        actual = result["actual"]
        delta = result["delta"]
        orientation = result["orientation"]
        message = (
            f"[{self.config_label}] 放箱后 FP 实测偏差(map): "
            f"target=({expected['x']:.3f}, {expected['y']:.3f}, {expected['z']:.3f}), "
            f"actual=({actual['x']:.3f}, {actual['y']:.3f}, {actual['z']:.3f}), "
            f"delta=(dx={delta['x']:.3f}, dy={delta['y']:.3f}, dz={delta['z']:.3f}), "
            f"planar={delta['planar']:.3f}m, distance={delta['distance']:.3f}m"
        )
        if orientation["actual_rpy_deg"] is not None:
            expected_rpy = orientation["expected_rpy_deg"]
            comparison_rpy = orientation["comparison_expected_rpy_deg"]
            actual_rpy = orientation["actual_rpy_deg"]
            delta_rpy = orientation["delta_rpy_deg"]
            message += (
                f"; target_rpy=({expected_rpy['roll']:.1f},"
                f"{expected_rpy['pitch']:.1f},{expected_rpy['yaw']:.1f})deg, "
                f"actual_rpy=({actual_rpy['roll']:.1f},"
                f"{actual_rpy['pitch']:.1f},{actual_rpy['yaw']:.1f})deg, "
                f"delta_rpy=({delta_rpy['roll']:+.1f},"
                f"{delta_rpy['pitch']:+.1f},{delta_rpy['yaw']:+.1f})deg"
            )
            if orientation["target_axis_flipped_180"]:
                message += (
                    f"; target_axis=flipped_180_for_symmetric_box, "
                    f"compare_target_yaw={comparison_rpy['yaw']:.1f}deg"
                )
        else:
            message += "; RPY=unavailable(FP axes invalid)"
        self.ros_node.get_logger().info(f"\033[1;97;44m{message}\033[0m")

    def _publish_visualization(self, expected_pose, actual, result):
        if self.visualization_pub is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        expected = np.array([result["expected"]["x"], result["expected"]["y"], result["expected"]["z"]], dtype=float)
        actual_center = np.array(actual["center_map"], dtype=float)
        expected_axes, _ = self._canonical_expected_box_axes(expected_pose, actual)
        marker_id = self._append_expected_box(marker_array, marker_id, expected_pose, expected_axes)
        marker_id = self._append_sphere(marker_array, marker_id, "expected_box_center", expected, (0.1, 1.0, 0.1, 1.0), 0.08)
        marker_id = self._append_sphere(marker_array, marker_id, "actual_fp_box_center", actual_center, (1.0, 0.1, 0.1, 1.0), 0.08)
        marker_id = self._append_line(marker_array, marker_id, "place_error_vector", expected, actual_center, (1.0, 0.8, 0.0, 1.0), 0.025)
        marker_id = self._append_actual_box(marker_array, marker_id, actual)
        marker_id = self._append_expected_axes(marker_array, marker_id, expected_pose, expected_axes)
        marker_id = self._append_actual_axes(marker_array, marker_id, actual)
        self._append_text(marker_array, marker_id, expected, actual_center, result)
        self.visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布放箱后 FP 偏差 RViz 标记: topic={self.visualization_topic}"
        )

    def _append_actual_box(self, marker_array, marker_id, actual):
        center = np.array(actual["center_map"], dtype=float)
        axes = self._actual_box_axes(actual)
        box_size = actual.get("box_size")
        if axes is None or box_size is None:
            return marker_id
        right_axis, front_axis, up_axis = axes

        box_marker = self._new_marker(marker_id, "actual_fp_box_outline", Marker.LINE_LIST)
        marker_id += 1
        box_marker.scale.x = 0.025
        self._set_color(box_marker, 0.0, 0.9, 1.0, 0.95)
        corners = _fp_box_corners(center, right_axis, front_axis, up_axis, box_size)
        for start_index, end_index in _box_edge_indices():
            box_marker.points.append(self._point(corners[start_index]))
            box_marker.points.append(self._point(corners[end_index]))
        marker_array.markers.append(box_marker)
        return marker_id

    def _append_expected_box(self, marker_array, marker_id, expected_pose, axes=None):
        """绘制绿色规划箱体，避免误差话题只剩两个箱心而难以判断真实占位。"""
        right_axis, front_axis, up_axis = axes or self._expected_box_axes(expected_pose)
        center = np.array(
            [expected_pose["x"], expected_pose["y"], expected_pose.get("z", 0.0)], dtype=float
        )
        marker = self._new_marker(marker_id, "expected_box_outline", Marker.LINE_LIST)
        marker.scale.x = 0.025
        self._set_color(marker, 0.15, 1.0, 0.25, 0.95)
        corners = _fp_box_corners(
            center,
            right_axis,
            front_axis,
            up_axis,
            (self.expected_box_size_x, self.expected_box_size_y, self.expected_box_size_z),
        )
        for start_index, end_index in _box_edge_indices():
            marker.points.append(self._point(corners[start_index]))
            marker.points.append(self._point(corners[end_index]))
        marker_array.markers.append(marker)
        return marker_id + 1

    @staticmethod
    def _expected_box_axes(expected_pose):
        """以新 FP 的 right(+X)/front(+Y)/up(+Z) 定义生成目标坐标轴。"""
        rotation = tf_trans.euler_matrix(
            math.radians(float(expected_pose.get("roll", 0.0))),
            math.radians(float(expected_pose.get("pitch", 0.0))),
            math.radians(float(expected_pose.get("yaw", 0.0))),
        )[:3, :3]
        return rotation[:, 0], rotation[:, 1], rotation[:, 2]

    def _canonical_expected_box_axes(self, expected_pose, actual):
        """目标矩形箱轴与实测反向时，选择等价的 X/Y 同时翻转版本。"""
        expected_axes = self._expected_box_axes(expected_pose)
        if not self.yaw_axis_symmetric:
            return expected_axes, False
        actual_axes = self._actual_box_axes(actual)
        if actual_axes is None:
            return expected_axes, False
        if float(np.dot(expected_axes[0], actual_axes[0])) < 0.0:
            return (-expected_axes[0], -expected_axes[1], expected_axes[2]), True
        return expected_axes, False

    @staticmethod
    def _actual_box_axes(actual):
        """返回经正交化后的实测 right/front/up 轴，与 RPY 计算严格同源。"""
        right_axis = _normalize(actual.get("right_axis_map"))
        up_axis = _normalize(actual.get("up_axis_map"))
        raw_front_axis = _normalize(actual.get("front_axis_map"))
        if right_axis is None or up_axis is None or raw_front_axis is None:
            return None

        right_axis = _normalize(right_axis - np.dot(right_axis, up_axis) * up_axis)
        if right_axis is None:
            return None
        front_axis = _normalize(np.cross(up_axis, right_axis))
        if front_axis is None:
            return None
        if float(np.dot(front_axis, raw_front_axis)) < 0.0:
            front_axis = -front_axis
        return right_axis, front_axis, up_axis

    def _append_expected_axes(self, marker_array, marker_id, expected_pose, axes=None):
        center = np.array(
            [expected_pose["x"], expected_pose["y"], expected_pose.get("z", 0.0)],
            dtype=float,
        )
        return self._append_coordinate_axes(
            marker_array,
            marker_id,
            "target_axes_right_x_front_y_up_z",
            center,
            axes or self._expected_box_axes(expected_pose),
            alpha=0.60,
            width=0.016,
        )

    def _append_actual_axes(self, marker_array, marker_id, actual):
        axes = self._actual_box_axes(actual)
        if axes is None:
            return marker_id
        return self._append_coordinate_axes(
            marker_array,
            marker_id,
            "actual_fp_axes_right_x_front_y_up_z",
            np.array(actual["center_map"], dtype=float),
            axes,
            alpha=1.0,
            width=0.025,
        )

    def _append_coordinate_axes(self, marker_array, marker_id, namespace, center, axes, alpha, width):
        """绘制坐标轴：X/right=红、Y/front=绿、Z/up=蓝。"""
        axis_length = 0.26
        for label, axis, color in (
            ("right_x", axes[0], (1.0, 0.15, 0.15)),
            ("front_y", axes[1], (0.15, 1.0, 0.15)),
            ("up_z", axes[2], (0.15, 0.35, 1.0)),
        ):
            marker = self._new_marker(marker_id, f"{namespace}/{label}", Marker.ARROW)
            marker_id += 1
            marker.scale.x = float(width)
            marker.scale.y = float(width * 2.2)
            marker.scale.z = float(width * 2.8)
            marker.points = [
                self._point(center),
                self._point(center + np.array(axis, dtype=float) * axis_length),
            ]
            self._set_color(marker, color[0], color[1], color[2], alpha)
            marker_array.markers.append(marker)
        return marker_id

    def _append_sphere(self, marker_array, marker_id, namespace, point, color, scale):
        marker = self._new_marker(marker_id, namespace, Marker.SPHERE)
        marker_id += 1
        marker.pose.position = self._point(point)
        marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id

    def _append_line(self, marker_array, marker_id, namespace, start, end, color, width):
        marker = self._new_marker(marker_id, namespace, Marker.LINE_LIST)
        marker_id += 1
        marker.scale.x = float(width)
        marker.points = [self._point(start), self._point(end)]
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id

    def _append_text(self, marker_array, marker_id, expected, actual, result):
        delta = result["delta"]
        midpoint = (np.array(expected, dtype=float) + np.array(actual, dtype=float)) * 0.5
        marker = self._new_marker(marker_id, "pallet_place_error_text", Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = float(midpoint[0])
        marker.pose.position.y = float(midpoint[1])
        marker.pose.position.z = float(max(expected[2], actual[2]) + self.text_height_m)
        marker.scale.z = 0.09
        self._set_color(marker, 1.0, 1.0, 1.0, 1.0)
        marker.text = (
            "PALLET PLACE ERROR\n"
            f"target=({result['expected']['x']:.2f},{result['expected']['y']:.2f},{result['expected']['z']:.2f})\n"
            f"actual=({result['actual']['x']:.2f},{result['actual']['y']:.2f},{result['actual']['z']:.2f})\n"
            f"delta=({delta['x']:.3f},{delta['y']:.3f},{delta['z']:.3f})\n"
            f"planar={delta['planar']:.3f}m dist={delta['distance']:.3f}m"
        )
        orientation = result.get("orientation", {})
        actual_rpy = orientation.get("actual_rpy_deg")
        delta_rpy = orientation.get("delta_rpy_deg")
        if actual_rpy is not None and delta_rpy is not None:
            marker.text += (
                f"\nactual_rpy=({actual_rpy['roll']:.1f},{actual_rpy['pitch']:.1f},"
                f"{actual_rpy['yaw']:.1f})deg"
                f"\ndelta_rpy=({delta_rpy['roll']:+.1f},{delta_rpy['pitch']:+.1f},"
                f"{delta_rpy['yaw']:+.1f})deg"
            )
            if orientation.get("target_axis_flipped_180"):
                marker.text += "\ntarget_axes=180deg axial-equivalent"
        marker_array.markers.append(marker)

    def _read_expected_pose(self):
        if not self.blackboard.exists(self.expected_box_pose_key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 缺少目标放置箱心: key={self.expected_box_pose_key}"
            )
            return None
        raw = self.blackboard.get(self.expected_box_pose_key)
        if not isinstance(raw, dict):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 目标放置箱心必须是 dict: key={self.expected_box_pose_key}, value={raw!r}"
            )
            return None
        try:
            return {
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "z": float(raw.get("z", 0.0)),
                "yaw": float(raw.get("yaw", 0.0)),
                "pitch": float(raw.get("pitch", raw.get("p", 0.0))),
                "roll": float(raw.get("roll", raw.get("r", 0.0))),
            }
        except (KeyError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 目标放置箱心缺少有效 x/y/z: key={self.expected_box_pose_key}, value={raw!r}, error={exc}"
            )
            return None

    @staticmethod
    def _as_vector(value):
        if value is None:
            return None
        try:
            vector = np.array(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if vector.ndim == 0 or vector.shape[0] < 3:
            return None
        return vector[:3]

    @staticmethod
    def _box_size(detector):
        getter = getattr(detector, "get_latest_box_size", None)
        if callable(getter):
            size = getter()
            if isinstance(size, dict):
                return (
                    float(size.get("x", 0.60)),
                    float(size.get("y", 0.40)),
                    float(size.get("z", 0.34)),
                )
        return (
            float(getattr(detector, "box_size_x", 0.60)),
            float(getattr(detector, "box_size_y", 0.40)),
            float(getattr(detector, "box_size_z", 0.34)),
        )

    def _new_marker(self, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns = namespace
        marker.id = int(marker_id)
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _point(point):
        return Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))

    @staticmethod
    def _set_color(marker, red, green, blue, alpha):
        marker.color.r = float(red)
        marker.color.g = float(green)
        marker.color.b = float(blue)
        marker.color.a = float(alpha)
