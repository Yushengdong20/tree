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
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from ..base import TimedMockAction
from .fp_grasp_visualization import (
    _axis_base_to_map,
    _box_edge_indices,
    _fp_box_corners,
    _map_from_odom_message,
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
        left_axis = self._as_vector(box_axes.get("left") if box_axes else None)
        up_axis = self._as_vector(box_axes.get("up") if box_axes else None)
        front_axis = self._as_vector(getattr(detector, "get_latest_box_front_axis", lambda: None)())
        if box_center is None:
            return None

        odom_msg = self.odom_transformer.get_latest_odom() if self.odom_transformer is not None else None
        if odom_msg is None:
            self.ros_node.get_logger().warning(f"[{self.config_label}] 等待 odom，无法把 FP 箱心转换到 map")
            return None

        map_from_base = _map_from_odom_message(odom_msg)
        center_map = _point_base_to_map(box_center, map_from_base)
        left_axis_map = _axis_base_to_map(box_center, left_axis, map_from_base) if left_axis is not None else None
        up_axis_map = _axis_base_to_map(box_center, up_axis, map_from_base) if up_axis is not None else None
        front_axis_map = _axis_base_to_map(box_center, front_axis, map_from_base) if front_axis is not None else None

        box_size = self._box_size(detector)
        return {
            "center_base": box_center,
            "center_map": center_map,
            "left_axis_map": left_axis_map,
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
        return {
            "expected": {
                "x": float(expected_center[0]),
                "y": float(expected_center[1]),
                "z": float(expected_center[2]),
                "yaw": float(expected_pose.get("yaw", 0.0)),
            },
            "actual": {
                "x": float(actual_center[0]),
                "y": float(actual_center[1]),
                "z": float(actual_center[2]),
            },
            "delta": {
                "x": float(delta[0]),
                "y": float(delta[1]),
                "z": float(delta[2]),
                "planar": float(planar),
                "distance": float(distance),
            },
        }

    def _log_result(self, result):
        expected = result["expected"]
        actual = result["actual"]
        delta = result["delta"]
        message = (
            f"[{self.config_label}] 放箱后 FP 实测偏差(map): "
            f"target=({expected['x']:.3f}, {expected['y']:.3f}, {expected['z']:.3f}), "
            f"actual=({actual['x']:.3f}, {actual['y']:.3f}, {actual['z']:.3f}), "
            f"delta=(dx={delta['x']:.3f}, dy={delta['y']:.3f}, dz={delta['z']:.3f}), "
            f"planar={delta['planar']:.3f}m, distance={delta['distance']:.3f}m"
        )
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
        marker_id = self._append_sphere(marker_array, marker_id, "expected_box_center", expected, (0.1, 1.0, 0.1, 1.0), 0.08)
        marker_id = self._append_sphere(marker_array, marker_id, "actual_fp_box_center", actual_center, (1.0, 0.1, 0.1, 1.0), 0.08)
        marker_id = self._append_line(marker_array, marker_id, "place_error_vector", expected, actual_center, (1.0, 0.8, 0.0, 1.0), 0.025)
        marker_id = self._append_actual_box(marker_array, marker_id, actual)
        self._append_text(marker_array, marker_id, expected, actual_center, result)
        self.visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布放箱后 FP 偏差 RViz 标记: topic={self.visualization_topic}"
        )

    def _append_actual_box(self, marker_array, marker_id, actual):
        center = np.array(actual["center_map"], dtype=float)
        left_axis = _normalize(actual.get("left_axis_map"))
        front_axis = _normalize(actual.get("front_axis_map"))
        up_axis = _normalize(actual.get("up_axis_map"))
        box_size = actual.get("box_size")
        if left_axis is None or front_axis is None or up_axis is None or box_size is None:
            return marker_id

        box_marker = self._new_marker(marker_id, "actual_fp_box_outline", Marker.LINE_LIST)
        marker_id += 1
        box_marker.scale.x = 0.025
        self._set_color(box_marker, 0.0, 0.9, 1.0, 0.95)
        corners = _fp_box_corners(center, left_axis, front_axis, up_axis, box_size)
        for start_index, end_index in _box_edge_indices():
            box_marker.points.append(self._point(corners[start_index]))
            box_marker.points.append(self._point(corners[end_index]))
        marker_array.markers.append(box_marker)
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
