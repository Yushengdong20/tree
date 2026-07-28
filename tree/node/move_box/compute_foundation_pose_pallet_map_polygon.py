"""将 FoundationPose 检出的托盘中心/朝向转换为 map 四角禁入区域。"""

import math

import numpy as np
import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.geometry import transform_base_point_to_global

from ..base import TimedMockAction


class ComputeFoundationPosePalletMapPolygon(TimedMockAction):
    """根据 FP 托盘检测输出，生成与托盘真实朝向一致的 map 四边形。

    FP 当前只提供中心和朝向，托盘长宽由配置提供。节点读取共享 FP detector
    最近一次的 base_link 下中心、左右轴和前向轴，结合当前 odom 的平面位姿生成：

    - ``pallet_map_polygon_key``：四个 ``[{x, y}, ...]`` map 顶点；
    - ``pallet_pose_key``：中心、边轴与尺寸，供后续导航/可视化诊断使用。

    该 polygon 是实体托盘区域，后续拆垛选箱会把它作为底盘禁入区，而不是用作
    箱体过滤区。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.pallet_size_x_m = float(params.get("pallet_size_x_m", 1.30))
        self.pallet_size_y_m = float(params.get("pallet_size_y_m", 0.90))
        self.pallet_map_polygon_key = str(
            params.get("pallet_map_polygon_key", "move_box_detected_pallet_map_polygon")
        ).strip()
        self.pallet_pose_key = str(
            params.get("pallet_pose_key", "move_box_detected_pallet_map_pose")
        ).strip()
        # 托盘粗靠近阶段的 YOLO 原始 OBB 也保留到黑板，方便对比 YOLO 尺寸与
        # 实测/FP 区域。它不参与 FP polygon 的几何计算。
        self.yolo_pallet_box_key = str(params.get("yolo_pallet_box_key", "")).strip()
        self.yolo_pallet_size_key = str(
            params.get("yolo_pallet_size_key", "move_box_detected_pallet_yolo_size")
        ).strip()
        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/fp_pallet_markers")
        ).strip()
        if self.pallet_size_x_m <= 0.0 or self.pallet_size_y_m <= 0.0:
            raise ValueError("pallet_size_x_m/pallet_size_y_m 必须大于 0")
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key=self.pallet_map_polygon_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(key=self.pallet_pose_key, access=py_trees.common.Access.WRITE)
        if self.yolo_pallet_box_key:
            self.blackboard.register_key(
                key=self.yolo_pallet_box_key,
                access=py_trees.common.Access.READ,
            )
        if self.yolo_pallet_size_key:
            self.blackboard.register_key(
                key=self.yolo_pallet_size_key,
                access=py_trees.common.Access.WRITE,
            )
        self.odom_transformer = self.get_odom_pose_transformer(self.odom_topic)
        self.publisher = None
        if self.visualization_enabled and self.visualization_topic:
            self.publisher = self.ros_node.create_publisher(
                self.visualization_topic, MarkerArray, queue_size=1, latch=True
            )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        detector = getattr(services, "box_detector", None) if services is not None else None
        if detector is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 缺少 FP detector: services_key={self.services_key}"
            )
            return Status.FAILURE
        center_base = detector.get_latest_box_center()
        axes = detector.get_latest_box_axes()
        front_axis = detector.get_latest_box_front_axis()
        pose = self.odom_transformer.get_current_pose()
        if center_base is None or axes is None or front_axis is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 FP 托盘中心与方向轴"
            )
            return Status.RUNNING
        if pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 odom 以生成托盘 map 区域: {self.odom_topic}"
            )
            return Status.RUNNING
        try:
            center_base = np.asarray(center_base, dtype=float)
            side_base = self._normalize_xy(axes["left"], "left_axis")
            front_base = self._normalize_xy(front_axis, "front_axis")
            # 处理数值误差；投影后的两轴须保持正交，以免四边形被拉斜。
            front_base = front_base - side_base * float(np.dot(front_base, side_base))
            front_base = self._normalize_xy(front_base, "orthogonal_front_axis")
        except (KeyError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] FP 托盘方向轴无效: {exc}"
            )
            return Status.FAILURE

        center_map_xy = transform_base_point_to_global(
            self._pose2d(pose), float(center_base[0]), float(center_base[1])
        )
        side_map = self._rotate_base_axis_to_map(side_base, float(pose[3]))
        front_map = self._rotate_base_axis_to_map(front_base, float(pose[3]))
        half_x = self.pallet_size_x_m * 0.5
        half_y = self.pallet_size_y_m * 0.5
        corners = [
            np.asarray([center_map_xy["x"], center_map_xy["y"]]) + sx * side_map * half_x + sy * front_map * half_y
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        polygon = [{"x": float(point[0]), "y": float(point[1])} for point in corners]
        pose_data = {
            "x": float(center_map_xy["x"]),
            "y": float(center_map_xy["y"]),
            # 当前 base_link 在 map 的 z 也应计入，避免日志中的“map 托盘高度”
            # 实际仍是 base_link 高度而造成后续使用者误解。
            "z": float(center_base[2]) + float(pose[2]),
            "side_axis": [float(side_map[0]), float(side_map[1])],
            "front_axis": [float(front_map[0]), float(front_map[1])],
            "size": [self.pallet_size_x_m, self.pallet_size_y_m],
            "yaw": math.degrees(math.atan2(float(front_map[1]), float(front_map[0]))),
        }
        self.blackboard.set(self.pallet_map_polygon_key, polygon, overwrite=True)
        self.blackboard.set(self.pallet_pose_key, pose_data, overwrite=True)
        yolo_size = self._store_yolo_pallet_size()
        self._publish(polygon, pose_data)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] FP 托盘 map 区域已生成: "
            f"center=({pose_data['x']:.3f},{pose_data['y']:.3f},{pose_data['z']:.3f}), "
            f"size=({self.pallet_size_x_m:.3f},{self.pallet_size_y_m:.3f}), "
            f"yaw={pose_data['yaw']:.1f}deg, "
            f"yolo_size_xyz={yolo_size if yolo_size is not None else '<unavailable>'}, "
            f"polygon={polygon}"
        )
        return Status.SUCCESS

    def _store_yolo_pallet_size(self):
        """传递粗靠近阶段选中的 YOLO 托盘 size，供日志和后续策略读取。"""
        if not self.yolo_pallet_box_key or not self.blackboard.exists(self.yolo_pallet_box_key):
            return None
        try:
            raw_box = self.blackboard.get(self.yolo_pallet_box_key)
            size = [abs(float(value)) for value in raw_box.get("size", [])]
            if len(size) != 3:
                raise ValueError(f"size={raw_box.get('size')!r}")
        except (AttributeError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] YOLO 托盘尺寸不可用: "
                f"key={self.yolo_pallet_box_key}, error={exc}"
            )
            return None
        if self.yolo_pallet_size_key:
            self.blackboard.set(self.yolo_pallet_size_key, size, overwrite=True)
        return size

    @staticmethod
    def _normalize_xy(vector, name):
        value = np.asarray(vector, dtype=float)[:2]
        norm = float(np.linalg.norm(value))
        if norm < 1e-6 or not np.isfinite(norm):
            raise ValueError(f"{name} 的 map 平面投影非法: {vector}")
        return value / norm

    @staticmethod
    def _rotate_base_axis_to_map(axis, yaw_deg):
        yaw = math.radians(yaw_deg)
        return np.asarray([
            math.cos(yaw) * axis[0] - math.sin(yaw) * axis[1],
            math.sin(yaw) * axis[0] + math.cos(yaw) * axis[1],
        ])

    @staticmethod
    def _pose2d(pose):
        class Pose2D:
            pass
        result = Pose2D()
        result.x, result.y, result.yaw = float(pose[0]), float(pose[1]), float(pose[3])
        return result

    def _publish(self, polygon, pose_data):
        if self.publisher is None:
            return
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)
        line = Marker()
        line.header.frame_id = MAP_FRAME
        line.header.stamp = self.ros_node.now()
        line.ns = "fp_detected_pallet_polygon"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.035
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.25, 0.05, 1.0
        line.points = [Point(x=p["x"], y=p["y"], z=0.05) for p in polygon + polygon[:1]]
        marker_array.markers.append(line)
        text = Marker()
        text.header.frame_id = MAP_FRAME
        text.header.stamp = self.ros_node.now()
        text.ns = "fp_detected_pallet_text"
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x, text.pose.position.y, text.pose.position.z = pose_data["x"], pose_data["y"], 0.20
        text.pose.orientation.w = 1.0
        text.scale.z = 0.10
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = (
            "FP PALLET REGION\\n"
            f"size=({self.pallet_size_x_m:.2f},{self.pallet_size_y_m:.2f}) "
            f"yaw={pose_data['yaw']:.1f}deg"
        )
        marker_array.markers.append(text)
        self.publisher.publish(marker_array)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeFoundationPosePalletMapPolygon start: "
            f"size=({self.pallet_size_x_m:.3f},{self.pallet_size_y_m:.3f}), "
            f"polygon_key={self.pallet_map_polygon_key}"
        )
