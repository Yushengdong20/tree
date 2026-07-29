"""将 FoundationPose 检出的托盘中心/朝向转换为 map 四角禁入区域及 3D 可视化。"""

import math

import numpy as np
import py_trees
import tf.transformations as tf_trans
from py_trees.common import Status
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.geometry import map_from_source_matrix_via_melon_odom, transform_base_point_to_global

from ..base import TimedMockAction


class ComputeFoundationPosePalletMapPolygon(TimedMockAction):
    """根据 FP 托盘检测输出，生成与托盘真实朝向一致的 map 四边形。

    默认读取共享 FP detector 最近一次的 base_link 下中心、左右轴和前向轴，结合
    当前 odom 的平面位姿生成。启用 ``use_raw_box_obb=True`` 时，则直接读取共享
    detector 已锁定的 ``/foundationpose/box`` 原始 OBB，并采用完整
    ``map <- base_link <- camera_link`` 矩阵变换；此模式可与原始 OBB 诊断可视化
    严格保持同一中心、朝向和时间对齐链路。

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
        # FP 返回的是托盘模型中心与朝向；实体厚度由现场实测配置，供 map 下
        # 3D 线框/半透明实体显示使用，不影响底盘禁入 polygon 的 XY 计算。
        self.pallet_height_m = float(params.get("pallet_height_m", 0.15))
        # True 时不再使用 /foundationpose/pose 的平面近似链路，而是使用已被
        # FPBoxDetector 消费的 /foundationpose/box 原始帧。默认 False 保持历史树行为。
        self.use_raw_box_obb = self._to_bool(params.get("use_raw_box_obb", False))
        self.raw_box_tf_timeout_sec = max(float(params.get("raw_box_tf_timeout_sec", 0.5)), 0.05)
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
        if min(self.pallet_size_x_m, self.pallet_size_y_m, self.pallet_height_m) <= 0.0:
            raise ValueError("pallet_size_x_m/pallet_size_y_m/pallet_height_m 必须大于 0")
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
        if self.use_raw_box_obb:
            raw_map_pose = self._build_raw_obb_map_pose(services, detector)
            if raw_map_pose is None:
                return Status.RUNNING
            center_map = raw_map_pose["center"]
            pallet_size = raw_map_pose["size"]
            try:
                side_map = self._normalize_xy(raw_map_pose["side_axis"], "raw_box_x_axis")
                front_map = self._normalize_xy(raw_map_pose["front_axis"], "raw_box_y_axis")
                front_map = front_map - side_map * float(np.dot(front_map, side_map))
                front_map = self._normalize_xy(front_map, "raw_box_orthogonal_y_axis")
            except (TypeError, ValueError) as exc:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 原始 FP 托盘 OBB 方向轴无效: {exc}"
                )
                return Status.FAILURE
            transform_source = "raw_box_obb_3d"
            raw_stamp = raw_map_pose["stamp_sec"]
            odom_delta_sec = raw_map_pose["odom_delta_sec"]
        else:
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
            center_map = [float(center_map_xy["x"]), float(center_map_xy["y"]), float(center_base[2]) + float(pose[2])]
            pallet_size = [self.pallet_size_x_m, self.pallet_size_y_m, self.pallet_height_m]
            side_map = self._rotate_base_axis_to_map(side_base, float(pose[3]))
            front_map = self._rotate_base_axis_to_map(front_base, float(pose[3]))
            transform_source = "detector_pose_2d"
            raw_stamp = None
            odom_delta_sec = None
        half_x = float(pallet_size[0]) * 0.5
        half_y = float(pallet_size[1]) * 0.5
        corners = [
            np.asarray([center_map[0], center_map[1]]) + sx * side_map * half_x + sy * front_map * half_y
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        polygon = [{"x": float(point[0]), "y": float(point[1])} for point in corners]
        pose_data = {
            "x": float(center_map[0]),
            "y": float(center_map[1]),
            "z": float(center_map[2]),
            "side_axis": [float(side_map[0]), float(side_map[1])],
            "front_axis": [float(front_map[0]), float(front_map[1])],
            "size": [float(value) for value in pallet_size],
            "yaw": math.degrees(math.atan2(float(front_map[1]), float(front_map[0]))),
            "transform_source": transform_source,
            "raw_box_stamp": raw_stamp,
            "odom_delta_sec": odom_delta_sec,
        }
        self.blackboard.set(self.pallet_map_polygon_key, polygon, overwrite=True)
        self.blackboard.set(self.pallet_pose_key, pose_data, overwrite=True)
        yolo_size = self._store_yolo_pallet_size()
        self._publish(polygon, pose_data)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] FP 托盘 map 区域已生成: "
            f"center=({pose_data['x']:.3f},{pose_data['y']:.3f},{pose_data['z']:.3f}), "
            f"size=({pose_data['size'][0]:.3f},{pose_data['size'][1]:.3f},{pose_data['size'][2]:.3f}), "
            f"yaw={pose_data['yaw']:.1f}deg, "
            f"transform={transform_source}, "
            f"raw_stamp={raw_stamp if raw_stamp is not None else '<n/a>'}, "
            f"odom_delta_ms={odom_delta_sec * 1000.0 if odom_delta_sec is not None else '<n/a>'}, "
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

    def _build_raw_obb_map_pose(self, services, detector):
        """将 detector 已消费的 /foundationpose/box 原始 OBB 直接转换到 map。

        这里不使用 detector 的 base_link 箱心再做 2D 平面近似，而是完整组合
        ``map <- base_link <- source_frame <- box``。raw frame 是
        ``update_latest_grasp_pose()`` 实际参与本次检测计算的一帧，因此与抓取
        缓存、原始 OBB 对比可视化使用同一 stamp。
        """
        getter = getattr(detector, "get_latest_raw_box_frame", None)
        raw_box = getter() if callable(getter) else None
        if raw_box is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待共享 detector 消费 /foundationpose/box 原始 OBB"
            )
            return None
        arm_controller = getattr(services, "arm_controller", None)
        tf_listener = getattr(arm_controller, "tf_listener", None)
        if tf_listener is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 缺少 tf_listener，无法转换原始 FP OBB"
            )
            return None
        stamp_sec = float(raw_box.get("stamp_sec", 0.0))
        odom_msg = self.odom_transformer.get_nearest_odom_by_stamp_sec(
            stamp_sec if stamp_sec > 0.0 else None
        )
        if odom_msg is None:
            odom_msg = self.odom_transformer.get_latest_odom()
        if odom_msg is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 odom 以转换原始 FP OBB: {self.odom_topic}"
            )
            return None
        try:
            map_from_source = map_from_source_matrix_via_melon_odom(
                tf_listener,
                self.ros_node,
                odom_msg,
                str(raw_box["frame_id"]),
                map_frame=MAP_FRAME,
                base_frame=BASE_LINK_FRAME,
                timeout=self.raw_box_tf_timeout_sec,
            )
            source_from_box = tf_trans.concatenate_matrices(
                tf_trans.translation_matrix(raw_box["center"]),
                tf_trans.quaternion_matrix(raw_box["quat"]),
            )
            map_from_box = tf_trans.concatenate_matrices(map_from_source, source_from_box)
            matched_stamp = self._stamp_to_sec(odom_msg.header.stamp)
            return {
                "center": [float(value) for value in map_from_box[:3, 3]],
                "side_axis": [float(value) for value in map_from_box[:3, 0]],
                "front_axis": [float(value) for value in map_from_box[:3, 1]],
                "size": [float(value) for value in raw_box["size"]],
                "stamp_sec": stamp_sec,
                "odom_delta_sec": abs(matched_stamp - stamp_sec) if stamp_sec > 0.0 else 0.0,
            }
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 原始 FP OBB 转 map 失败: {exc}"
            )
            return None

    @staticmethod
    def _stamp_to_sec(stamp):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9

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

        # pose_data 的 size 在 raw OBB 模式直接来自 /foundationpose/box；历史模式则
        # 来自 JSON。二者都按“模型中心 + 完整三维尺寸”绘制。
        pallet_size = [float(value) for value in pose_data["size"]]
        # 禁入 polygon 的 XY 几何不变，但不再把显示高度写死为 z=0.05。
        center_z = float(pose_data["z"])
        half_height = pallet_size[2] * 0.5
        bottom_z = center_z - half_height
        top_z = center_z + half_height
        bottom_corners = [Point(x=p["x"], y=p["y"], z=bottom_z) for p in polygon]
        top_corners = [Point(x=p["x"], y=p["y"], z=top_z) for p in polygon]

        solid = Marker()
        solid.header.frame_id = MAP_FRAME
        solid.header.stamp = self.ros_node.now()
        solid.ns, solid.id = "fp_detected_pallet_solid", 1
        solid.type, solid.action = Marker.CUBE, Marker.ADD
        solid.pose.position.x = float(pose_data["x"])
        solid.pose.position.y = float(pose_data["y"])
        solid.pose.position.z = center_z
        side_axis = pose_data["side_axis"]
        side_yaw = math.atan2(float(side_axis[1]), float(side_axis[0]))
        solid.pose.orientation.z = math.sin(side_yaw * 0.5)
        solid.pose.orientation.w = math.cos(side_yaw * 0.5)
        solid.scale.x, solid.scale.y, solid.scale.z = pallet_size
        solid.color.r, solid.color.g, solid.color.b, solid.color.a = 1.0, 0.32, 0.06, 0.16
        marker_array.markers.append(solid)

        wireframe = Marker()
        wireframe.header.frame_id = MAP_FRAME
        wireframe.header.stamp = self.ros_node.now()
        wireframe.ns, wireframe.id = "fp_detected_pallet_wireframe", 2
        wireframe.type, wireframe.action = Marker.LINE_LIST, Marker.ADD
        wireframe.scale.x = 0.035
        wireframe.color.r, wireframe.color.g, wireframe.color.b, wireframe.color.a = 1.0, 0.25, 0.05, 1.0
        for index in range(4):
            next_index = (index + 1) % 4
            wireframe.points.extend((bottom_corners[index], bottom_corners[next_index]))
            wireframe.points.extend((top_corners[index], top_corners[next_index]))
            wireframe.points.extend((bottom_corners[index], top_corners[index]))
        marker_array.markers.append(wireframe)

        center_marker = Marker()
        center_marker.header.frame_id = MAP_FRAME
        center_marker.header.stamp = self.ros_node.now()
        center_marker.ns, center_marker.id = "fp_detected_pallet_center", 3
        center_marker.type, center_marker.action = Marker.SPHERE, Marker.ADD
        center_marker.pose.position.x = float(pose_data["x"])
        center_marker.pose.position.y = float(pose_data["y"])
        center_marker.pose.position.z = center_z
        center_marker.pose.orientation.w = 1.0
        center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.08
        center_marker.color.r, center_marker.color.g, center_marker.color.b, center_marker.color.a = 1.0, 1.0, 0.1, 1.0
        marker_array.markers.append(center_marker)

        text = Marker()
        text.header.frame_id = MAP_FRAME
        text.header.stamp = self.ros_node.now()
        text.ns = "fp_detected_pallet_text"
        text.id = 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(pose_data["x"])
        text.pose.position.y = float(pose_data["y"])
        text.pose.position.z = top_z + 0.12
        text.pose.orientation.w = 1.0
        text.scale.z = 0.10
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = (
            "FP PALLET\\n"
            f"center=({pose_data['x']:.2f},{pose_data['y']:.2f},{center_z:.2f})\\n"
            f"size=({pallet_size[0]:.2f},{pallet_size[1]:.2f},{pallet_size[2]:.2f}) "
            f"yaw={pose_data['yaw']:.1f}deg"
        )
        marker_array.markers.append(text)
        self.publisher.publish(marker_array)

    def describe_start(self):
        size_description = "raw /foundationpose/box" if self.use_raw_box_obb else (
            f"({self.pallet_size_x_m:.3f},{self.pallet_size_y_m:.3f},{self.pallet_height_m:.3f})"
        )
        return (
            f"[{self.config_label}] ComputeFoundationPosePalletMapPolygon start: "
            f"size={size_description}, "
            f"polygon_key={self.pallet_map_polygon_key}"
        )
