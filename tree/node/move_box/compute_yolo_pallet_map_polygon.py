"""由 YOLO class 5 的 map OBB 记录托盘禁入四边形。"""

import math

import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME

from ..base import TimedMockAction


class ComputeYoloPalletMapPolygon(TimedMockAction):
    """将 YOLO 托盘 OBB 的 map 角点投影成底盘禁入 polygon。

    输入为 ``MoveBoxYoloApproachToBox.selected_box_output_key`` 输出的完整 map
    box。节点原样保存其 ``size=[x,y,z]``，这样可对比 YOLO 估计尺寸与 FP/实测
    托盘尺寸；polygon 优先使用已按时间对齐转换到 map 的 OBB corners。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.input_box_key = str(
            params.get("input_box_key", "move_box_yolo_detected_pallet")
        ).strip()
        self.expected_class_id = int(params.get("expected_class_id", 5))
        self.pallet_map_polygon_key = str(
            params.get("pallet_map_polygon_key", "move_box_detected_pallet_map_polygon")
        ).strip()
        self.pallet_pose_key = str(
            params.get("pallet_pose_key", "move_box_detected_pallet_map_pose")
        ).strip()
        self.pallet_size_key = str(
            params.get("pallet_size_key", "move_box_detected_pallet_yolo_size")
        ).strip()
        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/yolo_pallet_markers")
        ).strip()
        self.blackboard.register_key(key=self.input_box_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.pallet_map_polygon_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.pallet_pose_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.pallet_size_key, access=py_trees.common.Access.WRITE)
        self.publisher = None
        if self.visualization_enabled and self.visualization_topic:
            self.publisher = self.ros_node.create_publisher(
                self.visualization_topic, MarkerArray, queue_size=1, latch=True
            )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        if not self.blackboard.exists(self.input_box_key):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 YOLO 托盘输出: key={self.input_box_key}"
            )
            return Status.RUNNING
        raw_box = self.blackboard.get(self.input_box_key)
        try:
            class_id = int(raw_box.get("class_id"))
            if class_id != self.expected_class_id:
                raise ValueError(
                    f"class_id={class_id}，期望托盘类别={self.expected_class_id}"
                )
            map_position = raw_box.get("map_position") or {}
            center = [
                float(map_position["x"]),
                float(map_position["y"]),
                float(map_position.get("z", 0.0)),
            ]
            size = [abs(float(value)) for value in raw_box.get("size", [])]
            if len(size) != 3 or min(size) <= 0.0:
                raise ValueError(f"YOLO 托盘 size 非法: {raw_box.get('size')!r}")
            corners = (raw_box.get("geometry") or {}).get("corners") or []
            polygon = self._build_polygon(corners)
            if len(polygon) < 3:
                raise ValueError("YOLO 托盘 map OBB corners 不足，无法生成禁入区域")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 计算 YOLO 托盘 map 区域失败: {exc}; "
                f"input={raw_box!r}"
            )
            return Status.FAILURE

        pose = {
            "x": center[0], "y": center[1], "z": center[2],
            "size": list(size), "class_id": class_id, "source": "yolo_obb",
        }
        self.blackboard.set(self.pallet_map_polygon_key, polygon, overwrite=True)
        self.blackboard.set(self.pallet_pose_key, pose, overwrite=True)
        self.blackboard.set(self.pallet_size_key, list(size), overwrite=True)
        self._publish(polygon, pose)
        self.ros_node.get_logger().info(
            f"\033[1;96m[{self.config_label}] YOLO 托盘 map 区域已生成: "
            f"class_id={class_id}, center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}), "
            f"size_xyz=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f}), "
            f"polygon={polygon}\033[0m"
        )
        return Status.SUCCESS

    @staticmethod
    def _build_polygon(corners):
        """求 map XY 凸包；水平 OBB 的 8 个角投影后正好得到四个托盘边角。"""
        points = sorted({(round(float(p[0]), 8), round(float(p[1]), 8)) for p in corners})
        if len(points) < 3:
            return []

        def cross(origin, first, second):
            return ((first[0] - origin[0]) * (second[1] - origin[1])
                    - (first[1] - origin[1]) * (second[0] - origin[0]))

        lower = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        hull = lower[:-1] + upper[:-1]
        return [{"x": float(x), "y": float(y)} for x, y in hull]

    def _publish(self, polygon, pose):
        if self.publisher is None:
            return
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)
        line = Marker()
        line.header.frame_id = MAP_FRAME
        line.header.stamp = self.ros_node.now()
        line.ns, line.id = "yolo_detected_pallet_polygon", 1
        line.type, line.action = Marker.LINE_STRIP, Marker.ADD
        line.scale.x = 0.035
        line.color.r, line.color.g, line.color.b, line.color.a = 0.15, 0.7, 1.0, 1.0
        line.points = [Point(x=p["x"], y=p["y"], z=0.05) for p in polygon + polygon[:1]]
        marker_array.markers.append(line)
        text = Marker()
        text.header.frame_id = MAP_FRAME
        text.header.stamp = self.ros_node.now()
        text.ns, text.id = "yolo_detected_pallet_text", 2
        text.type, text.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y, text.pose.position.z = pose["x"], pose["y"], 0.20
        text.pose.orientation.w = 1.0
        text.scale.z = 0.10
        text.color.r = 0.15
        text.color.g = 0.85
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = "YOLO PALLET REGION\\nsize=(%.2f, %.2f, %.2f)" % tuple(pose["size"])
        marker_array.markers.append(text)
        self.publisher.publish(marker_array)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeYoloPalletMapPolygon start: "
            f"input={self.input_box_key}, polygon_key={self.pallet_map_polygon_key}"
        )
