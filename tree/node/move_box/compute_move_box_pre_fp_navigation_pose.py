"""根据当前位姿与抓取策略点关系，决定是否跳过预备导航。"""

import math

import py_trees
from py_trees.common import Status

from tree.utils.box_map_polygon import is_map_position_in_polygon, parse_map_polygon
from tree.utils.geometry import get_odom_pose_transformer

from ..base import TimedMockAction


class ComputeMoveBoxPreFpNavigationPose(TimedMockAction):
    """若当前位置已满足 FP 初始化条件，则直接使用当前位置；否则退到预备位。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.source_pose_key = str(
            params.get("source_pose_key", "move_box_selected_approach_pose")
        ).strip()
        self.selected_box_key = str(
            params.get("selected_box_key", "move_box_selected_highest_yolo_box")
        ).strip()
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_pre_fp_navigation_pose")
        ).strip()
        self.skip_navigation_key = str(
            params.get("skip_navigation_key", "move_box_skip_pre_fp_navigation")
        ).strip()
        self.offset_distance_m = float(params.get("offset_distance_m", 1.0))
        self.max_direct_distance_to_approach_m = float(
            params.get("max_direct_distance_to_approach_m", 0.8)
        )
        self.max_direct_yaw_error_deg = float(
            params.get("max_direct_yaw_error_deg", 45.0)
        )
        self.same_side_dot_min = float(params.get("same_side_dot_min", 0.0))
        self.pallet_map_polygon = parse_map_polygon(
            params.get("pallet_map_polygon", [])
        )
        self.pallet_clearance_m = float(params.get("pallet_clearance_m", 0.0))

        self.blackboard.register_key(
            key=self.source_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.selected_box_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.target_pose_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.skip_navigation_key,
            access=py_trees.common.Access.WRITE,
        )
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
        )

    def update(self):
        source_pose = self._load_source_pose()
        selected_box = self._load_selected_box()
        current_pose = self.odom_transformer.get_current_pose()
        if source_pose is None or selected_box is None:
            return Status.FAILURE
        if current_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 odom 数据: topic={self.odom_topic}"
            )
            return Status.RUNNING

        approach_x = float(source_pose["x"])
        approach_y = float(source_pose["y"])
        approach_yaw_deg = float(source_pose["yaw"])
        box_center = self._extract_box_center(selected_box)
        if box_center is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 无法从选中箱体提取 map 中心: "
                f"key={self.selected_box_key}, value={selected_box!r}"
            )
            return Status.FAILURE

        current_x, current_y, _current_z, current_yaw_deg = current_pose
        direct_distance = math.hypot(current_x - approach_x, current_y - approach_y)
        yaw_error_deg = abs(self._normalize_angle_deg(current_yaw_deg - approach_yaw_deg))
        same_side_score = self._same_side_score(
            box_center[0],
            box_center[1],
            approach_x,
            approach_y,
            current_x,
            current_y,
        )
        current_pose_safe = self._is_pose_safe(current_x, current_y)

        skip_navigation = (
            direct_distance <= self.max_direct_distance_to_approach_m
            and yaw_error_deg <= self.max_direct_yaw_error_deg
            and same_side_score >= self.same_side_dot_min
            and current_pose_safe
        )

        if skip_navigation:
            target_pose = {
                "x": float(current_x),
                "y": float(current_y),
                "yaw": float(current_yaw_deg),
                "source": "current_pose",
            }
            reason = (
                f"当前位置已满足FP初始化条件: direct_distance={direct_distance:.3f}m, "
                f"yaw_error={yaw_error_deg:.2f}deg, same_side_score={same_side_score:.3f}"
            )
        else:
            target_pose = self._build_offset_pose(source_pose)
            reason = (
                f"需前往预备位: direct_distance={direct_distance:.3f}m, "
                f"yaw_error={yaw_error_deg:.2f}deg, same_side_score={same_side_score:.3f}, "
                f"current_pose_safe={current_pose_safe}"
            )

        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.blackboard.set(self.skip_navigation_key, skip_navigation, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 预备导航决策: skip_navigation={skip_navigation}, "
            f"target=({target_pose['x']:.3f}, {target_pose['y']:.3f}, {target_pose['yaw']:.2f}deg), "
            f"{reason}"
        )
        return Status.SUCCESS

    def _load_source_pose(self):
        if not self.blackboard.exists(self.source_pose_key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 缺少抓取策略点: key={self.source_pose_key}"
            )
            return None
        source_pose = self.blackboard.get(self.source_pose_key)
        if not isinstance(source_pose, dict):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 抓取策略点必须是 dict: "
                f"key={self.source_pose_key}, value={source_pose!r}"
            )
            return None
        try:
            float(source_pose["x"])
            float(source_pose["y"])
            float(source_pose["yaw"])
        except (KeyError, TypeError, ValueError):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 抓取策略点缺少有效 x/y/yaw: "
                f"key={self.source_pose_key}, value={source_pose!r}"
            )
            return None
        return source_pose

    def _load_selected_box(self):
        if not self.blackboard.exists(self.selected_box_key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 缺少选中箱体: key={self.selected_box_key}"
            )
            return None
        selected_box = self.blackboard.get(self.selected_box_key)
        if not isinstance(selected_box, dict):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 选中箱体必须是 dict: "
                f"key={self.selected_box_key}, value={selected_box!r}"
            )
            return None
        return selected_box

    @staticmethod
    def _extract_box_center(selected_box):
        center = selected_box.get("map")
        if center is None:
            center = selected_box.get("map_position")
        if center is None:
            center = selected_box.get("center")

        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError):
                return None

        if isinstance(center, dict) and "x" in center and "y" in center:
            try:
                return float(center["x"]), float(center["y"])
            except (TypeError, ValueError):
                return None

        return None

    def _build_offset_pose(self, source_pose):
        source_x = float(source_pose["x"])
        source_y = float(source_pose["y"])
        source_yaw_deg = float(source_pose["yaw"])
        yaw_rad = math.radians(source_yaw_deg)
        target_x = source_x - self.offset_distance_m * math.cos(yaw_rad)
        target_y = source_y - self.offset_distance_m * math.sin(yaw_rad)
        target_pose = dict(source_pose)
        target_pose["x"] = float(target_x)
        target_pose["y"] = float(target_y)
        target_pose["yaw"] = float(source_yaw_deg)
        target_pose["source"] = "offset_pose"
        target_pose["offset_distance_m"] = float(self.offset_distance_m)
        return target_pose

    def _is_pose_safe(self, pose_x, pose_y):
        map_position = {"x": float(pose_x), "y": float(pose_y)}
        if self.pallet_map_polygon and is_map_position_in_polygon(
            map_position, self.pallet_map_polygon
        ):
            return False
        if self.pallet_map_polygon and self.pallet_clearance_m > 0.0:
            clearance = self._point_to_polygon_boundary_distance(
                (pose_x, pose_y),
                [(vertex["x"], vertex["y"]) for vertex in self.pallet_map_polygon],
            )
            if clearance < self.pallet_clearance_m:
                return False
        return True

    @staticmethod
    def _same_side_score(box_x, box_y, approach_x, approach_y, current_x, current_y):
        normal_x = approach_x - box_x
        normal_y = approach_y - box_y
        normal_norm = math.hypot(normal_x, normal_y)
        if normal_norm <= 1e-6:
            return -1.0
        normal_x /= normal_norm
        normal_y /= normal_norm
        current_vec_x = current_x - box_x
        current_vec_y = current_y - box_y
        return current_vec_x * normal_x + current_vec_y * normal_y

    @staticmethod
    def _point_to_polygon_boundary_distance(point, polygon):
        if len(polygon) < 2:
            return float("inf")
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        distances = [
            ComputeMoveBoxPreFpNavigationPose._point_segment_distance(point, start, end)
            for start, end in edges
        ]
        return min(distances) if distances else float("inf")

    @staticmethod
    def _point_segment_distance(point, start, end):
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        squared_length = delta_x * delta_x + delta_y * delta_y
        if squared_length <= 1e-12:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        ratio = (
            (point[0] - start[0]) * delta_x
            + (point[1] - start[1]) * delta_y
        ) / squared_length
        ratio = max(0.0, min(1.0, ratio))
        nearest_x = start[0] + ratio * delta_x
        nearest_y = start[1] + ratio * delta_y
        return math.hypot(point[0] - nearest_x, point[1] - nearest_y)

    @staticmethod
    def _normalize_angle_deg(angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPreFpNavigationPose start: "
            f"source_key={self.source_pose_key}, selected_box_key={self.selected_box_key}, "
            f"target_key={self.target_pose_key}, skip_key={self.skip_navigation_key}"
        )
