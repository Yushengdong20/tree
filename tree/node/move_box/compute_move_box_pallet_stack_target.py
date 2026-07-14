"""计算 move_box 码垛目标。

第一版码垛采用“固定垛盘 polygon + 网格槽位 + stack_count”的确定性策略：

1. 从 JSON 读取垛盘 map 多边形，自动取最长边作为 pallet-x，垂直方向作为 pallet-y。
2. 按 box_size / rows / cols / gap 在垛盘中心铺开网格。
3. 根据 blackboard 中的 stack_count 选择当前槽位和层数。
4. 反推出一个垛盘外导航位：机器人面向槽位，槽位位于机器人前方
   place_box_forward_offset_m 处。后续放置节点只需要下降 z。
5. 写出动态放置平面高度，供 ComputeMoveBoxTorsoToPlaceHeightPose /
   ComputeMoveBoxPlaceTargets 使用。

注意：本节点只计算当前目标，不自增 stack_count。放置成功后由
IncrementBlackboardValue 推进计数，避免导航/放置失败时跳过槽位。
"""

import math

import py_trees
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME
from tree.utils.params import parse_param_value

from ..base import TimedMockAction


class ComputeMoveBoxPalletStackTarget(TimedMockAction):
    """根据固定垛盘网格计算本轮码垛 slot、导航位和放置高度。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)

        self.pallet_map_polygon = self._parse_polygon(
            params.get(
                "pallet_map_polygon",
                [[-0.3, 1.0], [1.0, 1.0], [1.0, 0.1], [-0.3, 0.1]],
            )
        )
        self.rows = int(params.get("slot_rows", 2))
        self.cols = int(params.get("slot_cols", 2))
        self.max_layers = int(params.get("max_layers", 1))
        self.box_size_x = float(params.get("box_size_x", 0.60))
        self.box_size_y = float(params.get("box_size_y", 0.40))
        self.box_size_z = float(params.get("box_size_z", 0.34))
        self.slot_gap_x = float(params.get("slot_gap_x", 0.04))
        self.slot_gap_y = float(params.get("slot_gap_y", 0.04))
        self.pallet_surface_z = float(params.get("pallet_surface_z", 0.0))
        self.place_clearance_z = float(params.get("place_clearance_z", 0.0))
        self.place_box_forward_offset_m = float(params.get("place_box_forward_offset_m", 0.90))
        self.approach_side = str(params.get("approach_side", "positive_y")).strip().lower()
        self.stack_count_key = str(params.get("stack_count_key", "move_box_pallet_stack_count")).strip()
        self.navigation_target_key = str(
            params.get("navigation_target_key", "move_box_pallet_stack_navigation_target")
        ).strip()
        self.place_plane_height_key = str(
            params.get("place_plane_height_key", "move_box_pallet_stack_place_plane_height")
        ).strip()
        self.slot_pose_key = str(
            params.get("slot_pose_key", "move_box_pallet_stack_slot_pose")
        ).strip()
        self.expected_box_pose_key = str(
            params.get("expected_box_pose_key", "move_box_pallet_stack_expected_box_pose")
        ).strip()
        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/pallet_stack_markers")
        ).strip()

        self.blackboard.register_key(key=self.stack_count_key, access=py_trees.common.Access.READ)
        for key in (
            self.navigation_target_key,
            self.place_plane_height_key,
            self.slot_pose_key,
            self.expected_box_pose_key,
        ):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

        self.visualization_pub = None
        if self.visualization_enabled and self.visualization_topic:
            self.visualization_pub = self.ros_node.create_publisher(
                self.visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )

    def update(self):
        if self.rows <= 0 or self.cols <= 0:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] rows/cols 必须为正数: rows={self.rows}, cols={self.cols}"
            )
            return Status.FAILURE

        geometry = self._build_pallet_geometry()
        if geometry is None:
            return Status.FAILURE

        stack_count = self._get_stack_count()
        slots_per_layer = self.rows * self.cols
        layer = stack_count // slots_per_layer
        if self.max_layers > 0 and layer >= self.max_layers:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 码垛层数已超限: "
                f"stack_count={stack_count}, layer={layer}, max_layers={self.max_layers}"
            )
            return Status.FAILURE

        slot_index = stack_count % slots_per_layer
        row = slot_index // self.cols
        col = slot_index % self.cols
        slot_pose = self._compute_slot_pose(geometry, row, col, layer)
        navigation_pose = self._compute_navigation_pose(geometry, slot_pose)
        place_plane_height = (
            self.pallet_surface_z
            + layer * self.box_size_z
            + self.place_clearance_z
        )
        expected_box_pose = dict(slot_pose)
        expected_box_pose["z"] = place_plane_height + self.box_size_z * 0.5

        self.blackboard.set(self.slot_pose_key, slot_pose, overwrite=True)
        self.blackboard.set(self.expected_box_pose_key, expected_box_pose, overwrite=True)
        self.blackboard.set(self.place_plane_height_key, float(place_plane_height), overwrite=True)
        self.blackboard.set(self.navigation_target_key, navigation_pose, overwrite=True)

        self._publish_visualization(geometry, row, col, layer, slot_pose, navigation_pose, expected_box_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算码垛目标: "
            f"stack_count={stack_count}, layer={layer}, row={row}, col={col}, "
            f"slot=({slot_pose['x']:.3f}, {slot_pose['y']:.3f}, {slot_pose['yaw']:.2f}deg), "
            f"nav=({navigation_pose['x']:.3f}, {navigation_pose['y']:.3f}, {navigation_pose['yaw']:.2f}deg), "
            f"place_plane_z={place_plane_height:.3f}, "
            f"keys nav={self.navigation_target_key}, plane={self.place_plane_height_key}"
        )
        return Status.SUCCESS

    def _get_stack_count(self):
        if not self.blackboard.exists(self.stack_count_key):
            return 0
        try:
            return max(0, int(self.blackboard.get(self.stack_count_key)))
        except (TypeError, ValueError):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] stack_count 无效，按 0 处理: "
                f"key={self.stack_count_key}, value={self.blackboard.get(self.stack_count_key)!r}"
            )
            return 0

    def _build_pallet_geometry(self):
        if len(self.pallet_map_polygon) < 3:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] pallet_map_polygon 至少需要 3 个点"
            )
            return None

        cx = sum(point[0] for point in self.pallet_map_polygon) / len(self.pallet_map_polygon)
        cy = sum(point[1] for point in self.pallet_map_polygon) / len(self.pallet_map_polygon)
        center = (cx, cy)

        longest = None
        longest_len = -1.0
        for index, start in enumerate(self.pallet_map_polygon):
            end = self.pallet_map_polygon[(index + 1) % len(self.pallet_map_polygon)]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length > longest_len:
                longest = (dx, dy)
                longest_len = length

        if longest is None or longest_len <= 1e-6:
            self.ros_node.get_logger().error(f"[{self.config_label}] 无法从垛盘 polygon 提取方向")
            return None

        x_axis = (longest[0] / longest_len, longest[1] / longest_len)
        y_axis = (-x_axis[1], x_axis[0])
        projections_x = [self._dot((p[0] - cx, p[1] - cy), x_axis) for p in self.pallet_map_polygon]
        projections_y = [self._dot((p[0] - cx, p[1] - cy), y_axis) for p in self.pallet_map_polygon]
        extent_x = max(projections_x) - min(projections_x)
        extent_y = max(projections_y) - min(projections_y)

        required_x = self.cols * self.box_size_x + max(0, self.cols - 1) * self.slot_gap_x
        required_y = self.rows * self.box_size_y + max(0, self.rows - 1) * self.slot_gap_y
        if required_x > extent_x + 1e-6 or required_y > extent_y + 1e-6:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 码垛网格尺寸可能超出垛盘: "
                f"required=({required_x:.3f}, {required_y:.3f}), "
                f"pallet_extent=({extent_x:.3f}, {extent_y:.3f})"
            )

        return {
            "center": center,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "extent_x": extent_x,
            "extent_y": extent_y,
            "required_x": required_x,
            "required_y": required_y,
            "yaw": math.degrees(math.atan2(x_axis[1], x_axis[0])),
        }

    def _compute_slot_pose(self, geometry, row, col, layer):
        del layer
        start_x = -0.5 * geometry["required_x"] + 0.5 * self.box_size_x
        start_y = -0.5 * geometry["required_y"] + 0.5 * self.box_size_y
        offset_x = start_x + col * (self.box_size_x + self.slot_gap_x)
        offset_y = start_y + row * (self.box_size_y + self.slot_gap_y)
        center = geometry["center"]
        x_axis = geometry["x_axis"]
        y_axis = geometry["y_axis"]
        x = center[0] + offset_x * x_axis[0] + offset_y * y_axis[0]
        y = center[1] + offset_x * x_axis[1] + offset_y * y_axis[1]
        return {
            "x": float(x),
            "y": float(y),
            "z": float(self.pallet_surface_z),
            "yaw": float(geometry["yaw"]),
            "row": int(row),
            "col": int(col),
        }

    def _compute_navigation_pose(self, geometry, slot_pose):
        approach_axis = self._approach_axis(geometry)
        nav_x = slot_pose["x"] + approach_axis[0] * self.place_box_forward_offset_m
        nav_y = slot_pose["y"] + approach_axis[1] * self.place_box_forward_offset_m
        yaw = math.degrees(math.atan2(slot_pose["y"] - nav_y, slot_pose["x"] - nav_x))
        return {
            "x": float(nav_x),
            "y": float(nav_y),
            "yaw": float(yaw),
            "slot_x": float(slot_pose["x"]),
            "slot_y": float(slot_pose["y"]),
            "place_box_forward_offset_m": float(self.place_box_forward_offset_m),
        }

    def _approach_axis(self, geometry):
        x_axis = geometry["x_axis"]
        y_axis = geometry["y_axis"]
        if self.approach_side == "negative_y":
            return (-y_axis[0], -y_axis[1])
        if self.approach_side == "positive_x":
            return x_axis
        if self.approach_side == "negative_x":
            return (-x_axis[0], -x_axis[1])
        return y_axis

    def _publish_visualization(self, geometry, selected_row, selected_col, layer, slot_pose, navigation_pose, expected_box_pose):
        if self.visualization_pub is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        polygon = self._new_marker(marker_id, "pallet_polygon", Marker.LINE_STRIP)
        marker_id += 1
        polygon.scale.x = 0.035
        polygon.pose.orientation.w = 1.0
        polygon.points = [
            Point(x=float(p[0]), y=float(p[1]), z=self.pallet_surface_z + 0.02)
            for p in self.pallet_map_polygon
        ]
        polygon.points.append(polygon.points[0])
        self._set_color(polygon, 1.0, 0.45, 0.0, 1.0)
        marker_array.markers.append(polygon)

        for row in range(self.rows):
            for col in range(self.cols):
                pose = self._compute_slot_pose(geometry, row, col, layer)
                selected = row == selected_row and col == selected_col
                box_marker = self._box_marker(
                    marker_id,
                    "pallet_slots",
                    pose["x"],
                    pose["y"],
                    self.pallet_surface_z + layer * self.box_size_z + self.box_size_z * 0.5,
                    pose["yaw"],
                    selected,
                )
                marker_id += 1
                marker_array.markers.append(box_marker)

        nav_arrow = self._new_marker(marker_id, "pallet_stack_navigation", Marker.ARROW)
        marker_id += 1
        nav_arrow.pose.position.x = navigation_pose["x"]
        nav_arrow.pose.position.y = navigation_pose["y"]
        nav_arrow.pose.position.z = self.pallet_surface_z + 0.12
        yaw_rad = math.radians(navigation_pose["yaw"])
        nav_arrow.pose.orientation.z = math.sin(yaw_rad * 0.5)
        nav_arrow.pose.orientation.w = math.cos(yaw_rad * 0.5)
        nav_arrow.scale.x = 0.7
        nav_arrow.scale.y = 0.12
        nav_arrow.scale.z = 0.12
        self._set_color(nav_arrow, 0.2, 0.9, 1.0, 1.0)
        marker_array.markers.append(nav_arrow)

        line = self._new_marker(marker_id, "pallet_stack_relation", Marker.LINE_LIST)
        marker_id += 1
        line.scale.x = 0.025
        line.points = [
            Point(x=navigation_pose["x"], y=navigation_pose["y"], z=self.pallet_surface_z + 0.08),
            Point(x=slot_pose["x"], y=slot_pose["y"], z=expected_box_pose["z"]),
        ]
        self._set_color(line, 0.2, 0.9, 1.0, 0.95)
        marker_array.markers.append(line)

        text = self._new_marker(marker_id, "pallet_stack_text", Marker.TEXT_VIEW_FACING)
        text.pose.position.x = slot_pose["x"]
        text.pose.position.y = slot_pose["y"]
        text.pose.position.z = expected_box_pose["z"] + 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.12
        text.text = (
            "PALLET STACK TARGET\n"
            f"layer={layer} row={selected_row} col={selected_col}\n"
            f"slot=({slot_pose['x']:.2f},{slot_pose['y']:.2f}) yaw={slot_pose['yaw']:.1f}\n"
            f"nav=({navigation_pose['x']:.2f},{navigation_pose['y']:.2f}) yaw={navigation_pose['yaw']:.1f}\n"
            f"plane_z={self.pallet_surface_z + layer * self.box_size_z + self.place_clearance_z:.2f}"
        )
        self._set_color(text, 1.0, 1.0, 1.0, 1.0)
        marker_array.markers.append(text)

        self.visualization_pub.publish(marker_array)

    def _box_marker(self, marker_id, namespace, x, y, z, yaw_deg, selected):
        marker = self._new_marker(marker_id, namespace, Marker.CUBE)
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        yaw_rad = math.radians(yaw_deg)
        marker.pose.orientation.z = math.sin(yaw_rad * 0.5)
        marker.pose.orientation.w = math.cos(yaw_rad * 0.5)
        marker.scale.x = self.box_size_x
        marker.scale.y = self.box_size_y
        marker.scale.z = self.box_size_z
        if selected:
            self._set_color(marker, 0.0, 1.0, 0.25, 0.45)
        else:
            self._set_color(marker, 0.55, 0.55, 0.55, 0.22)
        return marker

    def _new_marker(self, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _set_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    @staticmethod
    def _dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def _parse_polygon(raw_polygon):
        polygon = parse_param_value(raw_polygon)
        if not isinstance(polygon, (list, tuple)):
            return []
        parsed = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            parsed.append((float(point[0]), float(point[1])))
        return parsed

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPalletStackTarget start: "
            f"rows={self.rows}, cols={self.cols}, max_layers={self.max_layers}, "
            f"box=({self.box_size_x:.2f},{self.box_size_y:.2f},{self.box_size_z:.2f}), "
            f"stack_count_key={self.stack_count_key}, nav_key={self.navigation_target_key}, "
            f"plane_key={self.place_plane_height_key}"
        )
