"""计算 move_box 码垛目标。

第一版码垛采用“固定垛盘 polygon/参考点 + 网格槽位 + stack_count”的确定性策略：

1. 优先从 JSON 读取两个 slot_reference_points，作为 row0 的两个箱心参考点；
   未配置时读取垛盘 map 多边形，自动取最长边作为 pallet-x。
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
        # 可选：两个 map 参考点，表示 row0/col0 与 row0/col1 的目标箱心。
        # 配置后 slot0/slot1 直接落在这两个点上，后续 row 按箱宽+间隙朝垛盘内部推进。
        self.slot_reference_points = self._parse_polygon(params.get("slot_reference_points", []))
        # 可选：显式指定 row 增长方向；不填时由“参考点连线 -> 垛盘 polygon 中心”自动判断。
        self.slot_reference_row_axis = self._parse_vector(params.get("slot_reference_row_axis", []))
        # 可选：显式指定码垛箱体 yaw。调换 slot_reference_points 顺序时，参考点连线 yaw
        # 可能从 0deg 翻成 180deg；若只想改变放置顺序，不想改变箱体朝向，就配置本参数。
        self.slot_yaw_deg = self._optional_float(params.get("slot_yaw_deg", None))
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
        self._log_stack_target_detail(
            geometry,
            stack_count,
            layer,
            row,
            col,
            slot_pose,
            navigation_pose,
            place_plane_height,
            expected_box_pose,
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算码垛目标: "
            f"stack_count={stack_count}, layer={layer}, row={row}, col={col}, "
            f"slot=({slot_pose['x']:.3f}, {slot_pose['y']:.3f}, {slot_pose['yaw']:.2f}deg), "
            f"nav=({navigation_pose['x']:.3f}, {navigation_pose['y']:.3f}, {navigation_pose['yaw']:.2f}deg), "
            f"place_plane_z={place_plane_height:.3f}, "
            f"keys nav={self.navigation_target_key}, plane={self.place_plane_height_key}"
        )
        return Status.SUCCESS

    def _log_stack_target_detail(
        self,
        geometry,
        stack_count,
        layer,
        row,
        col,
        slot_pose,
        navigation_pose,
        place_plane_height,
        expected_box_pose,
    ):
        color_start = "\033[95m"
        color_end = "\033[0m"
        geometry_mode = geometry.get("mode", "polygon_centered")
        reference_text = ""
        if geometry_mode == "reference_points":
            p0 = geometry.get("reference_origin", (0.0, 0.0))
            p1 = geometry.get("reference_second", (0.0, 0.0))
            reference_text = (
                f", ref0=({p0[0]:.3f}, {p0[1]:.3f}), "
                f"ref1=({p1[0]:.3f}, {p1[1]:.3f})"
            )

        self.ros_node.get_logger().info(
            color_start
            + (
                f"[{self.config_label}] 🟣 PALLET_STACK_TARGET_DETAIL | "
                f"mode={geometry_mode}{reference_text} | "
                f"stack_count={stack_count}, layer={layer}, row={row}, col={col} | "
                f"slot_surface_map=({slot_pose['x']:.3f}, {slot_pose['y']:.3f}, "
                f"{slot_pose['z']:.3f}, yaw={slot_pose['yaw']:.2f}deg) | "
                f"expected_box_center_map=({expected_box_pose['x']:.3f}, "
                f"{expected_box_pose['y']:.3f}, {expected_box_pose['z']:.3f}, "
                f"yaw={expected_box_pose['yaw']:.2f}deg) | "
                f"navigation_target_map=({navigation_pose['x']:.3f}, "
                f"{navigation_pose['y']:.3f}, yaw={navigation_pose['yaw']:.2f}deg) | "
                f"place_plane_z_map={place_plane_height:.3f}, "
                f"pallet_surface_z_map={self.pallet_surface_z:.3f}, "
                f"place_clearance_z={self.place_clearance_z:.3f}, "
                f"box_size=({self.box_size_x:.3f}, {self.box_size_y:.3f}, {self.box_size_z:.3f}), "
                f"approach_side={self.approach_side}, "
                f"place_box_forward_offset_m={self.place_box_forward_offset_m:.3f}"
            )
            + color_end
        )

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
        if len(self.slot_reference_points) >= 2:
            return self._build_reference_points_geometry()

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
            "yaw": self._slot_yaw_or_axis_yaw(x_axis),
        }

    def _build_reference_points_geometry(self):
        p0 = self.slot_reference_points[0]
        p1 = self.slot_reference_points[1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        slot_step_x = math.hypot(dx, dy)
        if slot_step_x <= 1e-6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] slot_reference_points 两点重合，无法定义码垛方向: {self.slot_reference_points}"
            )
            return None

        x_axis = (dx / slot_step_x, dy / slot_step_x)
        y_axis = self._reference_row_axis(x_axis, p0, p1)
        required_x = self.cols * self.box_size_x + max(0, self.cols - 1) * self.slot_gap_x
        required_y = self.rows * self.box_size_y + max(0, self.rows - 1) * self.slot_gap_y
        expected_step_x = self.box_size_x + self.slot_gap_x
        if self.cols > 1 and abs(slot_step_x - expected_step_x) > 0.08:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 两参考点间距与箱长+间隙差异较大: "
                f"reference_step={slot_step_x:.3f}, expected={expected_step_x:.3f}"
            )

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 使用码垛参考点生成槽位: "
            f"p0=({p0[0]:.3f},{p0[1]:.3f}), p1=({p1[0]:.3f},{p1[1]:.3f}), "
            f"x_axis=({x_axis[0]:.3f},{x_axis[1]:.3f}), "
            f"row_axis=({y_axis[0]:.3f},{y_axis[1]:.3f})"
        )
        return {
            "mode": "reference_points",
            "reference_origin": p0,
            "reference_second": p1,
            "slot_step_x": slot_step_x,
            "center": ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "extent_x": required_x,
            "extent_y": required_y,
            "required_x": required_x,
            "required_y": required_y,
            "yaw": self._slot_yaw_or_axis_yaw(x_axis),
        }

    def _reference_row_axis(self, x_axis, p0, p1):
        explicit_axis = self.slot_reference_row_axis
        if explicit_axis is not None:
            return explicit_axis

        candidate = (-x_axis[1], x_axis[0])
        if len(self.pallet_map_polygon) >= 3:
            cx = sum(point[0] for point in self.pallet_map_polygon) / len(self.pallet_map_polygon)
            cy = sum(point[1] for point in self.pallet_map_polygon) / len(self.pallet_map_polygon)
            mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
            to_center = (cx - mid[0], cy - mid[1])
            if self._dot(to_center, candidate) < 0.0:
                candidate = (-candidate[0], -candidate[1])
        return candidate

    def _compute_slot_pose(self, geometry, row, col, layer):
        del layer
        if geometry.get("mode") == "reference_points":
            origin = geometry["reference_origin"]
            x_axis = geometry["x_axis"]
            y_axis = geometry["y_axis"]
            offset_x = col * geometry["slot_step_x"]
            offset_y = row * (self.box_size_y + self.slot_gap_y)
            x = origin[0] + offset_x * x_axis[0] + offset_y * y_axis[0]
            y = origin[1] + offset_x * x_axis[1] + offset_y * y_axis[1]
            return {
                "x": float(x),
                "y": float(y),
                "z": float(self.pallet_surface_z),
                "yaw": float(geometry["yaw"]),
                "row": int(row),
                "col": int(col),
            }

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

        if geometry.get("mode") == "reference_points":
            for ref_index, ref_point in enumerate(
                (geometry["reference_origin"], geometry["reference_second"])
            ):
                ref_marker = self._new_marker(
                    marker_id, "pallet_slot_reference_points", Marker.SPHERE
                )
                marker_id += 1
                ref_marker.pose.position.x = float(ref_point[0])
                ref_marker.pose.position.y = float(ref_point[1])
                ref_marker.pose.position.z = self.pallet_surface_z + 0.08
                ref_marker.scale.x = ref_marker.scale.y = ref_marker.scale.z = 0.12
                self._set_color(ref_marker, 1.0, 0.0, 1.0, 1.0)
                marker_array.markers.append(ref_marker)

                ref_text = self._new_marker(
                    marker_id, "pallet_slot_reference_text", Marker.TEXT_VIEW_FACING
                )
                marker_id += 1
                ref_text.pose.position.x = float(ref_point[0])
                ref_text.pose.position.y = float(ref_point[1])
                ref_text.pose.position.z = self.pallet_surface_z + 0.22
                ref_text.scale.z = 0.09
                ref_text.text = f"STACK REF{ref_index}\n({ref_point[0]:.2f},{ref_point[1]:.2f})"
                self._set_color(ref_text, 1.0, 0.0, 1.0, 1.0)
                marker_array.markers.append(ref_text)

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

    @staticmethod
    def _parse_vector(raw_vector):
        vector = parse_param_value(raw_vector)
        if not isinstance(vector, (list, tuple)) or len(vector) < 2:
            return None
        try:
            x = float(vector[0])
            y = float(vector[1])
        except (TypeError, ValueError):
            return None
        norm = math.hypot(x, y)
        if not math.isfinite(norm) or norm <= 1e-6:
            return None
        return (x / norm, y / norm)

    @staticmethod
    def _optional_float(raw_value):
        value = parse_param_value(raw_value)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _slot_yaw_or_axis_yaw(self, x_axis):
        if self.slot_yaw_deg is not None:
            return float(self.slot_yaw_deg)
        return math.degrees(math.atan2(x_axis[1], x_axis[0]))

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPalletStackTarget start: "
            f"rows={self.rows}, cols={self.cols}, max_layers={self.max_layers}, "
            f"box=({self.box_size_x:.2f},{self.box_size_y:.2f},{self.box_size_z:.2f}), "
            f"slot_yaw={self.slot_yaw_deg if self.slot_yaw_deg is not None else '<axis>'}, "
            f"stack_count_key={self.stack_count_key}, nav_key={self.navigation_target_key}, "
            f"plane_key={self.place_plane_height_key}"
        )
