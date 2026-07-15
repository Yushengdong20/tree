"""计算 move_box 码垛目标、导航站位、放箱策略与动作参考点。

本节点是码垛策略版流程里的唯一“码垛规划节点”：

1. 根据固定垛盘 polygon / slot_reference_points / stack_count 选择本轮槽位。
2. 输出最终放置箱心、垛盘外导航目标和放置平面高度。
3. 根据 stack_count 与 slot 行列推断相邻已放箱，选择放置策略：
   - direct_place：无明显邻箱，直接放。
   - right_push_left_place：先放在目标右侧，再由右爪向左推到最终位。
   - left_push_right_place：先放在目标左侧，再由左爪向右推到最终位。
4. 根据当前双爪实际点近似估计“手里箱子”的当前姿态。
5. 输出策略、预落位箱体、最终箱体、推送方向、释放爪侧、动作夹爪点等 blackboard key。
6. 发布独立 RViz 诊断话题，供实机调试和方案确认。

注意：本节点只负责“规划 + 可视化 + blackboard 输出”，不直接控制手臂。
实际放箱动作由后续 ArmsToPose / OpenClaw 等节点执行。
"""

import math

import numpy as np
import py_trees
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from tree.runtime.http.move_and_grab_flow import (
    Pose2D,
    transform_base_point_to_global,
    transform_global_point_to_base,
)
from tree.utils.geometry import get_odom_pose_transformer, ypr_to_rotation_matrix
from tree.utils.params import parse_param_value

from ..base import TimedMockAction


class ComputeMoveBoxPalletPlaceStrategy(TimedMockAction):
    """根据码垛目标和当前夹爪状态计算放箱策略。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.stack_count_key = str(params.get("stack_count_key", "move_box_pallet_stack_count")).strip()

        self.pallet_map_polygon = self._parse_polygon(
            params.get(
                "pallet_map_polygon",
                [[-0.3, 1.0], [1.0, 1.0], [1.0, 0.1], [-0.3, 0.1]],
            )
        )
        self.slot_reference_points = self._parse_polygon(params.get("slot_reference_points", []))
        self.slot_reference_row_axis = self._parse_vector(params.get("slot_reference_row_axis", []))
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

        self.navigation_target_key = str(
            params.get("navigation_target_key", "move_box_pallet_stack_navigation_target")
        ).strip()
        self.slot_pose_key = str(params.get("slot_pose_key", "move_box_pallet_stack_slot_pose")).strip()
        self.expected_box_pose_key = str(
            params.get("expected_box_pose_key", "move_box_pallet_stack_expected_box_pose")
        ).strip()
        self.place_plane_height_key = str(
            params.get("place_plane_height_key", "move_box_pallet_stack_place_plane_height")
        ).strip()

        self.strategy_key = str(params.get("strategy_key", "move_box_pallet_place_strategy")).strip()
        self.final_box_pose_key = str(
            params.get("final_box_pose_key", "move_box_pallet_place_final_box_pose")
        ).strip()
        self.pre_box_pose_key = str(
            params.get("pre_box_pose_key", "move_box_pallet_place_pre_box_pose")
        ).strip()
        self.held_box_pose_key = str(
            params.get("held_box_pose_key", "move_box_pallet_place_held_box_pose")
        ).strip()
        self.push_direction_key = str(
            params.get("push_direction_key", "move_box_pallet_place_push_direction")
        ).strip()
        self.release_first_side_key = str(
            params.get("release_first_side_key", "move_box_pallet_place_release_first_side")
        ).strip()
        self.push_side_key = str(params.get("push_side_key", "move_box_pallet_place_push_side")).strip()
        self.pre_place_left_claw_point_key = str(
            params.get("pre_place_left_claw_point_key", "move_box_pallet_pre_place_left_claw_point")
        ).strip()
        self.pre_place_right_claw_point_key = str(
            params.get("pre_place_right_claw_point_key", "move_box_pallet_pre_place_right_claw_point")
        ).strip()
        self.push_left_claw_point_key = str(
            params.get("push_left_claw_point_key", "move_box_pallet_push_left_claw_point")
        ).strip()
        self.push_right_claw_point_key = str(
            params.get("push_right_claw_point_key", "move_box_pallet_push_right_claw_point")
        ).strip()
        self.lift_left_claw_point_key = str(
            params.get("lift_left_claw_point_key", "move_box_pallet_lift_left_claw_point")
        ).strip()
        self.lift_right_claw_point_key = str(
            params.get("lift_right_claw_point_key", "move_box_pallet_lift_right_claw_point")
        ).strip()

        self.pre_place_lateral_offset_m = float(params.get("pre_place_lateral_offset_m", 0.10))
        self.push_distance_m = float(params.get("push_distance_m", self.pre_place_lateral_offset_m))
        self.claw_lift_clearance_m = float(params.get("claw_lift_clearance_m", 0.18))
        self.strategy_mode = str(params.get("strategy_mode", "auto")).strip().lower()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()

        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/pallet_place_dynamic_estimate_markers")
        ).strip()

        for key in (self.services_key, self.stack_count_key):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

        for key in (
            self.navigation_target_key,
            self.slot_pose_key,
            self.expected_box_pose_key,
            self.place_plane_height_key,
            self.strategy_key,
            self.final_box_pose_key,
            self.pre_box_pose_key,
            self.held_box_pose_key,
            self.push_direction_key,
            self.release_first_side_key,
            self.push_side_key,
            self.pre_place_left_claw_point_key,
            self.pre_place_right_claw_point_key,
            self.push_left_claw_point_key,
            self.push_right_claw_point_key,
            self.lift_left_claw_point_key,
            self.lift_right_claw_point_key,
        ):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame="base_link",
        )
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

        stack_count = self._read_stack_count()
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

        x_axis, y_axis = self._slot_axes(expected_box_pose)
        strategy_info = self._select_strategy(row=row, col=col, x_axis=x_axis, stack_count=stack_count)
        final_box_pose = dict(expected_box_pose)
        pre_box_pose = self._compute_pre_box_pose(final_box_pose, strategy_info)
        held_box_pose = self._estimate_held_box_pose(final_box_pose)
        push_direction = self._compute_push_direction(strategy_info)
        action_points = self._compute_action_claw_points(
            final_box_pose=final_box_pose,
            pre_box_pose=pre_box_pose,
        )

        self.blackboard.set(self.navigation_target_key, navigation_pose, overwrite=True)
        self.blackboard.set(self.slot_pose_key, slot_pose, overwrite=True)
        self.blackboard.set(self.expected_box_pose_key, expected_box_pose, overwrite=True)
        self.blackboard.set(self.place_plane_height_key, float(place_plane_height), overwrite=True)
        self.blackboard.set(self.strategy_key, strategy_info["strategy"], overwrite=True)
        self.blackboard.set(self.final_box_pose_key, final_box_pose, overwrite=True)
        self.blackboard.set(self.pre_box_pose_key, pre_box_pose, overwrite=True)
        self.blackboard.set(self.held_box_pose_key, held_box_pose, overwrite=True)
        self.blackboard.set(self.push_direction_key, push_direction, overwrite=True)
        self.blackboard.set(self.release_first_side_key, strategy_info["release_first_side"], overwrite=True)
        self.blackboard.set(self.push_side_key, strategy_info["push_side"], overwrite=True)
        if action_points is not None:
            self.blackboard.set(self.pre_place_left_claw_point_key, action_points["pre_left"], overwrite=True)
            self.blackboard.set(self.pre_place_right_claw_point_key, action_points["pre_right"], overwrite=True)
            self.blackboard.set(self.push_left_claw_point_key, action_points["push_left"], overwrite=True)
            self.blackboard.set(self.push_right_claw_point_key, action_points["push_right"], overwrite=True)
            self.blackboard.set(self.lift_left_claw_point_key, action_points["lift_left"], overwrite=True)
            self.blackboard.set(self.lift_right_claw_point_key, action_points["lift_right"], overwrite=True)

        self._publish_visualization(
            geometry=geometry,
            selected_row=row,
            selected_col=col,
            layer=layer,
            slot_pose=slot_pose,
            navigation_pose=navigation_pose,
            final_box_pose=final_box_pose,
            pre_box_pose=pre_box_pose,
            held_box_pose=held_box_pose,
            push_direction=push_direction,
            strategy_info=strategy_info,
            x_axis=x_axis,
            y_axis=y_axis,
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算码垛放箱策略: "
            f"strategy={strategy_info['strategy']}, reason={strategy_info['reason']}, "
            f"stack_count={stack_count}, layer={layer}, row={row}, col={col}, "
            f"slot=({slot_pose['x']:.3f}, {slot_pose['y']:.3f}, {slot_pose['yaw']:.2f}deg), "
            f"nav=({navigation_pose['x']:.3f}, {navigation_pose['y']:.3f}, "
            f"{navigation_pose['yaw']:.2f}deg), place_plane_z={place_plane_height:.3f}, "
            f"final_box=({final_box_pose['x']:.3f}, {final_box_pose['y']:.3f}, "
            f"{final_box_pose.get('z', 0.0):.3f}, yaw={final_box_pose.get('yaw', 0.0):.2f}), "
            f"pre_box=({pre_box_pose['x']:.3f}, {pre_box_pose['y']:.3f}, "
            f"{pre_box_pose.get('z', 0.0):.3f}), "
            f"push=({push_direction['x']:.3f}, {push_direction['y']:.3f}, "
            f"distance={push_direction['distance']:.3f}), "
            f"release_first={strategy_info['release_first_side']}, push_side={strategy_info['push_side']}, "
            f"action_points={'ready' if action_points is not None else 'unavailable'}, "
            f"topic={self.visualization_topic or '<disabled>'}"
        )
        return Status.SUCCESS

    def _select_strategy(self, row, col, x_axis, stack_count):
        if self.strategy_mode in ("direct", "direct_place"):
            return self._strategy("direct_place", "none", "none", "配置强制直接放箱", (0.0, 0.0))
        if self.strategy_mode in ("right_push_left", "right_push_left_place"):
            return self._strategy(
                "right_push_left_place",
                "left",
                "right",
                "配置强制右爪向左推",
                tuple(-value for value in x_axis),
            )
        if self.strategy_mode in ("left_push_right", "left_push_right_place"):
            return self._strategy(
                "left_push_right_place",
                "right",
                "left",
                "配置强制左爪向右推",
                x_axis,
            )

        # 自动策略第一版基于 stack_count/slot 推断已有码垛箱：
        # 同一 row 中，如果左/右相邻 col 已经在本轮之前出现过，就用推箱策略贴靠邻箱。
        slot_index = row * max(1, self.cols) + col
        left_neighbor_index = row * max(1, self.cols) + col - 1
        right_neighbor_index = row * max(1, self.cols) + col + 1
        left_neighbor_placed = col > 0 and left_neighbor_index < stack_count
        right_neighbor_placed = col < self.cols - 1 and right_neighbor_index < stack_count
        if left_neighbor_placed:
            return self._strategy(
                "right_push_left_place",
                "left",
                "right",
                "左侧已有槽位，先右偏预落位，再由右爪向左推",
                tuple(-value for value in x_axis),
            )
        if right_neighbor_placed:
            return self._strategy(
                "left_push_right_place",
                "right",
                "left",
                "右侧已有槽位，先左偏预落位，再由左爪向右推",
                x_axis,
            )
        return self._strategy("direct_place", "both", "none", "当前槽位无同排邻箱，直接放箱", (0.0, 0.0))

    def _strategy(self, strategy, release_first_side, push_side, reason, push_axis):
        return {
            "strategy": strategy,
            "release_first_side": release_first_side,
            "push_side": push_side,
            "reason": reason,
            "push_axis": tuple(float(value) for value in push_axis),
        }

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
            "mode": "polygon_centered",
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
                f"[{self.config_label}] slot_reference_points 两点重合，无法定义码垛方向: "
                f"{self.slot_reference_points}"
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

    def _compute_pre_box_pose(self, final_box_pose, strategy_info):
        pre_pose = dict(final_box_pose)
        strategy = strategy_info["strategy"]
        push_axis = strategy_info["push_axis"]
        # push_axis 表示从预落位推到最终位的方向；预落位在反方向。
        if strategy != "direct_place":
            pre_pose["x"] = float(final_box_pose["x"]) - push_axis[0] * self.pre_place_lateral_offset_m
            pre_pose["y"] = float(final_box_pose["y"]) - push_axis[1] * self.pre_place_lateral_offset_m
        return pre_pose

    def _compute_push_direction(self, strategy_info):
        push_axis = strategy_info["push_axis"]
        distance = 0.0 if strategy_info["strategy"] == "direct_place" else self.push_distance_m
        return {
            "x": float(push_axis[0]),
            "y": float(push_axis[1]),
            "distance": float(distance),
        }

    def _estimate_held_box_pose(self, fallback_pose):
        current_pose = self.odom_transformer.get_current_pose()
        claw_pair = self._get_current_claw_pair()
        if current_pose is None or claw_pair is None:
            pose = dict(fallback_pose)
            pose["source"] = "fallback_final_box_pose"
            return pose

        left_claw, right_claw = claw_pair
        center_base = (left_claw + right_claw) * 0.5
        current_pose_2d = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        center_map = transform_base_point_to_global(current_pose_2d, center_base[0], center_base[1])
        pose = {
            "x": float(center_map["x"]),
            "y": float(center_map["y"]),
            "z": float(fallback_pose.get("z", center_base[2] + current_pose[2])),
            "yaw": float(fallback_pose.get("yaw", current_pose[3])),
            "source": "current_claw_midpoint",
        }
        return pose

    def _compute_action_claw_points(self, final_box_pose, pre_box_pose):
        """根据当前双爪点和目标箱心，生成放箱动作需要的 base_link 夹爪点。

        这里采用保守的一阶近似：当前双爪中点代表当前手中箱心。
        将当前双爪整体平移到预落位箱心/最终箱心，保持左右爪相对距离不变。
        """
        current_pose = self.odom_transformer.get_current_pose()
        claw_pair = self._get_current_claw_pair()
        if current_pose is None or claw_pair is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 无法计算真实放箱动作点: "
                f"odom_ready={current_pose is not None}, claw_pair_ready={claw_pair is not None}"
            )
            return None

        left_claw, right_claw = claw_pair
        current_center = (left_claw + right_claw) * 0.5
        target_claw_z = self._resolve_target_claw_z(current_pose)
        final_center = self._box_center_map_to_base(final_box_pose, current_pose, target_claw_z)
        pre_center = self._box_center_map_to_base(pre_box_pose, current_pose, target_claw_z)
        if final_center is None or pre_center is None:
            return None

        pre_delta = pre_center - current_center
        final_delta = final_center - current_center
        pre_left = left_claw + pre_delta
        pre_right = right_claw + pre_delta
        final_left = left_claw + final_delta
        final_right = right_claw + final_delta
        lift_left = np.array(final_left, dtype=float)
        lift_right = np.array(final_right, dtype=float)
        lift_left[2] += self.claw_lift_clearance_m
        lift_right[2] += self.claw_lift_clearance_m

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算码垛放箱动作点(base_link): "
            f"current_center=({current_center[0]:.3f},{current_center[1]:.3f},{current_center[2]:.3f}), "
            f"pre_center=({pre_center[0]:.3f},{pre_center[1]:.3f},{pre_center[2]:.3f}), "
            f"final_center=({final_center[0]:.3f},{final_center[1]:.3f},{final_center[2]:.3f}), "
            f"pre_left=({pre_left[0]:.3f},{pre_left[1]:.3f},{pre_left[2]:.3f}), "
            f"pre_right=({pre_right[0]:.3f},{pre_right[1]:.3f},{pre_right[2]:.3f}), "
            f"final_left=({final_left[0]:.3f},{final_left[1]:.3f},{final_left[2]:.3f}), "
            f"final_right=({final_right[0]:.3f},{final_right[1]:.3f},{final_right[2]:.3f})"
        )
        return {
            "pre_left": pre_left,
            "pre_right": pre_right,
            "push_left": final_left,
            "push_right": final_right,
            "lift_left": lift_left,
            "lift_right": lift_right,
        }

    def _box_center_map_to_base(self, box_pose, current_pose, target_claw_z):
        current_pose_2d = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        center_xy = transform_global_point_to_base(
            current_pose_2d,
            float(box_pose["x"]),
            float(box_pose["y"]),
        )
        return np.array(
            [
                float(center_xy["x"]),
                float(center_xy["y"]),
                float(target_claw_z),
            ],
            dtype=float,
        )

    def _resolve_target_claw_z(self, current_pose):
        place_plane_height = None
        if self.place_plane_height_key and self.blackboard.exists(self.place_plane_height_key):
            try:
                place_plane_height = float(self.blackboard.get(self.place_plane_height_key))
            except (TypeError, ValueError):
                place_plane_height = None
        if place_plane_height is None:
            return float(current_pose[2]) + self.box_size_z
        # 目前码垛 place_plane_height 来自 map 系，转成 base_link 后再加箱高。
        return float(place_plane_height) - float(current_pose[2]) + self.box_size_z

    def _get_current_claw_pair(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            return None

        left = self._current_claw_point(services.arm_controller, "left")
        right = self._current_claw_point(services.arm_controller, "right")
        if left is None or right is None:
            return None
        return left, right

    def _current_claw_point(self, arm_controller, side):
        if hasattr(arm_controller, "get_current_end_effector_pose"):
            current_pose = arm_controller.get_current_end_effector_pose(side)
        else:
            current_pose = None

        if current_pose is None:
            current_pose = (
                arm_controller.current_left_target
                if side == "left"
                else arm_controller.current_right_target
            )
        if current_pose is None or len(current_pose) != 6:
            return None

        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            return None

        translation, _ = transform
        rotation = ypr_to_rotation_matrix(current_pose[3:6])
        end_effector_point = np.array(current_pose[:3], dtype=float)
        return end_effector_point + rotation.dot(translation)

    def _read_pose(self, key, required=False):
        if not key or not self.blackboard.exists(key):
            if required:
                self.ros_node.get_logger().error(f"[{self.config_label}] blackboard 缺少 pose: key={key}")
            return None
        value = self.blackboard.get(key)
        if not isinstance(value, dict):
            if required:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] pose 必须是 dict: key={key}, value={value!r}"
                )
            return None
        try:
            pose = dict(value)
            pose["x"] = float(pose["x"])
            pose["y"] = float(pose["y"])
            pose["z"] = float(pose.get("z", 0.0))
            pose["yaw"] = float(pose.get("yaw", 0.0))
            return pose
        except (KeyError, TypeError, ValueError) as exc:
            if required:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] pose 缺少有效 x/y/z/yaw: key={key}, value={value!r}, error={exc}"
                )
            return None

    def _read_stack_count(self):
        if not self.stack_count_key or not self.blackboard.exists(self.stack_count_key):
            return 0
        try:
            return max(0, int(self.blackboard.get(self.stack_count_key)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _slot_axes(pose):
        yaw_rad = math.radians(float(pose.get("yaw", 0.0)))
        x_axis = (math.cos(yaw_rad), math.sin(yaw_rad))
        y_axis = (-math.sin(yaw_rad), math.cos(yaw_rad))
        return x_axis, y_axis

    def _publish_visualization(
        self,
        geometry,
        selected_row,
        selected_col,
        layer,
        slot_pose,
        navigation_pose,
        final_box_pose,
        pre_box_pose,
        held_box_pose,
        push_direction,
        strategy_info,
        x_axis,
        y_axis,
    ):
        if self.visualization_pub is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        marker_id = self._append_pallet_polygon(marker_array, marker_id)
        if geometry.get("mode") == "reference_points":
            marker_id = self._append_reference_points(marker_array, marker_id, geometry)
        marker_id = self._append_all_slots(
            marker_array,
            marker_id,
            geometry,
            selected_row,
            selected_col,
            layer,
        )
        marker_id = self._append_navigation_marker(
            marker_array,
            marker_id,
            navigation_pose,
            final_box_pose,
        )
        marker_id = self._append_box(
            marker_array,
            marker_id,
            "held_box_estimate",
            held_box_pose,
            color=(0.2, 0.8, 1.0, 0.35),
        )
        marker_id = self._append_box(
            marker_array,
            marker_id,
            "pre_place_box",
            pre_box_pose,
            color=(1.0, 0.75, 0.05, 0.38),
        )
        marker_id = self._append_box(
            marker_array,
            marker_id,
            "final_place_box",
            final_box_pose,
            color=(0.0, 1.0, 0.25, 0.45),
        )

        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "slot_center",
            slot_pose["x"],
            slot_pose["y"],
            final_box_pose.get("z", 0.0),
            color=(0.0, 1.0, 0.25, 1.0),
            scale=0.09,
        )

        if push_direction["distance"] > 1e-6:
            start_x = pre_box_pose["x"]
            start_y = pre_box_pose["y"]
            end_x = start_x + push_direction["x"] * push_direction["distance"]
            end_y = start_y + push_direction["y"] * push_direction["distance"]
            marker_id = self._append_arrow(
                marker_array,
                marker_id,
                "push_direction",
                start_x,
                start_y,
                pre_box_pose.get("z", 0.0) + 0.25,
                end_x,
                end_y,
                pre_box_pose.get("z", 0.0) + 0.25,
                color=(1.0, 0.25, 0.0, 1.0),
            )

        text = self._new_marker(marker_id, "pallet_place_strategy_text", Marker.TEXT_VIEW_FACING)
        text.pose.position.x = final_box_pose["x"]
        text.pose.position.y = final_box_pose["y"]
        text.pose.position.z = final_box_pose.get("z", 0.0) + 0.65
        text.pose.orientation.w = 1.0
        text.scale.z = 0.11
        text.text = (
            "PALLET PLACE STRATEGY\n"
            f"strategy={strategy_info['strategy']}\n"
            f"reason={strategy_info['reason']}\n"
            f"slot=({slot_pose['x']:.2f},{slot_pose['y']:.2f}) "
            f"nav=({navigation_pose['x']:.2f},{navigation_pose['y']:.2f})\n"
            f"held=({held_box_pose['x']:.2f},{held_box_pose['y']:.2f}) source={held_box_pose.get('source', '')}\n"
            f"pre=({pre_box_pose['x']:.2f},{pre_box_pose['y']:.2f}) "
            f"final=({final_box_pose['x']:.2f},{final_box_pose['y']:.2f})\n"
            f"release={strategy_info['release_first_side']} push={strategy_info['push_side']}"
        )
        self._set_color(text, 1.0, 1.0, 1.0, 1.0)
        marker_array.markers.append(text)

        self.visualization_pub.publish(marker_array)

    def _append_pallet_polygon(self, marker_array, marker_id):
        if len(self.pallet_map_polygon) < 3:
            return marker_id
        polygon = self._new_marker(marker_id, "pallet_polygon", Marker.LINE_STRIP)
        polygon.scale.x = 0.035
        polygon.points = [
            Point(x=float(p[0]), y=float(p[1]), z=self.pallet_surface_z + 0.02)
            for p in self.pallet_map_polygon
        ]
        polygon.points.append(polygon.points[0])
        self._set_color(polygon, 1.0, 0.45, 0.0, 1.0)
        marker_array.markers.append(polygon)
        return marker_id + 1

    def _append_reference_points(self, marker_array, marker_id, geometry):
        for ref_index, ref_point in enumerate((geometry["reference_origin"], geometry["reference_second"])):
            ref_marker = self._new_marker(marker_id, "pallet_slot_reference_points", Marker.SPHERE)
            marker_id += 1
            ref_marker.pose.position.x = float(ref_point[0])
            ref_marker.pose.position.y = float(ref_point[1])
            ref_marker.pose.position.z = self.pallet_surface_z + 0.08
            ref_marker.scale.x = ref_marker.scale.y = ref_marker.scale.z = 0.12
            self._set_color(ref_marker, 1.0, 0.0, 1.0, 1.0)
            marker_array.markers.append(ref_marker)

            ref_text = self._new_marker(marker_id, "pallet_slot_reference_text", Marker.TEXT_VIEW_FACING)
            marker_id += 1
            ref_text.pose.position.x = float(ref_point[0])
            ref_text.pose.position.y = float(ref_point[1])
            ref_text.pose.position.z = self.pallet_surface_z + 0.22
            ref_text.scale.z = 0.09
            ref_text.text = f"STACK REF{ref_index}\n({ref_point[0]:.2f},{ref_point[1]:.2f})"
            self._set_color(ref_text, 1.0, 0.0, 1.0, 1.0)
            marker_array.markers.append(ref_text)
        return marker_id

    def _append_all_slots(self, marker_array, marker_id, geometry, selected_row, selected_col, layer):
        for row in range(self.rows):
            for col in range(self.cols):
                pose = self._compute_slot_pose(geometry, row, col, layer)
                selected = row == selected_row and col == selected_col
                marker = self._new_marker(marker_id, "pallet_slots", Marker.CUBE)
                marker_id += 1
                marker.pose.position.x = pose["x"]
                marker.pose.position.y = pose["y"]
                marker.pose.position.z = self.pallet_surface_z + layer * self.box_size_z + self.box_size_z * 0.5
                yaw_rad = math.radians(pose["yaw"])
                marker.pose.orientation.z = math.sin(yaw_rad * 0.5)
                marker.pose.orientation.w = math.cos(yaw_rad * 0.5)
                marker.scale.x = self.box_size_x
                marker.scale.y = self.box_size_y
                marker.scale.z = self.box_size_z
                if selected:
                    self._set_color(marker, 0.0, 1.0, 0.25, 0.30)
                else:
                    self._set_color(marker, 0.55, 0.55, 0.55, 0.18)
                marker_array.markers.append(marker)
        return marker_id

    def _append_navigation_marker(self, marker_array, marker_id, navigation_pose, final_box_pose):
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
            Point(x=final_box_pose["x"], y=final_box_pose["y"], z=final_box_pose.get("z", 0.0)),
        ]
        self._set_color(line, 0.2, 0.9, 1.0, 0.95)
        marker_array.markers.append(line)
        return marker_id

    def _append_box(self, marker_array, marker_id, namespace, pose, color):
        marker = self._new_marker(marker_id, namespace, Marker.CUBE)
        marker.pose.position.x = float(pose["x"])
        marker.pose.position.y = float(pose["y"])
        marker.pose.position.z = float(pose.get("z", 0.0))
        yaw_rad = math.radians(float(pose.get("yaw", 0.0)))
        marker.pose.orientation.z = math.sin(yaw_rad * 0.5)
        marker.pose.orientation.w = math.cos(yaw_rad * 0.5)
        marker.scale.x = self.box_size_x
        marker.scale.y = self.box_size_y
        marker.scale.z = self.box_size_z
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_sphere(self, marker_array, marker_id, namespace, x, y, z, color, scale):
        marker = self._new_marker(marker_id, namespace, Marker.SPHERE)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_arrow(self, marker_array, marker_id, namespace, sx, sy, sz, ex, ey, ez, color):
        marker = self._new_marker(marker_id, namespace, Marker.ARROW)
        marker.scale.x = 0.035
        marker.scale.y = 0.09
        marker.scale.z = 0.09
        marker.points = [
            Point(x=float(sx), y=float(sy), z=float(sz)),
            Point(x=float(ex), y=float(ey), z=float(ez)),
        ]
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

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
        marker.color.r = float(red)
        marker.color.g = float(green)
        marker.color.b = float(blue)
        marker.color.a = float(alpha)

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
            f"[{self.config_label}] ComputeMoveBoxPalletPlaceStrategy start: "
            f"mode={self.strategy_mode}, rows={self.rows}, cols={self.cols}, "
            f"max_layers={self.max_layers}, stack_count_key={self.stack_count_key}, "
            f"nav_key={self.navigation_target_key}, plane_key={self.place_plane_height_key}, "
            f"slot_key={self.slot_pose_key}, expected_box_key={self.expected_box_pose_key}, "
            f"pre_offset={self.pre_place_lateral_offset_m:.3f}, push_distance={self.push_distance_m:.3f}, "
            f"topic={self.visualization_topic or '<disabled>'}"
        )
