"""计算 move_box 码垛目标、导航站位、放箱策略与动作参考点。

本节点是码垛策略版流程里的唯一“码垛规划节点”：

1. 根据固定或黑板动态托盘 polygon / slot_reference_points / stack_count 选择本轮槽位。
   ``near_column_first`` 模式会在每层第一箱时按机器人位置确定接近列，并把
   本层四格顺序冻结到 blackboard，避免机器人移动后槽位顺序反转。
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

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.box_map_polygon import is_map_position_in_polygon
from tree.utils.pallet_place_diagnostics import write_pallet_place_diagnostic
from tree.utils.chassis_navigation import Pose2D
from tree.utils.geometry import (
    transform_base_point_to_global,
    transform_global_point_to_base,
    ypr_to_rotation_matrix,
)
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
        # 可选：由 ``ComputeFoundationPosePalletMapPolygon`` 写入的真实托盘区域。
        # 配置此 key 后，每轮规划都读取最新四边形，槽位方向、导航禁入判断和
        # RViz 托盘轮廓将使用 FP 标定结果，而非 JSON 中的固定区域。
        self.pallet_map_polygon_key = str(params.get("pallet_map_polygon_key", "")).strip()
        self.pallet_pose_key = str(params.get("pallet_pose_key", "")).strip()
        self.use_pallet_pose_z_as_surface = self._to_bool(
            params.get("use_pallet_pose_z_as_surface", False)
        )
        self.pallet_surface_z_offset_m = float(
            params.get("pallet_surface_z_offset_m", 0.0)
        )
        self.slot_reference_points = self._parse_polygon(params.get("slot_reference_points", []))
        self.slot_reference_row_axis = self._parse_vector(params.get("slot_reference_row_axis", []))
        self.slot_yaw_deg = self._optional_float(params.get("slot_yaw_deg", None))
        self.rows = int(params.get("slot_rows", 2))
        self.cols = int(params.get("slot_cols", 2))
        self.max_layers = int(params.get("max_layers", 1))
        # row_major 保持历史顺序：(0,0)->(0,1)->(1,0)->(1,1)。
        # near_column_first：每层先完成更接近机器人所在侧的一整列，再处理远列。
        self.slot_order_mode = str(params.get("slot_order_mode", "row_major")).strip().lower()
        self.layer_plan_key = str(
            params.get("layer_plan_key", "move_box_pallet_stack_layer_plan")
        ).strip()
        self.box_size_x = float(params.get("box_size_x", 0.60))
        self.box_size_y = float(params.get("box_size_y", 0.40))
        self.box_size_z = float(params.get("box_size_z", 0.34))
        self.slot_gap_x = float(params.get("slot_gap_x", 0.04))
        self.slot_gap_y = float(params.get("slot_gap_y", 0.04))
        self.pallet_surface_z = float(params.get("pallet_surface_z", 0.0))
        self.place_clearance_z = float(params.get("place_clearance_z", 0.0))
        self.place_box_forward_offset_m = float(params.get("place_box_forward_offset_m", 0.90))
        self.approach_side = str(params.get("approach_side", "positive_y")).strip().lower()
        # fixed：沿 approach_side 固定一侧站位。
        # row_outside：按 slot row 选择垛盘外侧站位；前排用 -row_axis，后排用 +row_axis。
        # two_sided_feasible：参考 SelectAndPublishHighestYoloBox，围绕目标箱长边生成
        # 两个面对箱子的候选导航点，过滤垛盘内/clearance 不足的点，再按代价选择。
        self.navigation_approach_mode = str(
            params.get("navigation_approach_mode", "fixed")
        ).strip().lower()
        self.navigation_pallet_clearance_m = max(
            0.0,
            float(params.get("navigation_pallet_clearance_m", 0.0)),
        )
        self.navigation_yaw_cost_weight = float(params.get("navigation_yaw_cost_weight", 0.01))

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
        self.base_link_visualization_enabled = self._to_bool(
            params.get("base_link_visualization_enabled", True)
        )

        for key in (self.services_key, self.stack_count_key):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)
        if self.layer_plan_key:
            self.blackboard.register_key(
                key=self.layer_plan_key,
                access=py_trees.common.Access.READ,
            )
            self.blackboard.register_key(
                key=self.layer_plan_key,
                access=py_trees.common.Access.WRITE,
            )
        if self.pallet_map_polygon_key:
            self.blackboard.register_key(
                key=self.pallet_map_polygon_key,
                access=py_trees.common.Access.READ,
            )
        if self.pallet_pose_key:
            self.blackboard.register_key(
                key=self.pallet_pose_key,
                access=py_trees.common.Access.READ,
            )

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

        self.odom_transformer = self.get_odom_pose_transformer(
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
        if not self._refresh_dynamic_pallet_geometry():
            return Status.RUNNING
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

        slot_ordinal = stack_count % slots_per_layer
        layer_slot_order = self._get_layer_slot_order(geometry, layer, stack_count)
        if len(layer_slot_order) != slots_per_layer:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 本层槽位计划数量异常: "
                f"expected={slots_per_layer}, actual={len(layer_slot_order)}, "
                f"plan={layer_slot_order}"
            )
            return Status.FAILURE
        row, col = layer_slot_order[slot_ordinal]
        placed_slots = set(layer_slot_order[:slot_ordinal])
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
        # 码垛策略判断相邻 slot 时必须使用“槽位 col 方向”，而不是箱体 yaw 方向。
        # 例如当前配置 slot_reference_points 为 (0.67,0.77)->(0.03,0.77)，
        # col 增加方向是 map -x；若误用 slot_yaw_deg=0 得到的 map +x，
        # 第二/第四个箱子的左推/右推策略会整体反向。
        slot_col_axis = geometry.get("x_axis", x_axis)
        strategy_info = self._select_strategy(
            row=row,
            col=col,
            slot_col_axis=slot_col_axis,
            box_x_axis=x_axis,
            placed_slots=placed_slots,
            navigation_pose=navigation_pose,
        )
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
            f"slot_order_mode={self.slot_order_mode}, layer_order={layer_slot_order}, "
            f"slot=({slot_pose['x']:.3f}, {slot_pose['y']:.3f}, {slot_pose['yaw']:.2f}deg), "
            f"nav=({navigation_pose['x']:.3f}, {navigation_pose['y']:.3f}, "
            f"{navigation_pose['yaw']:.2f}deg), "
            f"nav_source={navigation_pose.get('approach_source', '<unknown>')}, "
            f"nav_axis=({navigation_pose.get('approach_axis_x', 0.0):.3f},"
            f"{navigation_pose.get('approach_axis_y', 0.0):.3f}), "
            f"slot_col_axis=({slot_col_axis[0]:.3f},{slot_col_axis[1]:.3f}), "
            f"box_x_axis=({x_axis[0]:.3f},{x_axis[1]:.3f}), "
            f"place_plane_z={place_plane_height:.3f}, "
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
        write_pallet_place_diagnostic(
            "strategy_plan",
            {
                "label": self.config_label,
                "stack_count": stack_count,
                "layer": layer,
                "row": row,
                "col": col,
                "strategy": strategy_info["strategy"],
                "slot_pose_map": slot_pose,
                "navigation_pose_map": navigation_pose,
                "place_plane_z_map": place_plane_height,
                "final_box_pose_map": final_box_pose,
                "pre_box_pose_map": pre_box_pose,
                "push_direction_map": push_direction,
                "release_first_side": strategy_info["release_first_side"],
                "push_side": strategy_info["push_side"],
            },
        )
        return Status.SUCCESS

    def _refresh_dynamic_pallet_geometry(self):
        """按需从黑板更新托盘区域与放置平面高度。

        当流程在抓住箱子后以 FP 识别托盘时，FP 节点先把 map 四边形和托盘
        中心位姿写入黑板。本方法在真正规划前读取它们；未就绪时保持 RUNNING，
        避免误回退到旧的静态托盘参数而把箱子放错区域。
        """
        if self.pallet_map_polygon_key:
            if not self.blackboard.exists(self.pallet_map_polygon_key):
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 等待动态托盘区域: "
                    f"key={self.pallet_map_polygon_key}"
                )
                return False
            try:
                polygon = self._parse_polygon(
                    self.blackboard.get(self.pallet_map_polygon_key)
                )
            except (TypeError, ValueError) as exc:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 动态托盘区域解析失败: "
                    f"key={self.pallet_map_polygon_key}, error={exc}"
                )
                return False
            if len(polygon) < 3:
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 等待有效动态托盘区域: "
                    f"key={self.pallet_map_polygon_key}, polygon={polygon}"
                )
                return False
            self.pallet_map_polygon = polygon

        if self.use_pallet_pose_z_as_surface:
            if not self.pallet_pose_key or not self.blackboard.exists(self.pallet_pose_key):
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 等待动态托盘位姿高度: "
                    f"key={self.pallet_pose_key or '<disabled>'}"
                )
                return False
            try:
                pallet_pose = self.blackboard.get(self.pallet_pose_key)
                pallet_z = float(pallet_pose["z"])
            except (KeyError, TypeError, ValueError) as exc:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 动态托盘位姿高度无效: "
                    f"key={self.pallet_pose_key}, error={exc}"
                )
                return False
            self.pallet_surface_z = pallet_z + self.pallet_surface_z_offset_m

        return True

    def _get_layer_slot_order(self, geometry, layer, stack_count):
        """返回本层稳定的槽位顺序。

        ``near_column_first`` 在某一层的第一箱（ordinal=0）读取机器人 map
        位置，比较每一列到机器人的平均距离，先排接近列。计划写入 blackboard
        后，该层的后续箱子即使已导航到别的位置，也不会重新排序或重复选格。
        """
        slots_per_layer = self.rows * self.cols
        ordinal = stack_count % slots_per_layer
        if self.slot_order_mode not in ("near_column_first", "near_column", "robot_near_column"):
            return [(row, col) for row in range(self.rows) for col in range(self.cols)]

        # 第一个槽位总是重新取一次机器人站位，避免上一次任务残留的 layer=0
        # 计划影响新的整层码垛；中途槽位才复用冻结计划。
        if ordinal != 0:
            saved_plan = self._read_saved_layer_plan(layer)
            if saved_plan is not None:
                return saved_plan

        # 非首箱但不存在计划通常意味着外部手动改了 stack_count 或重启后没有恢复
        # blackboard。此时仍生成一次计划并记录告警，避免退回 row-major 后造成重格。
        if ordinal != 0:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 接近侧分列码垛缺少当前层冻结计划，"
                f"将在中途重新生成: layer={layer}, ordinal={ordinal}, "
                f"key={self.layer_plan_key}"
            )

        current_pose = self.odom_transformer.get_current_pose()
        if current_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 无法读取当前底盘 map 位姿，"
                "接近侧分列码垛回退为行优先顺序"
            )
            return [(row, col) for row in range(self.rows) for col in range(self.cols)]

        robot_x, robot_y = float(current_pose[0]), float(current_pose[1])
        slot_poses = {
            (row, col): self._compute_slot_pose(geometry, row, col, layer)
            for row in range(self.rows)
            for col in range(self.cols)
        }
        # 每列先按“列内两个槽位中心的平均距离”确定接近列；再按单格距离确定
        # 该列先放哪个边角。这样机器人在托盘左/右侧时会先填对应整列。
        column_order = sorted(
            range(self.cols),
            key=lambda col: sum(
                (slot_poses[(row, col)]["x"] - robot_x) ** 2
                + (slot_poses[(row, col)]["y"] - robot_y) ** 2
                for row in range(self.rows)
            ) / max(1, self.rows),
        )
        order = []
        for col in column_order:
            row_order = sorted(
                range(self.rows),
                key=lambda row: (
                    (slot_poses[(row, col)]["x"] - robot_x) ** 2
                    + (slot_poses[(row, col)]["y"] - robot_y) ** 2
                ),
            )
            order.extend((row, col) for row in row_order)

        if self.layer_plan_key:
            plan = {
                "layer": int(layer),
                "rows": int(self.rows),
                "cols": int(self.cols),
                "mode": self.slot_order_mode,
                "robot_pose_map": {"x": robot_x, "y": robot_y},
                "slots": [{"row": row, "col": col} for row, col in order],
            }
            self.blackboard.set(self.layer_plan_key, plan, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已冻结本层接近侧分列码垛顺序: "
            f"layer={layer}, robot_map=({robot_x:.3f},{robot_y:.3f}), "
            f"column_order={column_order}, slots={order}, key={self.layer_plan_key}"
        )
        return order

    def _read_saved_layer_plan(self, layer):
        if not self.layer_plan_key or not self.blackboard.exists(self.layer_plan_key):
            return None
        try:
            plan = self.blackboard.get(self.layer_plan_key)
            if not isinstance(plan, dict):
                return None
            if int(plan.get("layer", -1)) != int(layer):
                return None
            if int(plan.get("rows", -1)) != self.rows or int(plan.get("cols", -1)) != self.cols:
                return None
            slots = plan.get("slots", [])
            if not isinstance(slots, list) or len(slots) != self.rows * self.cols:
                return None
            order = [(int(item["row"]), int(item["col"])) for item in slots]
            if len(set(order)) != len(order):
                return None
            if any(row < 0 or row >= self.rows or col < 0 or col >= self.cols for row, col in order):
                return None
            return order
        except (KeyError, TypeError, ValueError):
            return None

    def _select_strategy(self, row, col, slot_col_axis, box_x_axis, placed_slots, navigation_pose):
        if self.strategy_mode in ("direct", "direct_place"):
            return self._strategy("direct_place", "none", "none", "配置强制直接放箱", (0.0, 0.0))
        if self.strategy_mode in ("right_push_left", "right_push_left_place"):
            return self._strategy(
                "right_push_left_place",
                "left",
                "right",
                "配置强制右爪向左推",
                box_x_axis,
            )
        if self.strategy_mode in ("left_push_right", "left_push_right_place"):
            return self._strategy(
                "left_push_right_place",
                "right",
                "left",
                "配置强制左爪向右推",
                tuple(-value for value in box_x_axis),
            )

        # 自动策略根据本层已执行槽位推断相邻箱：同一 row 中，如果相邻 col
        # 已经在本轮之前出现过，就用推箱策略贴靠邻箱。这里不能再使用
        # ``slot_index < stack_count``，因为 near_column_first 的实际执行顺序
        # 与 row-major 槽位编号不同。
        #
        # 左/右不是 map 固定左右，也不是 col-1/col+1 的名字，而是“机器人站在
        # navigation_pose 面向目标 slot 时”的左右。这样第一排和第二排由于导航朝向相反，
        # 同一个 map 侧的邻箱会自动得到相反的左/右关系。
        left_neighbor_slot = (row, col - 1)
        right_neighbor_slot = (row, col + 1)
        left_neighbor_placed = col > 0 and left_neighbor_slot in placed_slots
        right_neighbor_placed = col < self.cols - 1 and right_neighbor_slot in placed_slots
        if left_neighbor_placed:
            return self._strategy_for_neighbor_axis(
                neighbor_axis=(-slot_col_axis[0], -slot_col_axis[1]),
                navigation_pose=navigation_pose,
                reason_prefix=f"col-1 邻箱已放(slot={left_neighbor_slot})",
            )
        if right_neighbor_placed:
            return self._strategy_for_neighbor_axis(
                neighbor_axis=slot_col_axis,
                navigation_pose=navigation_pose,
                reason_prefix=f"col+1 邻箱已放(slot={right_neighbor_slot})",
            )
        return self._strategy("direct_place", "both", "none", "当前槽位无同排邻箱，直接放箱", (0.0, 0.0))

    def _strategy_for_neighbor_axis(self, neighbor_axis, navigation_pose, reason_prefix):
        """根据相邻箱在机器人视角中的左右侧选择推箱方向。

        neighbor_axis 是从目标 slot 指向已放邻箱的 map 水平单位方向。
        push_axis 也使用这个方向：预落位在远离邻箱一侧，最终推向邻箱。
        """
        axis_norm = math.hypot(float(neighbor_axis[0]), float(neighbor_axis[1]))
        if axis_norm <= 1e-9:
            return self._strategy("direct_place", "both", "none", f"{reason_prefix}，邻箱方向无效，回退直接放箱", (0.0, 0.0))

        push_axis = (
            float(neighbor_axis[0]) / axis_norm,
            float(neighbor_axis[1]) / axis_norm,
        )
        # push_axis 表示“预落位 -> 最终位”的推箱方向。策略名也按这个方向命名：
        # - push_axis 在机器人左侧：右爪从右侧向左推，right_push_left_place。
        # - push_axis 在机器人右侧：左爪从左侧向右推，left_push_right_place。
        #
        # 这里优先使用 navigation_pose 中的 approach_axis：它表示从目标 slot 指向
        # 机器人导航站位的方向。机器人站在该点面向 slot 时，机器人左侧方向为
        # (approach_y, -approach_x)。这样不依赖底盘 yaw 的正负约定，避免第二/第四格
        # 由于导航朝向相反而策略翻转错误。
        approach_axis = (
            float(navigation_pose.get("approach_axis_x", 0.0)),
            float(navigation_pose.get("approach_axis_y", 0.0)),
        )
        approach_norm = math.hypot(approach_axis[0], approach_axis[1])
        if approach_norm > 1e-9:
            approach_axis = (
                approach_axis[0] / approach_norm,
                approach_axis[1] / approach_norm,
            )
            robot_left_axis = (approach_axis[1], -approach_axis[0])
            left_axis_source = "approach_axis"
        else:
            yaw_rad = math.radians(float(navigation_pose.get("yaw", 0.0)))
            robot_left_axis = (-math.sin(yaw_rad), math.cos(yaw_rad))
            left_axis_source = "yaw"

        push_left_score = self._dot(push_axis, robot_left_axis)

        if push_left_score > 0.0:
            return self._strategy(
                "right_push_left_place",
                "left",
                "right",
                f"{reason_prefix}，推箱方向在机器人左侧(score={push_left_score:.3f}, source={left_axis_source})，"
                "先右偏预落位，再由右爪向左推",
                push_axis,
            )

        if push_left_score < 0.0:
            return self._strategy(
                "left_push_right_place",
                "right",
                "left",
                f"{reason_prefix}，推箱方向在机器人右侧(score={push_left_score:.3f}, source={left_axis_source})，"
                "先左偏预落位，再由左爪向右推",
                push_axis,
            )

        return self._strategy("direct_place", "both", "none", f"{reason_prefix}，邻箱正对机器人中心，直接放箱", (0.0, 0.0))

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
        selected_candidate, evaluations = self._select_navigation_candidate(geometry, slot_pose)
        approach_axis = selected_candidate["axis"]
        approach_source = selected_candidate["source"]
        nav_x = selected_candidate["x"]
        nav_y = selected_candidate["y"]

        # 安全兜底：如果算出来的导航点落在垛盘 polygon 内，而反方向在外面，就翻转站位。
        # 这可以防止第二排 slot 仍使用第一排站位方向，导致目标点钻进垛盘区域。
        if self._is_in_pallet_polygon(nav_x, nav_y):
            opposite_axis = (-approach_axis[0], -approach_axis[1])
            opposite_nav_x = slot_pose["x"] + opposite_axis[0] * self.place_box_forward_offset_m
            opposite_nav_y = slot_pose["y"] + opposite_axis[1] * self.place_box_forward_offset_m
            if not self._is_in_pallet_polygon(opposite_nav_x, opposite_nav_y):
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 码垛导航点落入垛盘区域，已切换到反向外侧站位: "
                    f"slot=({slot_pose['x']:.3f},{slot_pose['y']:.3f}), "
                    f"old_nav=({nav_x:.3f},{nav_y:.3f}), "
                    f"new_nav=({opposite_nav_x:.3f},{opposite_nav_y:.3f}), "
                    f"source={approach_source}"
                )
                approach_axis = opposite_axis
                approach_source = f"{approach_source}:flipped_outside_pallet"
                nav_x = opposite_nav_x
                nav_y = opposite_nav_y

        yaw = math.degrees(math.atan2(slot_pose["y"] - nav_y, slot_pose["x"] - nav_x))
        self._log_navigation_candidate_result(slot_pose, selected_candidate, evaluations, yaw)
        return {
            "x": float(nav_x),
            "y": float(nav_y),
            "yaw": float(yaw),
            "slot_x": float(slot_pose["x"]),
            "slot_y": float(slot_pose["y"]),
            "place_box_forward_offset_m": float(self.place_box_forward_offset_m),
            "approach_source": approach_source,
            "approach_axis_x": float(approach_axis[0]),
            "approach_axis_y": float(approach_axis[1]),
        }

    def _select_navigation_candidate(self, geometry, slot_pose):
        if self.navigation_approach_mode in (
            "two_sided",
            "two_sided_feasible",
            "long_edge",
            "long_edge_feasible",
        ):
            candidates = self._build_two_sided_navigation_candidates(geometry, slot_pose)
            feasible = [candidate for candidate in candidates if candidate["feasible"]]
            if feasible:
                return min(
                    feasible,
                    key=lambda item: (
                        item.get("outside_distance", 0.0),
                        item["cost"],
                    ),
                ), candidates
            if candidates:
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 两侧码垛导航候选均不可行，回退到代价最低候选: "
                    + "; ".join(
                        f"{item['side']} reason={item['reason']} nav=({item['x']:.3f},{item['y']:.3f})"
                        for item in candidates
                    )
                )
                return min(candidates, key=lambda item: item["cost"]), candidates

        approach_axis, approach_source = self._navigation_approach_axis(geometry, slot_pose)
        nav_x = slot_pose["x"] + approach_axis[0] * self.place_box_forward_offset_m
        nav_y = slot_pose["y"] + approach_axis[1] * self.place_box_forward_offset_m
        feasible, reason = self._evaluate_navigation_geometry(nav_x, nav_y)
        candidate = {
            "x": float(nav_x),
            "y": float(nav_y),
            "axis": approach_axis,
            "side": approach_source,
            "source": approach_source,
            "feasible": feasible,
            "reason": reason,
            "travel": 0.0,
            "yaw_error": 0.0,
            "cost": 0.0,
        }
        return candidate, [candidate]

    def _build_two_sided_navigation_candidates(self, geometry, slot_pose):
        # slot_pose 的 yaw 表示箱体长边方向；两侧候选取长边法向 +/- normal。
        yaw_rad = math.radians(float(slot_pose.get("yaw", geometry.get("yaw", 0.0))))
        long_axis = (math.cos(yaw_rad), math.sin(yaw_rad))
        normal = (-long_axis[1], long_axis[0])
        robot_pose = self.odom_transformer.get_current_pose()
        robot_x = float(robot_pose[0]) if robot_pose is not None else float(slot_pose["x"])
        robot_y = float(robot_pose[1]) if robot_pose is not None else float(slot_pose["y"])
        robot_yaw = float(robot_pose[3]) if robot_pose is not None else 0.0

        candidates = []
        for side_sign, side_name in ((1.0, "positive_long_edge_normal"), (-1.0, "negative_long_edge_normal")):
            axis = (side_sign * normal[0], side_sign * normal[1])
            nav_x = float(slot_pose["x"]) + axis[0] * self.place_box_forward_offset_m
            nav_y = float(slot_pose["y"]) + axis[1] * self.place_box_forward_offset_m
            yaw = math.degrees(math.atan2(float(slot_pose["y"]) - nav_y, float(slot_pose["x"]) - nav_x))
            feasible, reason = self._evaluate_navigation_geometry(nav_x, nav_y)
            outside_distance = self._outside_distance_along_axis(
                (float(slot_pose["x"]), float(slot_pose["y"])),
                axis,
                max_distance=self.place_box_forward_offset_m,
            )
            travel = math.hypot(nav_x - robot_x, nav_y - robot_y)
            yaw_error = abs(self._normalize_angle_deg(yaw - robot_yaw))
            candidates.append({
                "x": float(nav_x),
                "y": float(nav_y),
                "yaw": float(yaw),
                "axis": axis,
                "side": side_name,
                "source": f"two_sided_feasible:{side_name}",
                "feasible": feasible,
                "reason": reason,
                "outside_distance": outside_distance,
                "travel": travel,
                "yaw_error": yaw_error,
                "cost": travel + self.navigation_yaw_cost_weight * yaw_error,
            })
        return candidates

    def _evaluate_navigation_geometry(self, x, y):
        point = (float(x), float(y))
        if self._is_in_pallet_polygon(point[0], point[1]):
            return False, "inside_pallet_polygon"
        if len(self.pallet_map_polygon) >= 3 and self.navigation_pallet_clearance_m > 0.0:
            clearance = self._point_to_polygon_boundary_distance(point, self.pallet_map_polygon)
            if clearance < self.navigation_pallet_clearance_m:
                return (
                    False,
                    "pallet_clearance_too_small({:.2f}m<{:.2f}m)".format(
                        clearance,
                        self.navigation_pallet_clearance_m,
                    ),
                )
        return True, "geometry_feasible"

    def _log_navigation_candidate_result(self, slot_pose, selected_candidate, evaluations, yaw):
        if not evaluations:
            return
        details = "; ".join(
            f"{item['side']}: nav=({item['x']:.3f},{item['y']:.3f}), "
            f"yaw={item.get('yaw', yaw):.1f}, feasible={item['feasible']}, "
            f"reason={item['reason']}, outside_distance={item.get('outside_distance', 0.0):.3f}, "
            f"cost={item['cost']:.3f}"
            for item in evaluations
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 码垛导航候选评估: "
            f"slot=({slot_pose['x']:.3f},{slot_pose['y']:.3f}), "
            f"selected={selected_candidate['side']}, "
            f"selected_nav=({selected_candidate['x']:.3f},{selected_candidate['y']:.3f},{yaw:.1f}deg), "
            f"mode={self.navigation_approach_mode}, candidates=[{details}]"
        )

    def _navigation_approach_axis(self, geometry, slot_pose):
        if self.navigation_approach_mode in (
            "row_outside",
            "by_row_outside",
            "auto_row_outside",
        ):
            return self._row_outside_approach_axis(geometry, slot_pose)
        return self._approach_axis(geometry), f"fixed:{self.approach_side}"

    def _row_outside_approach_axis(self, geometry, slot_pose):
        y_axis = geometry["y_axis"]
        row = int(slot_pose.get("row", 0))
        split = (max(1, self.rows) - 1) * 0.5
        if row < split:
            return (-y_axis[0], -y_axis[1]), f"row_outside:row{row}:negative_row_axis"
        if row > split:
            return y_axis, f"row_outside:row{row}:positive_row_axis"

        # 奇数行中间排没有天然外侧，回退到配置方向。
        return self._approach_axis(geometry), f"row_outside:row{row}:fallback:{self.approach_side}"

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

    def _is_in_pallet_polygon(self, x, y):
        if len(self.pallet_map_polygon) < 3:
            return False
        return is_map_position_in_polygon(
            {"x": float(x), "y": float(y)},
            [{"x": px, "y": py} for px, py in self.pallet_map_polygon],
        )

    def _outside_distance_along_axis(self, start, axis, max_distance):
        """估算从 slot 沿候选方向离开 pallet polygon 的距离，越小表示越靠近这一侧外边。"""
        if len(self.pallet_map_polygon) < 3:
            return 0.0
        axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm <= 1e-9:
            return 0.0
        ux = axis[0] / axis_norm
        uy = axis[1] / axis_norm
        if not self._is_in_pallet_polygon(start[0], start[1]):
            return 0.0

        # 先用垛盘边线精确求 ray 的第一个交点；失败时再用小步长兜底。
        intersections = []
        for edge_start, edge_end in zip(self.pallet_map_polygon, self.pallet_map_polygon[1:] + self.pallet_map_polygon[:1]):
            distance = self._ray_segment_intersection_distance(start, (ux, uy), edge_start, edge_end)
            if distance is not None and distance >= -1e-6:
                intersections.append(max(0.0, distance))
        positive = [value for value in intersections if value > 1e-5]
        if positive:
            return min(min(positive), float(max_distance))

        step = 0.02
        distance = 0.0
        while distance <= max_distance:
            x = start[0] + ux * distance
            y = start[1] + uy * distance
            if not self._is_in_pallet_polygon(x, y):
                return distance
            distance += step
        return float(max_distance)

    @staticmethod
    def _ray_segment_intersection_distance(origin, direction, start, end):
        ox, oy = origin
        dx, dy = direction
        sx, sy = start
        ex, ey = end
        vx = ex - sx
        vy = ey - sy
        # origin + t * direction = start + u * segment
        denom = dx * (-vy) - dy * (-vx)
        if abs(denom) <= 1e-9:
            return None
        bx = sx - ox
        by = sy - oy
        t = (bx * (-vy) - by * (-vx)) / denom
        u = (dx * by - dy * bx) / denom
        if t < -1e-6 or u < -1e-6 or u > 1.0 + 1e-6:
            return None
        return float(t)

    @classmethod
    def _point_to_polygon_boundary_distance(cls, point, polygon):
        if len(polygon) < 2:
            return float("inf")
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        return min(cls._point_segment_distance(point, start, end) for start, end in edges)

    @staticmethod
    def _point_segment_distance(point, start, end):
        px, py = point
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return math.hypot(px - sx, py - sy)
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        closest_x = sx + t * dx
        closest_y = sy + t * dy
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def _normalize_angle_deg(angle_deg):
        while angle_deg >= 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

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

    def _map_point_to_base(self, x, y, z, current_pose):
        current_pose_2d = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        point_base = transform_global_point_to_base(
            current_pose_2d,
            float(x),
            float(y),
        )
        return {
            "x": float(point_base["x"]),
            "y": float(point_base["y"]),
            "z": float(z) - float(current_pose[2]),
        }

    def _map_pose_to_base(self, pose, current_pose):
        if pose is None:
            return None
        try:
            point_base = self._map_point_to_base(
                pose["x"],
                pose["y"],
                pose.get("z", 0.0),
                current_pose,
            )
            point_base["yaw"] = self._normalize_angle_deg(
                float(pose.get("yaw", 0.0)) - float(current_pose[3])
            )
            if "source" in pose:
                point_base["source"] = pose["source"]
            return point_base
        except (KeyError, TypeError, ValueError):
            return None

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
        marker_id += 1

        if self.base_link_visualization_enabled:
            marker_id = self._append_base_link_visualization(
                marker_array=marker_array,
                marker_id=marker_id,
                geometry=geometry,
                selected_row=selected_row,
                selected_col=selected_col,
                layer=layer,
                slot_pose=slot_pose,
                navigation_pose=navigation_pose,
                final_box_pose=final_box_pose,
                pre_box_pose=pre_box_pose,
                held_box_pose=held_box_pose,
                push_direction=push_direction,
                strategy_info=strategy_info,
            )

        self.visualization_pub.publish(marker_array)

    def _append_base_link_visualization(
        self,
        marker_array,
        marker_id,
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
    ):
        """在同一话题中追加 base_link 下的码垛策略可视化。"""
        current_pose = self.odom_transformer.get_current_pose()
        if current_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 码垛策略base_link可视化跳过: 尚未获得底盘位姿"
            )
            return marker_id

        marker_id = self._append_pallet_polygon_base(marker_array, marker_id, current_pose)
        if geometry.get("mode") == "reference_points":
            marker_id = self._append_reference_points_base(marker_array, marker_id, geometry, current_pose)
        marker_id = self._append_all_slots_base(
            marker_array,
            marker_id,
            geometry,
            selected_row,
            selected_col,
            layer,
            current_pose,
        )

        navigation_pose_map = dict(navigation_pose)
        navigation_pose_map["z"] = self.pallet_surface_z + 0.12
        navigation_pose_base = self._map_pose_to_base(navigation_pose_map, current_pose)
        final_box_pose_base = self._map_pose_to_base(final_box_pose, current_pose)
        pre_box_pose_base = self._map_pose_to_base(pre_box_pose, current_pose)
        held_box_pose_base = self._map_pose_to_base(held_box_pose, current_pose)
        slot_pose_base = self._map_pose_to_base(
            {
                "x": slot_pose["x"],
                "y": slot_pose["y"],
                "z": final_box_pose.get("z", 0.0),
                "yaw": final_box_pose.get("yaw", 0.0),
            },
            current_pose,
        )
        if (
            navigation_pose_base is None
            or final_box_pose_base is None
            or pre_box_pose_base is None
            or held_box_pose_base is None
            or slot_pose_base is None
        ):
            return marker_id

        marker_id = self._append_navigation_marker(
            marker_array,
            marker_id,
            navigation_pose_base,
            final_box_pose_base,
            frame_id=BASE_LINK_FRAME,
            namespace_prefix="base_link/",
        )
        marker_id = self._append_box(
            marker_array,
            marker_id,
            "base_link/held_box_estimate",
            held_box_pose_base,
            color=(0.2, 0.8, 1.0, 0.22),
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_box(
            marker_array,
            marker_id,
            "base_link/pre_place_box",
            pre_box_pose_base,
            color=(1.0, 0.75, 0.05, 0.25),
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_box(
            marker_array,
            marker_id,
            "base_link/final_place_box",
            final_box_pose_base,
            color=(0.0, 1.0, 0.25, 0.30),
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "base_link/slot_center",
            slot_pose_base["x"],
            slot_pose_base["y"],
            slot_pose_base.get("z", 0.0),
            color=(0.0, 1.0, 0.25, 0.75),
            scale=0.07,
            frame_id=BASE_LINK_FRAME,
        )

        if push_direction["distance"] > 1e-6:
            start = pre_box_pose_base
            # push_direction 是 map 平面方向，转换为 base 平面方向后再画箭头。
            end_map = {
                "x": pre_box_pose["x"] + push_direction["x"] * push_direction["distance"],
                "y": pre_box_pose["y"] + push_direction["y"] * push_direction["distance"],
                "z": pre_box_pose.get("z", 0.0) + 0.25,
                "yaw": pre_box_pose.get("yaw", 0.0),
            }
            end = self._map_pose_to_base(end_map, current_pose)
            start_arrow_map = dict(pre_box_pose)
            start_arrow_map["z"] = pre_box_pose.get("z", 0.0) + 0.25
            start = self._map_pose_to_base(start_arrow_map, current_pose)
            if start is not None and end is not None:
                marker_id = self._append_arrow(
                    marker_array,
                    marker_id,
                    "base_link/push_direction",
                    start["x"],
                    start["y"],
                    start.get("z", 0.0),
                    end["x"],
                    end["y"],
                    end.get("z", 0.0),
                    color=(1.0, 0.25, 0.0, 0.75),
                    frame_id=BASE_LINK_FRAME,
                )

        text = self._new_marker(
            marker_id,
            "base_link/pallet_place_strategy_text",
            Marker.TEXT_VIEW_FACING,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id += 1
        text.pose.position.x = final_box_pose_base["x"]
        text.pose.position.y = final_box_pose_base["y"]
        text.pose.position.z = final_box_pose_base.get("z", 0.0) + 0.48
        text.pose.orientation.w = 1.0
        text.scale.z = 0.08
        text.text = (
            "PALLET PLACE STRATEGY base_link\n"
            f"strategy={strategy_info['strategy']}\n"
            f"slot=({slot_pose_base['x']:.2f},{slot_pose_base['y']:.2f}) "
            f"nav=({navigation_pose_base['x']:.2f},{navigation_pose_base['y']:.2f})\n"
            f"held=({held_box_pose_base['x']:.2f},{held_box_pose_base['y']:.2f})\n"
            f"pre=({pre_box_pose_base['x']:.2f},{pre_box_pose_base['y']:.2f}) "
            f"final=({final_box_pose_base['x']:.2f},{final_box_pose_base['y']:.2f})\n"
            f"release={strategy_info['release_first_side']} push={strategy_info['push_side']}"
        )
        self._set_color(text, 0.75, 0.95, 1.0, 0.9)
        marker_array.markers.append(text)
        return marker_id

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

    def _append_pallet_polygon_base(self, marker_array, marker_id, current_pose):
        if len(self.pallet_map_polygon) < 3:
            return marker_id
        polygon = self._new_marker(
            marker_id,
            "base_link/pallet_polygon",
            Marker.LINE_STRIP,
            frame_id=BASE_LINK_FRAME,
        )
        polygon.scale.x = 0.024
        for point in self.pallet_map_polygon:
            base_point = self._map_point_to_base(
                point[0],
                point[1],
                self.pallet_surface_z + 0.02,
                current_pose,
            )
            if base_point is not None:
                polygon.points.append(Point(**base_point))
        if polygon.points:
            polygon.points.append(polygon.points[0])
        self._set_color(polygon, 1.0, 0.45, 0.0, 0.65)
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

    def _append_reference_points_base(self, marker_array, marker_id, geometry, current_pose):
        for ref_index, ref_point in enumerate((geometry["reference_origin"], geometry["reference_second"])):
            base_point = self._map_point_to_base(
                ref_point[0],
                ref_point[1],
                self.pallet_surface_z + 0.08,
                current_pose,
            )
            if base_point is None:
                continue
            ref_marker = self._new_marker(
                marker_id,
                "base_link/pallet_slot_reference_points",
                Marker.SPHERE,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            ref_marker.pose.position = Point(
                x=base_point["x"],
                y=base_point["y"],
                z=base_point["z"],
            )
            ref_marker.scale.x = ref_marker.scale.y = ref_marker.scale.z = 0.09
            self._set_color(ref_marker, 1.0, 0.0, 1.0, 0.75)
            marker_array.markers.append(ref_marker)

            ref_text = self._new_marker(
                marker_id,
                "base_link/pallet_slot_reference_text",
                Marker.TEXT_VIEW_FACING,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            ref_text.pose.position.x = base_point["x"]
            ref_text.pose.position.y = base_point["y"]
            ref_text.pose.position.z = base_point["z"] + 0.14
            ref_text.scale.z = 0.065
            ref_text.text = f"STACK REF{ref_index} base\n({base_point['x']:.2f},{base_point['y']:.2f})"
            self._set_color(ref_text, 1.0, 0.0, 1.0, 0.75)
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

    def _append_all_slots_base(self, marker_array, marker_id, geometry, selected_row, selected_col, layer, current_pose):
        for row in range(self.rows):
            for col in range(self.cols):
                pose = self._compute_slot_pose(geometry, row, col, layer)
                pose["z"] = self.pallet_surface_z + layer * self.box_size_z + self.box_size_z * 0.5
                pose_base = self._map_pose_to_base(pose, current_pose)
                if pose_base is None:
                    continue
                selected = row == selected_row and col == selected_col
                marker = self._new_marker(
                    marker_id,
                    "base_link/pallet_slots",
                    Marker.CUBE,
                    frame_id=BASE_LINK_FRAME,
                )
                marker_id += 1
                marker.pose.position.x = pose_base["x"]
                marker.pose.position.y = pose_base["y"]
                marker.pose.position.z = pose_base["z"]
                yaw_rad = math.radians(pose_base["yaw"])
                marker.pose.orientation.z = math.sin(yaw_rad * 0.5)
                marker.pose.orientation.w = math.cos(yaw_rad * 0.5)
                marker.scale.x = self.box_size_x
                marker.scale.y = self.box_size_y
                marker.scale.z = self.box_size_z
                if selected:
                    self._set_color(marker, 0.0, 1.0, 0.25, 0.20)
                else:
                    self._set_color(marker, 0.55, 0.55, 0.55, 0.12)
                marker_array.markers.append(marker)
        return marker_id

    def _append_navigation_marker(
        self,
        marker_array,
        marker_id,
        navigation_pose,
        final_box_pose,
        frame_id=MAP_FRAME,
        namespace_prefix="",
    ):
        nav_arrow = self._new_marker(
            marker_id,
            f"{namespace_prefix}pallet_stack_navigation",
            Marker.ARROW,
            frame_id=frame_id,
        )
        marker_id += 1
        nav_arrow.pose.position.x = navigation_pose["x"]
        nav_arrow.pose.position.y = navigation_pose["y"]
        nav_arrow.pose.position.z = navigation_pose.get("z", self.pallet_surface_z + 0.12)
        yaw_rad = math.radians(navigation_pose["yaw"])
        nav_arrow.pose.orientation.z = math.sin(yaw_rad * 0.5)
        nav_arrow.pose.orientation.w = math.cos(yaw_rad * 0.5)
        nav_arrow.scale.x = 0.7
        nav_arrow.scale.y = 0.12
        nav_arrow.scale.z = 0.12
        self._set_color(nav_arrow, 0.2, 0.9, 1.0, 1.0)
        marker_array.markers.append(nav_arrow)

        line = self._new_marker(
            marker_id,
            f"{namespace_prefix}pallet_stack_relation",
            Marker.LINE_LIST,
            frame_id=frame_id,
        )
        marker_id += 1
        line.scale.x = 0.025
        line.points = [
            Point(
                x=navigation_pose["x"],
                y=navigation_pose["y"],
                z=navigation_pose.get("z", self.pallet_surface_z + 0.08),
            ),
            Point(x=final_box_pose["x"], y=final_box_pose["y"], z=final_box_pose.get("z", 0.0)),
        ]
        self._set_color(line, 0.2, 0.9, 1.0, 0.95)
        marker_array.markers.append(line)
        return marker_id

    def _append_box(self, marker_array, marker_id, namespace, pose, color, frame_id=MAP_FRAME):
        marker = self._new_marker(marker_id, namespace, Marker.CUBE, frame_id=frame_id)
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

    def _append_sphere(
        self,
        marker_array,
        marker_id,
        namespace,
        x,
        y,
        z,
        color,
        scale,
        frame_id=MAP_FRAME,
    ):
        marker = self._new_marker(marker_id, namespace, Marker.SPHERE, frame_id=frame_id)
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_arrow(
        self,
        marker_array,
        marker_id,
        namespace,
        sx,
        sy,
        sz,
        ex,
        ey,
        ez,
        color,
        frame_id=MAP_FRAME,
    ):
        marker = self._new_marker(marker_id, namespace, Marker.ARROW, frame_id=frame_id)
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

    def _new_marker(self, marker_id, namespace, marker_type, frame_id=MAP_FRAME):
        marker = Marker()
        marker.header.frame_id = frame_id
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
            if isinstance(point, dict):
                if "x" not in point or "y" not in point:
                    continue
                parsed.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
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
            f"dynamic_pallet_polygon_key={self.pallet_map_polygon_key or '<static>'}, "
            f"dynamic_surface={self.use_pallet_pose_z_as_surface}, "
            f"nav_key={self.navigation_target_key}, plane_key={self.place_plane_height_key}, "
            f"slot_key={self.slot_pose_key}, expected_box_key={self.expected_box_pose_key}, "
            f"pre_offset={self.pre_place_lateral_offset_m:.3f}, push_distance={self.push_distance_m:.3f}, "
            f"topic={self.visualization_topic or '<disabled>'}"
        )
