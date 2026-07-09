"""从一帧 YOLO 多箱结果中锁定最高层目标箱，并发布其 map 位姿。"""

import math
import threading
import time

import py_trees
import tf
import tf.transformations as tf_trans
from geometry_msgs.msg import Point
from kuavo_humanoid_sdk.common.yolo_boxes import (
    parse_yolo_boxes_string,
    serialize_yolo_box,
    yolo_box_center_point,
)
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME
from tree.utils.box_map_polygon import is_map_position_in_polygon, parse_map_polygon

from ..base import TimedMockAction


class SelectAndPublishHighestYoloBox(TimedMockAction):
    """从 YOLO 多目标中选择一个箱子，并给后续导航和抓取决策提供结果。

    选择分两级进行：

    1. 使用 ``map z`` 找到最高层。相比相机坐标，map 高度不会随头部姿态变化。
       ``top_height_tolerance`` 用来把存在少量视觉高度误差的箱子归入同一层。
    2. 只在最高层候选中按 ``same_level_selection`` 二次选择：
       - ``nearest``：选择到机器人平面距离最小的箱子；
       - ``leftmost``：选择机器人视角最左侧的箱子，即 base_link y 最大；
       - ``rightmost``：选择机器人视角最右侧的箱子，即 base_link y 最小。

    输出有两份：

    - ROS ``PoseStamped``：map 坐标目标，供上位机/FoundationPose链路使用；
    - blackboard目标点和抓取策略，供粗导航及后续Selector使用。
    """

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.yolo_topic = str(
            params.get("yolo_topic", "/yolo/target_boxes3d_string")
        ).strip()
        self.output_topic = str(params.get("output_topic", "/move_box/yolo_box_pose_map")).strip()
        self.map_frame = str(params.get("map_frame", MAP_FRAME)).strip()
        self.distance_frame = str(params.get("distance_frame", BASE_LINK_FRAME)).strip()
        self.base_frame = str(params.get("base_frame", BASE_LINK_FRAME)).strip()
        self.chassis_frame = str(params.get("chassis_frame", "melon_odom")).strip()
        self.source_frame_fallback = str(params.get("source_frame_fallback", "camera")).strip()
        self.tf_timeout = float(params.get("tf_timeout", 0.5))
        # 选中箱子的 map 点，格式为 [x, y, z]。粗导航直接读取该 key，
        # 防止粗导航阶段再次从原始 YOLO 中选择另一个箱子。
        self.selected_point_key = str(
            params.get("selected_point_key", "move_box_selected_highest_yolo_map_point")
        ).strip()
        self.selected_box_key = str(
            params.get("selected_box_key", "move_box_selected_highest_yolo_box")
        ).strip()
        # 根据目标箱同层左右邻箱占用情况生成抓取策略，供行为树Selector分支。
        self.grasp_strategy_key = str(
            params.get("grasp_strategy_key", "move_box_grasp_strategy")
        ).strip()
        # 可选的通用拆垛接近规划。启用后，不再直接取最高层最近箱，而是给每个
        # 最高层箱体的两条长边各生成一个站位，并先做垛盘/邻箱通道几何过滤。
        self.approach_pose_planning_enabled = self._to_bool(
            params.get("approach_pose_planning_enabled", False)
        )
        self.approach_pose_key = str(
            params.get("approach_pose_key", "move_box_selected_approach_pose")
        ).strip()
        self.approach_distance_m = float(params.get("approach_distance_m", 1.2))
        self.approach_yaw_cost_weight = float(
            params.get("approach_yaw_cost_weight", 0.01)
        )
        # pallet_map_polygon 与 valid_box_map_polygon 语义不同：前者是底盘不可
        # 进入的实体垛盘区域，后者是允许参与选箱的视觉工作区。
        self.pallet_map_polygon = parse_map_polygon(
            params.get("pallet_map_polygon", [])
        )
        # 候选站位即使不落在垛盘内，也需要和垛盘边界保持最小安全间隙，
        # 否则机器人底盘 footprint 仍可能压到垛盘或在局部规划中不可达。
        self.approach_pallet_clearance_m = float(
            params.get("approach_pallet_clearance_m", 0.60)
        )
        if self.approach_pallet_clearance_m < 0.0:
            raise ValueError("approach_pallet_clearance_m cannot be negative")
        # 同层箱之间使用 map 平面的箱心绝对距离判断是否相邻。该距离与机器人
        # 当前朝向无关；下限用于排除同一箱子的重复检测，上限用于排除远处箱子。
        self.neighbor_center_min_distance = float(
            params.get("neighbor_center_min_distance", 0.20)
        )
        self.neighbor_center_max_distance = float(
            params.get("neighbor_center_max_distance", 1.0)
        )
        if self.neighbor_center_min_distance > self.neighbor_center_max_distance:
            raise ValueError(
                "neighbor_center_min_distance cannot exceed "
                "neighbor_center_max_distance"
            )
        # 新版YOLO同时提供 quat/size。有效时使用3D有向包围盒判断层级和邻接；
        # 数据异常时仍回退到旧的箱心距离逻辑，保证现场兼容性。
        self.use_box_geometry_for_strategy = self._to_bool(
            params.get("use_box_geometry_for_strategy", True)
        )
        self.same_level_vertical_overlap_ratio = float(
            params.get("same_level_vertical_overlap_ratio", 0.50)
        )
        self.same_level_center_height_tolerance = float(
            params.get("same_level_center_height_tolerance", 0.15)
        )
        self.neighbor_surface_max_gap = float(
            params.get("neighbor_surface_max_gap", 0.20)
        )
        self.minimum_valid_box_size = float(
            params.get("minimum_valid_box_size", 0.03)
        )
        # YOLO可能在同一帧中对同一个箱体给出多个3D中心。按机器人距离从近到远
        # 保留代表点，map三维距离小于该阈值的后续检测视为重复目标。
        self.duplicate_3d_distance_threshold = float(
            params.get("duplicate_3d_distance_threshold", 0.45)
        )
        if self.duplicate_3d_distance_threshold < 0.0:
            raise ValueError("duplicate_3d_distance_threshold cannot be negative")
        # 最高箱高度减去该容差后仍在范围内的目标，都视为同一最高层。
        # 当前箱高约0.30m，默认0.10m可容纳视觉高度抖动，同时小于层间高度。
        self.top_height_tolerance = float(params.get("top_height_tolerance", 0.10))
        self.same_level_selection = str(
            params.get("same_level_selection", "nearest")
        ).strip().lower()
        if self.same_level_selection not in ("nearest", "leftmost", "rightmost"):
            raise ValueError(
                "same_level_selection must be 'nearest', 'leftmost' or 'rightmost', got "
                f"{self.same_level_selection!r}"
            )
        self.min_map_height = self._optional_float(params.get("min_map_height", ""))
        self.max_planar_distance = self._optional_float(params.get("max_planar_distance", ""))
        # 可选的map平面抓箱区域。配置后，区域外的YOLO目标不会参与高度筛选、
        # 去重、抓取策略判断或后续导航。
        self.valid_box_map_polygon = parse_map_polygon(
            params.get("valid_box_map_polygon", [])
        )
        self.valid_box_polygon_required = self._to_bool(
            params.get("valid_box_polygon_required", False)
        )
        if self.valid_box_polygon_required and not self.valid_box_map_polygon:
            raise ValueError("valid_box_polygon_required=True 时必须配置 valid_box_map_polygon")
        self.no_target_log_interval_sec = float(params.get("no_target_log_interval_sec", 1.0))
        self.enable_colored_log = self._to_bool(params.get("enable_colored_log", True))
        self.visualization_enabled = self._to_bool(
            params.get("visualization_enabled", True)
        )
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/yolo_box_markers")
        ).strip()
        self.raw_visualization_topic = str(
            params.get("raw_visualization_topic", "/move_box/yolo_raw_box_markers")
        ).strip()
        self.require_new_frame_after_initialise = self._to_bool(
            params.get("require_new_frame_after_initialise", False)
        )
        self.freeze_after_first_valid_frame = self._to_bool(
            params.get("freeze_after_first_valid_frame", False)
        )
        self.latest_boxes = None
        self._message_generation = 0
        self._minimum_generation = 0
        self._frozen_boxes = None
        self._frozen_generation = 0
        self.lock = threading.Lock()
        self._last_no_target_log_time = 0.0
        self.tf_listener = tf.TransformListener()
        self.subscriber = self.ros_node.create_string_subscription(
            self.yolo_topic, self._on_yolo_boxes_string, queue_size=1
        )
        self.publisher = self.ros_node.create_string_publisher(
            self.output_topic, queue_size=1, latch=True
        )
        self.visualization_publisher = None
        if self.visualization_enabled and self.visualization_topic:
            self.visualization_publisher = self.ros_node.create_publisher(
                self.visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.raw_visualization_publisher = None
        if self.visualization_enabled and self.raw_visualization_topic:
            self.raw_visualization_publisher = self.ros_node.create_publisher(
                self.raw_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.blackboard.register_key(key=self.selected_point_key, access=py_trees.common.Access.WRITE)
        if self.selected_box_key:
            self.blackboard.register_key(
                key=self.selected_box_key, access=py_trees.common.Access.WRITE
            )
        if self.grasp_strategy_key:
            self.blackboard.register_key(
                key=self.grasp_strategy_key,
                access=py_trees.common.Access.WRITE,
            )
        if self.approach_pose_key:
            self.blackboard.register_key(
                key=self.approach_pose_key,
                access=py_trees.common.Access.WRITE,
            )

    @staticmethod
    def _optional_float(value):
        if value is None or str(value).strip() == "":
            return None
        return float(value)

    def initialise(self):
        super().initialise()
        self._last_no_target_log_time = 0.0
        self._clear_box_visualization()
        self._clear_raw_box_visualization()
        self._frozen_boxes = None
        self._frozen_generation = 0
        if self.require_new_frame_after_initialise:
            with self.lock:
                self._minimum_generation = self._message_generation + 1

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        boxes = self._get_latest_boxes()
        if not boxes:
            self._log_no_target(
                f"[{self.config_label}] 等待非空 YOLO boxes String: topic={self.yolo_topic}"
            )
            return Status.RUNNING

        source_frame = boxes[0].get("frame_id") or self.source_frame_fallback
        # 与头部盯箱链路保持一致，避免直接查询不稳定/不连通的 camera -> map：
        # T_map_camera = T_map_melon_odom * T_base_link_camera。
        map_from_source = self._build_split_map_transform(
            source_frame,
        )
        distance_from_source = self._lookup_transform_matrix(
            self.distance_frame, source_frame
        )
        if map_from_source is None or distance_from_source is None:
            return Status.RUNNING

        # 阶段1：把每个 YOLO 点同时转换到两套坐标系。
        # - map：只用于比较绝对高度，以及最终发布导航/视觉目标；
        # - distance_frame（通常为 base_link）：用于计算机器人视角的远近和左右。
        # 不能直接用 map x/y 判断左右，因为机器人转向后 map 轴不再等于机器人左右轴。
        raw_candidates = []
        candidates = []
        filtered_candidates = []
        for index, box in enumerate(boxes):
            center_point = yolo_box_center_point(box)
            map_xyz = self._matrix_dot_point(map_from_source, center_point)
            distance_xyz = self._matrix_dot_point(distance_from_source, center_point)
            geometry = self._transform_box_geometry(map_from_source, box, map_xyz)
            planar_distance = math.hypot(distance_xyz[0], distance_xyz[1])
            # 可选的任务区域过滤：高度过低或离机器人过远的目标不参与后续排序。
            candidate = {
                "index": index,
                "map": map_xyz,
                "selection_frame": distance_xyz,
                "distance": planar_distance,
                "box": box,
                "geometry": geometry,
            }
            raw_candidates.append(dict(candidate))
            if not is_map_position_in_polygon(
                {"x": map_xyz[0], "y": map_xyz[1]},
                self.valid_box_map_polygon,
            ):
                candidate["filter_reason"] = "outside_valid_box_map_polygon"
                filtered_candidates.append(candidate)
                continue
            if self.min_map_height is not None and map_xyz[2] < self.min_map_height:
                candidate["filter_reason"] = (
                    f"map_z_below_min({map_xyz[2]:.3f}<{self.min_map_height:.3f})"
                )
                filtered_candidates.append(candidate)
                continue
            if self.max_planar_distance is not None and planar_distance > self.max_planar_distance:
                candidate["filter_reason"] = (
                    f"distance_above_max({planar_distance:.3f}>"
                    f"{self.max_planar_distance:.3f})"
                )
                filtered_candidates.append(candidate)
                continue
            candidates.append(candidate)

        self._publish_raw_box_visualization(raw_candidates)
        self._log_filtered_candidates(filtered_candidates)

        if not candidates:
            self._publish_filtered_box_visualization(filtered_candidates)
            self._log_no_target(
                f"[{self.config_label}] YOLO目标均未通过区域/高度/距离过滤: "
                f"过滤数量={len(filtered_candidates)}"
            )
            return Status.RUNNING

        # 在高度分层和邻箱判断之前先去重，否则同一箱体的重复点可能被误判为
        # 目标左右两侧各有邻箱，最终错误地产生 no_safe_strategy。
        raw_candidate_count = len(candidates)
        candidates, duplicate_records = self._deduplicate_candidates(candidates)
        if duplicate_records:
            duplicate_text = ", ".join(
                f"#{removed_index}->#{kept_index}({distance:.3f}m)"
                for removed_index, kept_index, distance in duplicate_records
            )
            self.ros_node.get_logger().info(
                f"[{self.config_label}] YOLO单帧3D去重: 原始={raw_candidate_count}, "
                f"保留={len(candidates)}, 合并={duplicate_text}"
            )

        # 阶段2：先找最高 z，再用容差形成“最高层候选集合”。
        # 例如最高 z=0.62m、容差=0.06m，则 z>=0.56m 都属于最高层。
        # 这样可避免同一排箱子因检测抖动几厘米而被误判为上下两层。
        # 优先使用OBB真实顶部高度。只有尺寸/姿态无效时才退回箱心z。
        max_height = max(self._candidate_top_height(candidate) for candidate in candidates)
        highest_candidate = max(candidates, key=self._candidate_top_height)
        top_candidates = [
            candidate for candidate in candidates
            if self._is_same_level(highest_candidate, candidate, max_height)
        ]

        # 阶段3：先按既有最高层规则唯一选箱。approach规划只围绕这个目标箱生成
        # 两个长边外侧站位，不再跨多个箱体重选“更优箱”。
        if self.same_level_selection == "leftmost":
            # ROS 机器人坐标约定：x 向前、y 向左。先取 y 最大者；若 y
            # 相同，则距离更近者优先，保证排序结果稳定。
            selected = max(
                top_candidates,
                key=lambda candidate: (
                    candidate["selection_frame"][1],
                    -candidate["distance"],
                ),
            )
        elif self.same_level_selection == "rightmost":
            # ROS 机器人坐标约定：y 的负方向为机器人右侧。
            selected = min(
                top_candidates,
                key=lambda candidate: (
                    candidate["selection_frame"][1],
                    candidate["distance"],
                ),
            )
        else:
            selected = min(top_candidates, key=lambda candidate: candidate["distance"])

        # 阶段4：若启用approach规划，只针对已选中的目标箱生成两个长边候选站位。
        # 站位是否有效只看是否落入垛盘禁入区域；不再用其他箱子否决站位。
        approach_pose = None
        approach_evaluations = []
        if self.approach_pose_planning_enabled:
            map_from_distance = tf_trans.concatenate_matrices(
                map_from_source,
                tf_trans.inverse_matrix(distance_from_source),
            )
            robot_map_pose = [
                float(map_from_distance[0, 3]),
                float(map_from_distance[1, 3]),
                math.degrees(tf_trans.euler_from_matrix(map_from_distance)[2]),
            ]
            approach_pose, approach_evaluations = self._select_approach_for_selected_box(
                selected,
                robot_map_pose,
            )
            if approach_pose is None:
                self._publish_approach_only_visualization(
                    candidates, top_candidates, approach_evaluations
                )
                self._log_no_target(
                    f"[{self.config_label}] 已选中目标箱但其两个接近位姿均落入垛盘禁入区域"
                )
                return Status.RUNNING

        # 阶段5：发布并保存唯一选中的箱子。ROS话题用于跨机器通信，
        # blackboard用于同一行为树内部传递，二者表达的是同一个map目标点。
        selected_box = dict(selected["box"])
        selected_box["frame_id"] = self.map_frame
        selected_box["stamp"] = self._ros_stamp_to_seconds(self.ros_node.now())
        selected_box["center"] = list(selected["map"])
        # quat与center使用同一map变换，后续节点可使用真实有向包围盒。
        if selected["geometry"] is not None:
            selected_box["quat"] = list(selected["geometry"]["map_quat"])
        else:
            selected_box["quat"] = [0.0, 0.0, 0.0, 1.0]
        self.publisher.publish(
            serialize_yolo_box(
                selected_box,
                frame_id=self.map_frame,
                stamp=selected_box["stamp"],
            )
        )
        self.blackboard.set(self.selected_point_key, list(selected["map"]), overwrite=True)
        if approach_pose is not None and self.approach_pose_key:
            self.blackboard.set(
                self.approach_pose_key,
                dict(approach_pose),
                overwrite=True,
            )
        if self.selected_box_key:
            self.blackboard.set(self.selected_box_key, selected_box, overwrite=True)
        # 阶段6：根据目标箱同层邻箱的占用情况决定抓取方式。是否相邻使用
        # map 平面箱心绝对距离，因此不会因机器人斜对箱堆而产生投影误判。
        grasp_strategy, left_neighbors, right_neighbors = self._decide_grasp_strategy(
            selected,
            top_candidates,
            approach_pose=approach_pose,
        )
        if self.grasp_strategy_key:
            self.blackboard.set(
                self.grasp_strategy_key,
                grasp_strategy,
                overwrite=True,
            )
        self._publish_box_visualization(
            candidates,
            top_candidates,
            selected,
            grasp_strategy,
            left_neighbors,
            right_neighbors,
            filtered_candidates,
            approach_evaluations,
        )

        candidate_text = ", ".join(
            "#{} map=({:.3f},{:.3f},{:.3f}) {}_y={:.3f} distance={:.3f}".format(
                candidate["index"], *candidate["map"], self.distance_frame,
                candidate["selection_frame"][1], candidate["distance"]
            )
            for candidate in candidates
        )
        self.ros_node.get_logger().info(f"[{self.config_label}] YOLO候选: {candidate_text}")
        if self.valid_box_map_polygon:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] map抓箱区域过滤: "
                f"区域外={sum(candidate.get('filter_reason') == 'outside_valid_box_map_polygon' for candidate in filtered_candidates)}, "
                f"区域内={raw_candidate_count}"
            )
        selected_message = (
            f"[{self.config_label}] 已锁定最高层目标箱并发布: "
            f"strategy={self.same_level_selection}, index={selected['index']}, "
            f"top_z={max_height:.3f}, selected=({selected['map'][0]:.3f}, "
            f"{selected['map'][1]:.3f}, {selected['map'][2]:.3f}), "
            f"geometry={'obb' if selected['geometry'] is not None else 'center_fallback'}, "
            f"left_neighbors={left_neighbors}, right_neighbors={right_neighbors}, "
            f"grasp_strategy={grasp_strategy}, "
            f"approach_pose={approach_pose}, "
            f"strategy_key={self.grasp_strategy_key}, topic={self.output_topic}"
        )
        self.ros_node.get_logger().info(
            self._color_text(selected_message, "highlight")
        )
        return Status.SUCCESS

    def _publish_box_visualization(
        self,
        candidates,
        top_candidates,
        selected,
        grasp_strategy,
        left_neighbors,
        right_neighbors,
        filtered_candidates=None,
        approach_evaluations=None,
    ):
        """发布所有有效YOLO箱、选中箱和相对关系到RViz。"""
        if self.visualization_publisher is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        top_indices = {candidate["index"] for candidate in top_candidates}
        selected_index = selected["index"]
        marker_id = 1
        for candidate in candidates:
            index = candidate["index"]
            is_selected = index == selected_index
            is_top_level = index in top_indices
            color = self._visualization_color(is_selected, is_top_level)
            corners = self._visualization_corners(candidate)

            outline = self._new_marker(marker_id, "yolo_box_outline", Marker.LINE_LIST)
            marker_id += 1
            outline.scale.x = 0.035 if is_selected else 0.018
            self._set_marker_color(outline, color, 1.0)
            for start_index, end_index in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_index]))
                outline.points.append(self._point_message(corners[end_index]))
            marker_array.markers.append(outline)

            center_marker = self._new_marker(marker_id, "yolo_box_center", Marker.SPHERE)
            marker_id += 1
            center_marker.pose.position = self._point_message(candidate["map"])
            center_marker.pose.orientation.w = 1.0
            center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = (
                0.10 if is_selected else 0.065
            )
            self._set_marker_color(center_marker, color, 1.0)
            marker_array.markers.append(center_marker)

            text_marker = self._new_marker(marker_id, "yolo_box_text", Marker.TEXT_VIEW_FACING)
            marker_id += 1
            text_marker.pose.position.x = candidate["map"][0]
            text_marker.pose.position.y = candidate["map"][1]
            text_marker.pose.position.z = self._candidate_top_height(candidate) + 0.12
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.11 if is_selected else 0.085
            self._set_marker_color(text_marker, color, 1.0)
            level_text = "TOP" if is_top_level else "LOWER"
            selected_text = f" SELECTED {grasp_strategy}" if is_selected else ""
            text_marker.text = (
                f"#{index} {level_text}{selected_text}\n"
                f"map=({candidate['map'][0]:.2f}, {candidate['map'][1]:.2f}, "
                f"{candidate['map'][2]:.2f})"
            )
            marker_array.markers.append(text_marker)

            if not is_selected:
                relation = self._new_marker(marker_id, "yolo_box_relation", Marker.LINE_LIST)
                marker_id += 1
                relation.scale.x = 0.012
                relation.points = [
                    self._point_message(selected["map"]),
                    self._point_message(candidate["map"]),
                ]
                if index in left_neighbors:
                    relation_color = (1.0, 0.1, 1.0)
                elif index in right_neighbors:
                    relation_color = (1.0, 0.35, 0.05)
                elif is_top_level:
                    relation_color = (0.1, 1.0, 1.0)
                else:
                    relation_color = (0.35, 0.45, 0.65)
                self._set_marker_color(relation, relation_color, 0.9)
                marker_array.markers.append(relation)

        marker_id = self._append_filtered_markers(
            marker_array,
            filtered_candidates or [],
            marker_id,
        )
        marker_id = self._append_pallet_marker(marker_array, marker_id)
        self._append_approach_markers(
            marker_array,
            approach_evaluations or [],
            marker_id,
        )

        self.visualization_publisher.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布RViz箱体关系: "
            f"topic={self.visualization_topic}, boxes={len(candidates)}, "
            f"filtered={len(filtered_candidates or [])}, "
            f"approaches={len(approach_evaluations or [])}, "
            f"selected=#{selected_index}, strategy={grasp_strategy}"
        )

    def _publish_approach_only_visualization(
        self, candidates, top_candidates, approach_evaluations
    ):
        """没有几何可行站位时，仍在RViz显示箱体与所有被拒绝的接近点。"""
        if self.visualization_publisher is None:
            return
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        marker_id = 1
        top_indices = {candidate["index"] for candidate in top_candidates}
        for candidate in candidates:
            corners = self._visualization_corners(candidate)
            outline = self._new_marker(marker_id, "yolo_box_outline", Marker.LINE_LIST)
            marker_id += 1
            outline.scale.x = 0.018
            self._set_marker_color(
                outline,
                (0.0, 0.85, 1.0) if candidate["index"] in top_indices else (0.25, 0.45, 0.85),
                1.0,
            )
            for start_index, end_index in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_index]))
                outline.points.append(self._point_message(corners[end_index]))
            marker_array.markers.append(outline)
        marker_id = self._append_pallet_marker(marker_array, marker_id)
        self._append_approach_markers(marker_array, approach_evaluations, marker_id)
        self.visualization_publisher.publish(marker_array)

    def _append_pallet_marker(self, marker_array, marker_id):
        """显示底盘不可进入的垛盘边界，便于核对站位过滤是否符合现场。"""
        if not self.pallet_map_polygon:
            return marker_id
        marker = self._new_marker(marker_id, "pallet_forbidden_polygon", Marker.LINE_STRIP)
        marker_id += 1
        marker.scale.x = 0.045
        self._set_marker_color(marker, (1.0, 0.45, 0.0), 1.0)
        for point in self.pallet_map_polygon + self.pallet_map_polygon[:1]:
            marker.points.append(self._point_message((point["x"], point["y"], 0.03)))
        marker_array.markers.append(marker)
        return marker_id

    def _append_approach_markers(self, marker_array, evaluations, marker_id):
        """绘制候选站位箭头；绿色可行、红色拒绝、紫色为最终选择。"""
        for evaluation in evaluations:
            pose = evaluation["pose"]
            feasible = evaluation["feasible"]
            selected = evaluation.get("selected", False)
            color = (0.8, 0.1, 1.0) if selected else (
                (0.1, 1.0, 0.2) if feasible else (1.0, 0.1, 0.1)
            )
            arrow = self._new_marker(marker_id, "box_approach_pose", Marker.ARROW)
            marker_id += 1
            arrow.pose.position.x = pose["x"]
            arrow.pose.position.y = pose["y"]
            arrow.pose.position.z = pose.get("z", 0.0) + 0.06
            yaw_rad = math.radians(pose["yaw"])
            arrow.pose.orientation.z = math.sin(yaw_rad * 0.5)
            arrow.pose.orientation.w = math.cos(yaw_rad * 0.5)
            arrow.scale.x = 0.45
            arrow.scale.y = 0.07
            arrow.scale.z = 0.07
            self._set_marker_color(arrow, color, 0.95)
            marker_array.markers.append(arrow)

            text_marker = self._new_marker(
                marker_id, "box_approach_text", Marker.TEXT_VIEW_FACING
            )
            marker_id += 1
            text_marker.pose.position.x = pose["x"]
            text_marker.pose.position.y = pose["y"]
            text_marker.pose.position.z = pose.get("z", 0.0) + 0.35
            text_marker.scale.z = 0.09
            self._set_marker_color(text_marker, color, 1.0)
            status = "SELECTED" if selected else ("FEASIBLE" if feasible else "REJECTED")
            text_marker.text = (
                f"box#{evaluation['box_index']} side={evaluation['side']} {status}\n"
                f"{evaluation.get('reason', '')}\n"
                f"({pose['x']:.2f}, {pose['y']:.2f}, {pose['yaw']:.1f}deg)"
            )
            marker_array.markers.append(text_marker)
        return marker_id

    def _publish_filtered_box_visualization(self, filtered_candidates):
        """没有有效候选时，也将被过滤箱体及原因保留在RViz中。"""
        if self.visualization_publisher is None:
            return
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self._append_filtered_markers(marker_array, filtered_candidates, 1)
        self.visualization_publisher.publish(marker_array)

    def _append_filtered_markers(self, marker_array, filtered_candidates, marker_id):
        """用红色线框和FILTERED文字显示未进入候选集合的YOLO箱体。"""
        filtered_color = (1.0, 0.08, 0.08)
        for candidate in filtered_candidates:
            corners = self._visualization_corners(candidate)
            outline = self._new_marker(marker_id, "yolo_box_filtered_outline", Marker.LINE_LIST)
            marker_id += 1
            outline.scale.x = 0.025
            self._set_marker_color(outline, filtered_color, 0.8)
            for start_index, end_index in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_index]))
                outline.points.append(self._point_message(corners[end_index]))
            marker_array.markers.append(outline)

            text_marker = self._new_marker(
                marker_id, "yolo_box_filtered_text", Marker.TEXT_VIEW_FACING
            )
            marker_id += 1
            text_marker.pose.position.x = candidate["map"][0]
            text_marker.pose.position.y = candidate["map"][1]
            text_marker.pose.position.z = self._candidate_top_height(candidate) + 0.14
            text_marker.scale.z = 0.10
            self._set_marker_color(text_marker, filtered_color, 1.0)
            text_marker.text = (
                f"#{candidate['index']} FILTERED\n"
                f"{candidate.get('filter_reason', 'unknown')}\n"
                f"map=({candidate['map'][0]:.2f}, {candidate['map'][1]:.2f}, "
                f"{candidate['map'][2]:.2f})"
            )
            marker_array.markers.append(text_marker)
        return marker_id

    def _log_filtered_candidates(self, filtered_candidates):
        """逐个打印被过滤目标，避免汇总数量掩盖TF或区域配置问题。"""
        for candidate in filtered_candidates:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] YOLO目标被过滤: index=#{candidate['index']}, "
                f"map=({candidate['map'][0]:.3f}, {candidate['map'][1]:.3f}, "
                f"{candidate['map'][2]:.3f}), distance={candidate['distance']:.3f}m, "
                f"reason={candidate.get('filter_reason', 'unknown')}"
            )

    def _publish_raw_box_visualization(self, raw_candidates):
        """把原始 YOLO 3D box 单独发到 raw 话题，方便和后续过滤结果分开看。"""
        if self.raw_visualization_publisher is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        raw_color = (0.75, 0.75, 0.75)
        marker_id = 1
        for candidate in raw_candidates:
            corners = self._visualization_corners(candidate)
            outline = self._new_marker(marker_id, "yolo_box_raw_outline", Marker.LINE_LIST)
            marker_id += 1
            outline.scale.x = 0.012
            self._set_marker_color(outline, raw_color, 0.55)
            for start_index, end_index in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_index]))
                outline.points.append(self._point_message(corners[end_index]))
            marker_array.markers.append(outline)

            text_marker = self._new_marker(
                marker_id, "yolo_box_raw_text", Marker.TEXT_VIEW_FACING
            )
            marker_id += 1
            text_marker.pose.position.x = candidate["map"][0]
            text_marker.pose.position.y = candidate["map"][1]
            text_marker.pose.position.z = self._candidate_top_height(candidate) + 0.08
            text_marker.scale.z = 0.07
            self._set_marker_color(text_marker, raw_color, 0.85)
            text_marker.text = (
                f"RAW #{candidate['index']}\n"
                f"map=({candidate['map'][0]:.2f}, {candidate['map'][1]:.2f}, "
                f"{candidate['map'][2]:.2f})"
            )
            marker_array.markers.append(text_marker)

        self.raw_visualization_publisher.publish(marker_array)

    def _clear_box_visualization(self):
        """新一轮选箱开始时清除RViz中上一轮的箱体，避免残影误导。"""
        if self.visualization_publisher is None:
            return
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.visualization_publisher.publish(marker_array)

    def _clear_raw_box_visualization(self):
        """清除原始 YOLO 3D box 的独立 RViz 话题，避免残影。"""
        if self.raw_visualization_publisher is None:
            return
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.raw_visualization_publisher.publish(marker_array)

    def _new_marker(self, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.ros_node.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _visualization_color(is_selected, is_top_level):
        if is_selected:
            return 1.0, 0.9, 0.0
        if is_top_level:
            return 0.0, 0.85, 1.0
        return 0.25, 0.45, 0.85

    @staticmethod
    def _set_marker_color(marker, color, alpha):
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = alpha

    @staticmethod
    def _point_message(position):
        return Point(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        )

    @staticmethod
    def _box_edge_indices():
        # _transform_box_geometry按sx/sy/sz依次枚举，索引每次改变一位即一条边。
        return (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )

    def _visualization_corners(self, candidate):
        geometry = candidate.get("geometry")
        if geometry is not None and len(geometry.get("corners", [])) == 8:
            return geometry["corners"]

        # 几何字段异常时仍画一个中心附近的默认线框，明确展示节点认为目标存在。
        center = candidate["map"]
        size = candidate.get("box", {}).get("size", [0.3, 0.3, 0.3])
        if len(size) != 3 or min(abs(float(value)) for value in size) < 0.01:
            size = [0.3, 0.3, 0.3]
        half = [abs(float(value)) * 0.5 for value in size]
        return [
            [center[0] + sx * half[0], center[1] + sy * half[1], center[2] + sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

    def _color_text(self, text, color):
        """给最终选箱与抓取决策日志增加醒目的 ANSI 颜色。"""
        if not self.enable_colored_log:
            return text

        color_codes = {
            "green": "\033[1;92m",
            "cyan": "\033[1;96m",
            "highlight": "\033[1;97;45m",
        }
        color_code = color_codes.get(color, "")
        reset_code = "\033[0m" if color_code else ""
        return f"{color_code}{text}{reset_code}"

    def _deduplicate_candidates(self, candidates):
        """合并map三维距离过近的单帧检测，并优先保留离机器人更近者。"""
        if self.duplicate_3d_distance_threshold <= 0.0:
            return list(candidates), []

        kept = []
        duplicate_records = []
        for candidate in sorted(candidates, key=lambda item: item["distance"]):
            duplicate = None
            duplicate_distance = None
            for existing in kept:
                distance = math.sqrt(sum(
                    (candidate["map"][axis] - existing["map"][axis]) ** 2
                    for axis in range(3)
                ))
                if distance < self.duplicate_3d_distance_threshold:
                    duplicate = existing
                    duplicate_distance = distance
                    break
            if duplicate is None:
                kept.append(candidate)
            else:
                duplicate_records.append(
                    (candidate["index"], duplicate["index"], duplicate_distance)
                )
        return kept, duplicate_records

    def _transform_box_geometry(self, map_from_source, box, map_xyz):
        """把YOLO局部OBB转换到map，返回姿态、角点、高度区间和水平轮廓。"""
        if not self.use_box_geometry_for_strategy:
            return None

        try:
            size = [abs(float(value)) for value in box.get("size", [])]
            quat = [float(value) for value in box.get("quat", [])]
            if len(size) != 3 or len(quat) != 4:
                return None
            if not all(math.isfinite(value) for value in size + quat):
                return None
            if min(size) < self.minimum_valid_box_size:
                return None
            quat_norm = math.sqrt(sum(value * value for value in quat))
            if quat_norm < 1e-6:
                return None
            quat = [value / quat_norm for value in quat]

            # T_map_box的旋转由T_map_source与YOLO给出的T_source_box相乘得到。
            map_box_rotation = tf_trans.concatenate_matrices(
                map_from_source,
                tf_trans.quaternion_matrix(quat),
            )
            map_box_rotation[0:3, 3] = [0.0, 0.0, 0.0]
            map_quat = tf_trans.quaternion_from_matrix(map_box_rotation)
            map_quat = [float(value) for value in map_quat]

            map_from_box = tf_trans.quaternion_matrix(map_quat)
            map_from_box[0:3, 3] = map_xyz
            half = [value * 0.5 for value in size]
            corners = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = map_from_box.dot(
                            [sx * half[0], sy * half[1], sz * half[2], 1.0]
                        )
                        corners.append([
                            float(corner[0]), float(corner[1]), float(corner[2])
                        ])

            return {
                "map_quat": map_quat,
                "size": size,
                "rotation": [
                    [float(map_from_box[row, column]) for column in range(3)]
                    for row in range(3)
                ],
                "corners": corners,
                "z_min": min(corner[2] for corner in corners),
                "z_max": max(corner[2] for corner in corners),
                "footprint": self._convex_hull_2d(
                    [(corner[0], corner[1]) for corner in corners]
                ),
            }
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] YOLO箱体OBB无效，回退箱心判断: {exc}"
            )
            return None

    @staticmethod
    def _candidate_top_height(candidate):
        geometry = candidate.get("geometry")
        return geometry["z_max"] if geometry is not None else candidate["map"][2]

    def _is_same_level(self, highest, candidate, max_height):
        """顶部接近或垂直实体区间充分重叠时视为同层。"""
        center_height_difference = abs(candidate["map"][2] - highest["map"][2])
        if center_height_difference > self.same_level_center_height_tolerance:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 排除非同层箱: highest=#{highest['index']}, "
                f"candidate=#{candidate['index']}, "
                f"center_z_difference={center_height_difference:.3f}m > "
                f"limit={self.same_level_center_height_tolerance:.3f}m"
            )
            return False
        if self._candidate_top_height(candidate) >= max_height - self.top_height_tolerance:
            return True
        highest_geometry = highest.get("geometry")
        candidate_geometry = candidate.get("geometry")
        if highest_geometry is None or candidate_geometry is None:
            return False
        return self._vertical_overlap_ratio(
            highest_geometry, candidate_geometry
        ) >= self.same_level_vertical_overlap_ratio

    @staticmethod
    def _vertical_overlap_ratio(first, second):
        overlap = max(
            0.0,
            min(first["z_max"], second["z_max"])
            - max(first["z_min"], second["z_min"]),
        )
        minimum_height = min(
            first["z_max"] - first["z_min"],
            second["z_max"] - second["z_min"],
        )
        return overlap / minimum_height if minimum_height > 1e-6 else 0.0

    @staticmethod
    def _convex_hull_2d(points):
        """Monotonic chain，生成OBB角点在map平面的凸包。"""
        unique_points = sorted(set((float(x), float(y)) for x, y in points))
        if len(unique_points) <= 1:
            return unique_points

        def cross(origin, first, second):
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        lower = []
        for point in unique_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    @classmethod
    def _polygon_distance_2d(cls, first, second):
        """返回两个凸多边形的最小表面间隙；相交时为0。"""
        if len(first) < 2 or len(second) < 2:
            return float("inf")
        if cls._point_in_convex_polygon(first[0], second) or cls._point_in_convex_polygon(
            second[0], first
        ):
            return 0.0
        first_edges = list(zip(first, first[1:] + first[:1]))
        second_edges = list(zip(second, second[1:] + second[:1]))
        for first_start, first_end in first_edges:
            for second_start, second_end in second_edges:
                if cls._segments_intersect(
                    first_start, first_end, second_start, second_end
                ):
                    return 0.0
        distances = [
            cls._point_segment_distance(point, start, end)
            for point in first
            for start, end in second_edges
        ] + [
            cls._point_segment_distance(point, start, end)
            for point in second
            for start, end in first_edges
        ]
        return min(distances) if distances else float("inf")

    @staticmethod
    def _point_in_convex_polygon(point, polygon):
        if len(polygon) < 3:
            return False
        signs = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            cross = (
                (end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0])
            )
            if abs(cross) > 1e-9:
                signs.append(cross > 0.0)
        return not signs or all(sign == signs[0] for sign in signs)

    @staticmethod
    def _segments_intersect(first_start, first_end, second_start, second_end):
        def orientation(start, end, point):
            return (
                (end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0])
            )

        o1 = orientation(first_start, first_end, second_start)
        o2 = orientation(first_start, first_end, second_end)
        o3 = orientation(second_start, second_end, first_start)
        o4 = orientation(second_start, second_end, first_end)
        epsilon = 1e-9

        def on_segment(start, end, point):
            return (
                min(start[0], end[0]) - epsilon <= point[0]
                <= max(start[0], end[0]) + epsilon
                and min(start[1], end[1]) - epsilon <= point[1]
                <= max(start[1], end[1]) + epsilon
            )

        if o1 * o2 < -epsilon and o3 * o4 < -epsilon:
            return True
        if abs(o1) <= epsilon and on_segment(first_start, first_end, second_start):
            return True
        if abs(o2) <= epsilon and on_segment(first_start, first_end, second_end):
            return True
        if abs(o3) <= epsilon and on_segment(second_start, second_end, first_start):
            return True
        if abs(o4) <= epsilon and on_segment(second_start, second_end, first_end):
            return True
        return False

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

    def _select_approach_for_selected_box(self, selected_candidate, robot_map_pose):
        """只围绕已选中的目标箱生成两条长边外侧站位，并选代价最低的可行者。"""
        evaluations = []
        geometry = selected_candidate.get("geometry")
        if geometry is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] box#{selected_candidate['index']} 缺少有效OBB，"
                "无法生成长边接近位姿"
            )
            return None, evaluations
        normal = self._long_edge_approach_normal(geometry)
        if normal is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] box#{selected_candidate['index']} 无法识别长边方向，"
                "无法生成长边接近位姿"
            )
            return None, evaluations

        for side_sign, side_name in ((1.0, "positive"), (-1.0, "negative")):
            point_x = selected_candidate["map"][0] + side_sign * normal[0] * self.approach_distance_m
            point_y = selected_candidate["map"][1] + side_sign * normal[1] * self.approach_distance_m
            yaw = math.degrees(math.atan2(
                selected_candidate["map"][1] - point_y,
                selected_candidate["map"][0] - point_x,
            ))
            pose = {
                "x": point_x,
                "y": point_y,
                "z": selected_candidate["map"][2],
                "yaw": yaw,
            }
            feasible, reason = self._evaluate_approach_geometry(pose)
            travel = math.hypot(point_x - robot_map_pose[0], point_y - robot_map_pose[1])
            yaw_error = abs(self._normalize_angle_deg(yaw - robot_map_pose[2]))
            evaluations.append({
                "box_index": selected_candidate["index"],
                "candidate": selected_candidate,
                "side": side_name,
                "pose": pose,
                "feasible": feasible,
                "reason": reason,
                "cost": travel + self.approach_yaw_cost_weight * yaw_error,
            })

        feasible = [evaluation for evaluation in evaluations if evaluation["feasible"]]
        if not feasible:
            return None, evaluations
        selected_evaluation = min(feasible, key=lambda item: item["cost"])
        selected_evaluation["selected"] = True
        pose = dict(selected_evaluation["pose"])
        pose["box_index"] = selected_evaluation["box_index"]
        pose["approach_side"] = selected_evaluation["side"]
        pose["geometric_only"] = True
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选中几何可行接近位姿: box=#{pose['box_index']}, "
            f"side={pose['approach_side']}, pose=({pose['x']:.3f}, {pose['y']:.3f}, "
            f"{pose['yaw']:.2f}deg), cost={selected_evaluation['cost']:.3f}, "
            f"travel={math.hypot(pose['x'] - robot_map_pose[0], pose['y'] - robot_map_pose[1]):.3f}, "
            f"yaw_error={abs(self._normalize_angle_deg(pose['yaw'] - robot_map_pose[2])):.2f}deg"
        )
        return pose, evaluations

    @staticmethod
    def _long_edge_approach_normal(geometry):
        """返回水平长边的单位法向；两个正负方向对应两条长边外侧。"""
        rotation = geometry.get("rotation")
        size = geometry.get("size")
        if rotation is None or size is None:
            return None
        axes = []
        for column in range(3):
            axis_x = float(rotation[0][column])
            axis_y = float(rotation[1][column])
            horizontal_norm = math.hypot(axis_x, axis_y)
            if horizontal_norm > 0.35:
                axes.append({
                    "axis": (axis_x / horizontal_norm, axis_y / horizontal_norm),
                    "size": float(size[column]),
                    "horizontal_norm": horizontal_norm,
                })
        if len(axes) < 2:
            return None
        horizontal_axes = sorted(axes, key=lambda item: item["horizontal_norm"], reverse=True)[:2]
        long_axis = max(horizontal_axes, key=lambda item: item["size"])
        # 在map水平面直接取长轴的垂线，比依赖另一个可能带倾斜误差的OBB轴更稳定。
        return -long_axis["axis"][1], long_axis["axis"][0]

    def _evaluate_approach_geometry(self, pose):
        point = (pose["x"], pose["y"])
        if self.pallet_map_polygon and is_map_position_in_polygon(
            {"x": point[0], "y": point[1]}, self.pallet_map_polygon
        ):
            return False, "inside_pallet_polygon"
        if self.pallet_map_polygon:
            clearance = self._point_to_polygon_boundary_distance(
                point,
                [(vertex["x"], vertex["y"]) for vertex in self.pallet_map_polygon],
            )
            if clearance < self.approach_pallet_clearance_m:
                return (
                    False,
                    "pallet_clearance_too_small({:.2f}m<{:.2f}m)".format(
                        clearance,
                        self.approach_pallet_clearance_m,
                    ),
                )
        return True, "geometry_feasible"

    @classmethod
    def _point_to_polygon_boundary_distance(cls, point, polygon):
        """返回点到多边形边界的最小平面距离。"""
        if len(polygon) < 2:
            return float("inf")
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        distances = [
            cls._point_segment_distance(point, start, end)
            for start, end in edges
        ]
        return min(distances) if distances else float("inf")

    @staticmethod
    def _normalize_angle_deg(angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def _decide_grasp_strategy(self, selected, top_candidates, approach_pose=None):
        """根据最高层邻箱，返回抓取策略及左右邻箱索引。

        新YOLO几何有效时，使用两个OBB水平轮廓的表面间隙与垂直重叠判断邻接；
        几何无效时才回退箱心距离。左右方向仍以机器人视角的
        ``selection_frame y`` 判断。

        - 右侧被占用、左侧空闲：向左拉；
        - 左侧被占用、右侧空闲：向右拉；
        - 左右均空闲：双爪直接抓；
        - 左右均被占用：没有安全外拉方向，返回no_safe_strategy。
        """
        selected_y = selected["selection_frame"][1]
        selected_map_x = selected["map"][0]
        selected_map_y = selected["map"][1]
        left_neighbors = []
        right_neighbors = []

        for candidate in top_candidates:
            if candidate["index"] == selected["index"]:
                continue
            if approach_pose is None:
                delta_y = candidate["selection_frame"][1] - selected_y
            else:
                yaw_rad = math.radians(approach_pose["yaw"])
                left_axis = (-math.sin(yaw_rad), math.cos(yaw_rad))
                delta_y = (
                    (candidate["map"][0] - selected_map_x) * left_axis[0]
                    + (candidate["map"][1] - selected_map_y) * left_axis[1]
                )
            map_delta_x = candidate["map"][0] - selected_map_x
            map_delta_y = candidate["map"][1] - selected_map_y
            center_distance = math.hypot(map_delta_x, map_delta_y)
            if not (
                self.neighbor_center_min_distance
                <= center_distance
                <= self.neighbor_center_max_distance
            ):
                continue
            selected_geometry = selected.get("geometry")
            candidate_geometry = candidate.get("geometry")
            if selected_geometry is not None and candidate_geometry is not None:
                vertical_overlap = self._vertical_overlap_ratio(
                    selected_geometry, candidate_geometry
                )
                surface_gap = self._polygon_distance_2d(
                    selected_geometry["footprint"],
                    candidate_geometry["footprint"],
                )
                if vertical_overlap < self.same_level_vertical_overlap_ratio:
                    continue
                if surface_gap > self.neighbor_surface_max_gap:
                    continue
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] OBB邻箱: target=#{selected['index']}, "
                    f"neighbor=#{candidate['index']}, center_distance={center_distance:.3f}m, "
                    f"surface_gap={surface_gap:.3f}m, "
                    f"vertical_overlap={vertical_overlap:.3f}"
                )
            if delta_y > 0.0:
                left_neighbors.append(candidate["index"])
            else:
                right_neighbors.append(candidate["index"])

        left_occupied = bool(left_neighbors)
        right_occupied = bool(right_neighbors)
        if not left_occupied and not right_occupied:
            strategy = "direct"
        elif not left_occupied and right_occupied:
            strategy = "left_pull"
        elif left_occupied and not right_occupied:
            strategy = "right_pull"
        else:
            strategy = "no_safe_strategy"
        return strategy, left_neighbors, right_neighbors

    def _on_yolo_boxes_string(self, data):
        """解析并缓存最新YOLO Boxes3D String；TF与选箱仍在行为树tick执行。"""
        boxes = parse_yolo_boxes_string(data)
        if not boxes:
            self._log_no_target(
                f"[{self.config_label}] 收到空或非法 YOLO boxes String: topic={self.yolo_topic}"
            )
            return
        with self.lock:
            self.latest_boxes = boxes
            self._message_generation += 1

    def _get_latest_boxes(self):
        with self.lock:
            if self._message_generation < self._minimum_generation:
                return None
            if self.freeze_after_first_valid_frame:
                if self._frozen_boxes is None and self.latest_boxes:
                    # 测试模式下只消费初始化后的第一帧有效 YOLO，避免边看边刷新。
                    self._frozen_boxes = list(self.latest_boxes)
                    self._frozen_generation = self._message_generation
                return self._frozen_boxes
            return self.latest_boxes

    def _lookup_transform_matrix(self, target_frame, source_frame):
        if target_frame == source_frame:
            return tf_trans.identity_matrix()
        try:
            zero_time = self.ros_node.zero_time()
            self.tf_listener.waitForTransform(
                target_frame, source_frame, zero_time,
                self.ros_node.duration(self.tf_timeout),
            )
            translation, rotation = self.tf_listener.lookupTransform(
                target_frame, source_frame, zero_time
            )
        except Exception as err:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 无法查询 {source_frame} -> {target_frame}: {err}"
            )
            return None
        return tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(translation), tf_trans.quaternion_matrix(rotation)
        )

    def _build_split_map_transform(self, source_frame):
        """组合得到 ``map <- YOLO源坐标系`` 的变换矩阵。

        实机上直接查询 camera -> map 的完整TF链可能不稳定，因此沿用头部盯箱链路，
        分别查询两段短链后相乘：

        ``T_map_source = T_map_chassis * T_base_source``

        当前底盘发布的 chassis_frame（通常为melon_odom）与base_link按项目既有约定
        表达同一底盘位姿关系，因此这里不额外插入一段长期TF查询。
        """
        map_from_chassis = self._lookup_transform_matrix(
            self.map_frame,
            self.chassis_frame,
        )
        base_from_source = self._lookup_transform_matrix(
            self.base_frame,
            source_frame,
        )
        if map_from_chassis is None or base_from_source is None:
            return None
        return tf_trans.concatenate_matrices(map_from_chassis, base_from_source)

    @staticmethod
    def _matrix_dot_point(matrix, point):
        transformed = matrix.dot([float(point.x), float(point.y), float(point.z), 1.0])
        return [float(transformed[0]), float(transformed[1]), float(transformed[2])]

    @staticmethod
    def _ros_stamp_to_seconds(stamp):
        if hasattr(stamp, "secs"):
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _log_no_target(self, message):
        now = time.monotonic()
        if now - self._last_no_target_log_time >= self.no_target_log_interval_sec:
            self.ros_node.get_logger().warning(message)
            self._last_no_target_log_time = now

    def describe_start(self):
        return (
            f"[{self.config_label}] SelectAndPublishHighestYoloBox start: "
            f"input={self.yolo_topic}, output={self.output_topic}, "
            f"map_frame={self.map_frame}, chassis_frame={self.chassis_frame}, "
            f"base_frame={self.base_frame}, "
            f"top_tolerance={self.top_height_tolerance:.3f}, "
            f"same_level_selection={self.same_level_selection}, "
            f"duplicate_3d_threshold={self.duplicate_3d_distance_threshold:.3f}, "
            f"map_region_enabled={bool(self.valid_box_map_polygon)}, "
            f"approach_planning={self.approach_pose_planning_enabled}, "
            f"approach_distance={self.approach_distance_m:.3f}, "
            f"pallet_region_enabled={bool(self.pallet_map_polygon)}, "
            f"neighbor_center_range=[{self.neighbor_center_min_distance:.3f}, "
            f"{self.neighbor_center_max_distance:.3f}], "
            f"use_obb={self.use_box_geometry_for_strategy}, "
            f"neighbor_surface_gap={self.neighbor_surface_max_gap:.3f}, "
            f"same_level_vertical_overlap={self.same_level_vertical_overlap_ratio:.3f}, "
            f"same_level_center_height_limit="
            f"{self.same_level_center_height_tolerance:.3f}"
        )
