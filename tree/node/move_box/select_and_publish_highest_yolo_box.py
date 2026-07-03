"""从一帧 YOLO 多箱结果中锁定最高层目标箱，并发布其 map 位姿。"""

import math
import threading
import time

import py_trees
import tf
import tf.transformations as tf_trans
from kuavo_humanoid_sdk.common.yolo_boxes import (
    parse_yolo_boxes_string,
    serialize_yolo_box,
    yolo_box_center_point,
)
from py_trees.common import Status

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
        self.require_new_frame_after_initialise = self._to_bool(
            params.get("require_new_frame_after_initialise", False)
        )
        self.latest_boxes = None
        self._message_generation = 0
        self._minimum_generation = 0
        self.lock = threading.Lock()
        self._last_no_target_log_time = 0.0
        self.tf_listener = tf.TransformListener()
        self.subscriber = self.ros_node.create_string_subscription(
            self.yolo_topic, self._on_yolo_boxes_string, queue_size=1
        )
        self.publisher = self.ros_node.create_string_publisher(
            self.output_topic, queue_size=1, latch=True
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

    @staticmethod
    def _optional_float(value):
        if value is None or str(value).strip() == "":
            return None
        return float(value)

    def initialise(self):
        super().initialise()
        self._last_no_target_log_time = 0.0
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
        candidates = []
        area_filtered_count = 0
        for index, box in enumerate(boxes):
            center_point = yolo_box_center_point(box)
            map_xyz = self._matrix_dot_point(map_from_source, center_point)
            distance_xyz = self._matrix_dot_point(distance_from_source, center_point)
            geometry = self._transform_box_geometry(map_from_source, box, map_xyz)
            planar_distance = math.hypot(distance_xyz[0], distance_xyz[1])
            # 可选的任务区域过滤：高度过低或离机器人过远的目标不参与后续排序。
            if not is_map_position_in_polygon(
                {"x": map_xyz[0], "y": map_xyz[1]},
                self.valid_box_map_polygon,
            ):
                area_filtered_count += 1
                continue
            if self.min_map_height is not None and map_xyz[2] < self.min_map_height:
                continue
            if self.max_planar_distance is not None and planar_distance > self.max_planar_distance:
                continue
            candidates.append({
                "index": index,
                "map": map_xyz,
                "selection_frame": distance_xyz,
                "distance": planar_distance,
                "box": box,
                "geometry": geometry,
            })

        if not candidates:
            self._log_no_target(
                f"[{self.config_label}] YOLO目标均未通过区域/高度/距离过滤: "
                f"区域外数量={area_filtered_count}"
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

        # 阶段3：仅在最高层中执行二次选择。distance是base_link平面距离；
        # selection_frame[1]是机器人横向坐标，正值在左、负值在右。
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

        # 阶段4：发布并保存唯一选中的箱子。ROS话题用于跨机器通信，
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
        if self.selected_box_key:
            self.blackboard.set(self.selected_box_key, selected_box, overwrite=True)
        # 阶段5：根据目标箱同层邻箱的占用情况决定抓取方式。是否相邻使用
        # map 平面箱心绝对距离，因此不会因机器人斜对箱堆而产生投影误判。
        grasp_strategy, left_neighbors, right_neighbors = self._decide_grasp_strategy(
            selected,
            top_candidates,
        )
        if self.grasp_strategy_key:
            self.blackboard.set(
                self.grasp_strategy_key,
                grasp_strategy,
                overwrite=True,
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
                f"区域外={area_filtered_count}, 区域内={raw_candidate_count}"
            )
        selected_message = (
            f"[{self.config_label}] 已锁定最高层目标箱并发布: "
            f"strategy={self.same_level_selection}, index={selected['index']}, "
            f"top_z={max_height:.3f}, selected=({selected['map'][0]:.3f}, "
            f"{selected['map'][1]:.3f}, {selected['map'][2]:.3f}), "
            f"geometry={'obb' if selected['geometry'] is not None else 'center_fallback'}, "
            f"left_neighbors={left_neighbors}, right_neighbors={right_neighbors}, "
            f"grasp_strategy={grasp_strategy}, "
            f"strategy_key={self.grasp_strategy_key}, topic={self.output_topic}"
        )
        self.ros_node.get_logger().info(
            self._color_text(selected_message, "highlight")
        )
        return Status.SUCCESS

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

    def _decide_grasp_strategy(self, selected, top_candidates):
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
            delta_y = candidate["selection_frame"][1] - selected_y
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
            f"neighbor_center_range=[{self.neighbor_center_min_distance:.3f}, "
            f"{self.neighbor_center_max_distance:.3f}], "
            f"use_obb={self.use_box_geometry_for_strategy}, "
            f"neighbor_surface_gap={self.neighbor_surface_max_gap:.3f}, "
            f"same_level_vertical_overlap={self.same_level_vertical_overlap_ratio:.3f}, "
            f"same_level_center_height_limit="
            f"{self.same_level_center_height_tolerance:.3f}"
        )
