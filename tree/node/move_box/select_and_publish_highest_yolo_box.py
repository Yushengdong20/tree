"""从一帧 YOLO 多箱结果中锁定最高层目标箱，并发布其 map 位姿。"""

import math
import threading
import time

import py_trees
import tf
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray, PoseStamped
from py_trees.common import Status

from tree.constants import BASE_LINK_FRAME, MAP_FRAME

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
        self.yolo_topic = str(params.get("yolo_topic", "/yolo/target_poses")).strip()
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
        # 根据目标箱同层左右邻箱占用情况生成抓取策略，供行为树Selector分支。
        self.grasp_strategy_key = str(
            params.get("grasp_strategy_key", "move_box_grasp_strategy")
        ).strip()
        # 两个箱子前后方向差值不超过该值，才认为处于同一排。
        self.same_row_forward_tolerance = float(
            params.get("same_row_forward_tolerance", 0.35)
        )
        # 横向距离过小通常是重复检测，过大则不是紧邻箱；只有落在此区间
        # 的同层同排箱子才会占用目标箱左侧或右侧的外拉空间。
        self.neighbor_lateral_min_distance = float(
            params.get("neighbor_lateral_min_distance", 0.10)
        )
        self.neighbor_lateral_max_distance = float(
            params.get("neighbor_lateral_max_distance", 0.80)
        )
        if self.neighbor_lateral_min_distance > self.neighbor_lateral_max_distance:
            raise ValueError(
                "neighbor_lateral_min_distance cannot exceed "
                "neighbor_lateral_max_distance"
            )
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
        self.no_target_log_interval_sec = float(params.get("no_target_log_interval_sec", 1.0))
        self.latest_msg = None
        self.lock = threading.Lock()
        self._last_no_target_log_time = 0.0
        self.tf_listener = tf.TransformListener()
        self.subscriber = self.ros_node.create_message_subscription(
            self.yolo_topic, PoseArray, self._on_yolo_pose_array, queue_size=1
        )
        self.publisher = self.ros_node.create_publisher(
            self.output_topic, PoseStamped, queue_size=1, latch=True
        )
        self.blackboard.register_key(key=self.selected_point_key, access=py_trees.common.Access.WRITE)
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

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        pose_array = self._get_latest_pose_array()
        if pose_array is None or not pose_array.poses:
            self._log_no_target(f"[{self.config_label}] 等待非空 YOLO PoseArray...")
            return Status.RUNNING

        source_frame = pose_array.header.frame_id or self.source_frame_fallback
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
        for index, pose in enumerate(pose_array.poses):
            map_xyz = self._matrix_dot_point(map_from_source, pose.position)
            distance_xyz = self._matrix_dot_point(distance_from_source, pose.position)
            planar_distance = math.hypot(distance_xyz[0], distance_xyz[1])
            # 可选的任务区域过滤：高度过低或离机器人过远的目标不参与后续排序。
            if self.min_map_height is not None and map_xyz[2] < self.min_map_height:
                continue
            if self.max_planar_distance is not None and planar_distance > self.max_planar_distance:
                continue
            candidates.append({
                "index": index,
                "map": map_xyz,
                "selection_frame": distance_xyz,
                "distance": planar_distance,
            })

        if not candidates:
            self._log_no_target(f"[{self.config_label}] YOLO目标均未通过高度/距离过滤")
            return Status.RUNNING

        # 阶段2：先找最高 z，再用容差形成“最高层候选集合”。
        # 例如最高 z=0.62m、容差=0.06m，则 z>=0.56m 都属于最高层。
        # 这样可避免同一排箱子因检测抖动几厘米而被误判为上下两层。
        max_height = max(candidate["map"][2] for candidate in candidates)
        top_candidates = [
            candidate for candidate in candidates
            if candidate["map"][2] >= max_height - self.top_height_tolerance
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
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = self.map_frame
        message.pose.position.x = selected["map"][0]
        message.pose.position.y = selected["map"][1]
        message.pose.position.z = selected["map"][2]
        message.pose.orientation.w = 1.0
        self.publisher.publish(message)
        self.blackboard.set(self.selected_point_key, list(selected["map"]), overwrite=True)
        # 阶段5：根据目标箱同层、同排的左右邻箱占用情况决定抓取方式。
        # 这使用箱子之间的相对位置，不使用目标在远处视野中的绝对y位置。
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
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已锁定最高层目标箱并发布: "
            f"strategy={self.same_level_selection}, index={selected['index']}, "
            f"top_z={max_height:.3f}, selected=({selected['map'][0]:.3f}, "
            f"{selected['map'][1]:.3f}, {selected['map'][2]:.3f}), "
            f"left_neighbors={left_neighbors}, right_neighbors={right_neighbors}, "
            f"grasp_strategy={grasp_strategy}, "
            f"strategy_key={self.grasp_strategy_key}, topic={self.output_topic}"
        )
        return Status.SUCCESS

    def _decide_grasp_strategy(self, selected, top_candidates):
        """根据最高层同排邻箱，返回抓取策略及左右邻箱索引。

        ``selection_frame`` 通常是base_link：x表示前后，y正方向表示左侧。
        只把与目标箱前后距离足够近、横向距离处于合理相邻范围的箱子
        视为邻箱，避免把另一排箱子或YOLO重复检测误当作碰撞障碍。

        - 右侧被占用、左侧空闲：向左拉；
        - 左侧被占用、右侧空闲：向右拉；
        - 左右均空闲：双爪直接抓；
        - 左右均被占用：没有安全外拉方向，返回no_safe_strategy。
        """
        selected_x = selected["selection_frame"][0]
        selected_y = selected["selection_frame"][1]
        left_neighbors = []
        right_neighbors = []

        for candidate in top_candidates:
            if candidate["index"] == selected["index"]:
                continue
            delta_x = candidate["selection_frame"][0] - selected_x
            delta_y = candidate["selection_frame"][1] - selected_y
            if abs(delta_x) > self.same_row_forward_tolerance:
                continue
            lateral_distance = abs(delta_y)
            if not (
                self.neighbor_lateral_min_distance
                <= lateral_distance
                <= self.neighbor_lateral_max_distance
            ):
                continue
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

    def _on_yolo_pose_array(self, msg):
        with self.lock:
            self.latest_msg = msg

    def _get_latest_pose_array(self):
        with self.lock:
            return self.latest_msg

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
            f"same_row_forward_tolerance={self.same_row_forward_tolerance:.3f}, "
            f"neighbor_lateral_range=[{self.neighbor_lateral_min_distance:.3f}, "
            f"{self.neighbor_lateral_max_distance:.3f}]"
        )
