"""从一帧 YOLO 多箱结果中锁定最高层最近箱，并发布其 map 位姿。"""

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
    """选择 map 高度最高的一层，再在同层中选择平面距离最近的箱子。"""

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
        self.selected_point_key = str(
            params.get("selected_point_key", "move_box_selected_highest_yolo_map_point")
        ).strip()
        self.top_height_tolerance = float(params.get("top_height_tolerance", 0.06))
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

        candidates = []
        for index, pose in enumerate(pose_array.poses):
            map_xyz = self._matrix_dot_point(map_from_source, pose.position)
            distance_xyz = self._matrix_dot_point(distance_from_source, pose.position)
            planar_distance = math.hypot(distance_xyz[0], distance_xyz[1])
            if self.min_map_height is not None and map_xyz[2] < self.min_map_height:
                continue
            if self.max_planar_distance is not None and planar_distance > self.max_planar_distance:
                continue
            candidates.append({"index": index, "map": map_xyz, "distance": planar_distance})

        if not candidates:
            self._log_no_target(f"[{self.config_label}] YOLO目标均未通过高度/距离过滤")
            return Status.RUNNING

        max_height = max(candidate["map"][2] for candidate in candidates)
        top_candidates = [
            candidate for candidate in candidates
            if candidate["map"][2] >= max_height - self.top_height_tolerance
        ]
        selected = min(top_candidates, key=lambda candidate: candidate["distance"])

        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = self.map_frame
        message.pose.position.x = selected["map"][0]
        message.pose.position.y = selected["map"][1]
        message.pose.position.z = selected["map"][2]
        message.pose.orientation.w = 1.0
        self.publisher.publish(message)
        self.blackboard.set(self.selected_point_key, list(selected["map"]), overwrite=True)

        candidate_text = ", ".join(
            "#{} map=({:.3f},{:.3f},{:.3f}) distance={:.3f}".format(
                candidate["index"], *candidate["map"], candidate["distance"]
            )
            for candidate in candidates
        )
        self.ros_node.get_logger().info(f"[{self.config_label}] YOLO候选: {candidate_text}")
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已锁定最高层最近箱并发布: index={selected['index']}, "
            f"top_z={max_height:.3f}, selected=({selected['map'][0]:.3f}, "
            f"{selected['map'][1]:.3f}, {selected['map'][2]:.3f}), topic={self.output_topic}"
        )
        return Status.SUCCESS

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
        """按 map<-melon_odom 与 base_link<-camera 两段短链组合 map<-camera。"""
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
            f"top_tolerance={self.top_height_tolerance:.3f}"
        )
