"""使用 YOLO 箱体中心进行远距离粗靠近。"""

import math
import os
import time
import uuid
from datetime import datetime

import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import Point
from py_trees.common import Status
from kuavo_humanoid_sdk.common.yolo_boxes import serialize_yolo_box
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import (
    BASE_LINK_FRAME,
    CHASSIS_FRAME,
    FINAL_POSE_KEY,
    FLOW_RESULT_KEY,
    MAP_FRAME,
    ROBOT_SERVICES_KEY,
)

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    DEFAULT_NAVIGATION_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_YOLO_TARGET_DISTANCE_M,
    TASK_STATUS_LABELS,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_FAILED_TASK_STATUSES,
    build_chassis_config,
    build_yolo_approach_pose,
    extract_navigation_task_id,
    extract_task_status,
    get_chassis_current_pose,
    post_chassis_navigation,
    post_navigation_task_status,
    transform_global_point_to_base,
)
from tree.utils.box_map_polygon import is_map_position_in_polygon, parse_map_polygon
from tree.utils.geometry import get_odom_pose_transformer, lookup_transform_matrix


class MoveBoxYoloApproachToBox(TimedMockAction):
    """远距离使用 YOLO 检测箱体中心，并导航到箱体前方指定距离。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip(),
        )
        self.target_distance_m = float(
            params.get("target_distance_m", DEFAULT_YOLO_TARGET_DISTANCE_M)
        )
        self.navigation_timeout_sec = int(
            params.get("navigation_timeout_sec", DEFAULT_NAVIGATION_TIMEOUT_SEC)
        )
        self.poll_interval_sec = float(params.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))
        self.navigation_target_key = str(
            params.get("navigation_target_key", "move_box_yolo_navigation_target")
        ).strip()
        self.selected_map_point_key = str(
            params.get("selected_map_point_key", "")
        ).strip()
        self.selected_box_key = str(params.get("selected_box_key", "")).strip()
        self.box_map_pose_topic = str(
            params.get("box_map_pose_topic", "/move_box/yolo_box_pose_map")
        ).strip()
        self.keep_running_after_success = self._to_bool(
            params.get("keep_running_after_success", False)
        )
        self.use_box_memory = self._to_bool(params.get("use_box_memory", False))
        self.box_memory_key = str(params.get("box_memory_key", "move_box_box_memory")).strip()
        self.current_box_target_key = str(
            params.get("current_box_target_key", "move_box_current_box_target")
        ).strip()
        self.finished_box_targets_key = str(
            params.get("finished_box_targets_key", "move_box_finished_box_targets")
        ).strip()
        self.memory_match_distance_m = float(params.get("memory_match_distance_m", 0.35))
        self.max_memory_targets = int(params.get("max_memory_targets", 1))
        self.max_memory_targets = max(self.max_memory_targets, 0)
        self.memory_update_during_navigation = self._to_bool(
            params.get("memory_update_during_navigation", True)
        )
        self.memory_update_interval_sec = float(
            params.get("memory_update_interval_sec", 0.5)
        )
        self.memory_update_interval_sec = max(self.memory_update_interval_sec, 0.05)
        self.min_detected_box_3d_distance_m = float(
            params.get("min_detected_box_3d_distance_m", 0.25)
        )
        self.use_tf_3d_transform = self._to_bool(params.get("use_tf_3d_transform", True))
        self.tf_target_frame = str(params.get("tf_target_frame", MAP_FRAME)).strip()
        self.tf_base_frame = str(params.get("tf_base_frame", BASE_LINK_FRAME)).strip()
        self.tf_timeout_sec = float(params.get("tf_timeout_sec", 0.2))
        self.odom_topic = str(params.get("odom_topic", CHASSIS_FRAME)).strip()
        self.odom_history_duration_sec = float(
            params.get("odom_history_duration_sec", 10.0)
        )
        self.odom_match_time_offset_sec = float(
            params.get("odom_match_time_offset_sec", 0.0)
        )
        self.odom_match_max_delta_sec = self._optional_float(
            params.get("odom_match_max_delta_sec", "")
        )
        self.allowed_class_ids = self._optional_int_set(
            params.get("allowed_class_ids", [])
        )
        self.valid_box_map_polygon = parse_map_polygon(
            params.get("valid_box_map_polygon", [])
        )
        self.valid_box_polygon_required = self._to_bool(
            params.get("valid_box_polygon_required", False)
        )
        if self.valid_box_polygon_required and not self.valid_box_map_polygon:
            raise ValueError("valid_box_polygon_required=True 时必须配置 valid_box_map_polygon")
        self.enable_colored_log = self._to_bool(params.get("enable_colored_log", True))
        self.navigation_visualization_enabled = self._to_bool(
            params.get("navigation_visualization_enabled", True)
        )
        self.navigation_visualization_topic = str(
            params.get(
                "navigation_visualization_topic",
                "/move_box/yolo_navigation_markers",
            )
        ).strip()
        self.enable_memory_file_log = self._to_bool(
            params.get("enable_memory_file_log", True)
        )
        self.memory_log_dir = str(params.get("memory_log_dir", "/mnt/ssd/log")).strip()
        self.memory_log_file = str(
            params.get("memory_log_file", "move_box_memory.log")
        ).strip()
        self._memory_file_log_warning_reported = False
        self.box_map_pose_pub = None
        if self.box_map_pose_topic:
            self.box_map_pose_pub = self.ros_node.create_string_publisher(
                self.box_map_pose_topic,
                queue_size=1,
                latch=True,
            )
        self.navigation_visualization_pub = None
        if self.navigation_visualization_enabled and self.navigation_visualization_topic:
            self.navigation_visualization_pub = self.ros_node.create_publisher(
                self.navigation_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        # 关键步骤：YOLO 目标转 map 不直接查完整 TF 链，而是复用工具里的 odom 位姿缓存。
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
            target_frame=self.tf_target_frame,
            base_frame=self.tf_base_frame,
            history_duration_sec=self.odom_history_duration_sec,
        )
        self._no_valid_yolo_warning_at = None
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=FINAL_POSE_KEY, access=py_trees.common.Access.WRITE)
        if self.navigation_target_key:
            self.blackboard.register_key(key=self.navigation_target_key, access=py_trees.common.Access.WRITE)
        if self.selected_map_point_key:
            self.blackboard.register_key(
                key=self.selected_map_point_key,
                access=py_trees.common.Access.READ,
            )
        if self.selected_box_key:
            self.blackboard.register_key(
                key=self.selected_box_key,
                access=py_trees.common.Access.READ,
            )
        if self.use_box_memory:
            self.blackboard.register_key(key=self.box_memory_key, access=py_trees.common.Access.READ)
            self.blackboard.register_key(key=self.box_memory_key, access=py_trees.common.Access.WRITE)
            self.blackboard.register_key(
                key=self.current_box_target_key,
                access=py_trees.common.Access.READ,
            )
            self.blackboard.register_key(
                key=self.current_box_target_key,
                access=py_trees.common.Access.WRITE,
            )
            self.blackboard.register_key(
                key=self.finished_box_targets_key,
                access=py_trees.common.Access.READ,
            )
        self._reset_state()

    def _reset_state(self):
        self._phase = "IDLE"
        self._current_pose = None
        self._box_base_position = None
        self._box_global_position = None
        self._detected_box_targets = []
        self._visualization_box_targets = []
        self._filtered_box_targets = []
        self._current_box_target = None
        self._current_target_source = "无"
        self._target_pose = None
        self._task_id = None
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = None
        self._next_poll_at = None
        self._next_memory_update_at = None
        self._no_valid_yolo_warning_at = None
        self._last_yolo_frame_generation = 0
        self._latest_detection_frame = None

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    @staticmethod
    def _optional_float(value):
        if value is None or str(value).strip() == "":
            return None
        return float(value)

    @staticmethod
    def _optional_int_set(value):
        """解析允许的YOLO类别；空列表表示不限制类别。"""
        if value is None:
            return set()
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return set()
            return {int(part.strip()) for part in stripped.split(",") if part.strip()}
        if isinstance(value, (list, tuple, set)):
            return {int(item) for item in value}
        return set()

    def initialise(self):
        super().initialise()
        self._reset_state()
        self._clear_navigation_target_pose()
        self._clear_navigation_visualization()
        self._phase = "GET_POSE"
        self._deadline = time.monotonic() + self.navigation_timeout_sec

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            now = time.monotonic()
            if self._phase == "FINISHED":
                self._update_memory_while_navigation(now)
                return Status.RUNNING
            if now > self._deadline:
                raise TimeoutError(
                    f"YOLO 靠近导航超时: task_instance_id={self._task_instance_id}, "
                    f"latest_response={self._navigation_status_response}"
                )

            if self._phase == "GET_POSE":
                self.ros_node.set_live_runtime(self.config_label, "YOLO_APPROACH", "读取底盘当前位姿")
                self._current_pose = get_chassis_current_pose(self.chassis_config)
                self._phase = "READ_YOLO"
                return Status.RUNNING

            if self._phase == "READ_YOLO":
                self.ros_node.set_live_runtime(self.config_label, "YOLO_APPROACH", "读取 YOLO 箱体中心")
                if self.selected_box_key or self.selected_map_point_key:
                    updated = self._load_preselected_map_target()
                else:
                    services = self._get_services()
                    updated = self._update_yolo_targets(services)
                    self._choose_current_target_from_yolo()
                if self._current_box_target is None:
                    if not (self.selected_box_key or self.selected_map_point_key):
                        self.ros_node.set_live_runtime(
                            self.config_label,
                            "YOLO_APPROACH",
                            "当前帧无可用 YOLO 箱体，继续等待下一帧",
                        )
                        self._log_no_valid_yolo_warning(updated)
                        return Status.RUNNING
                    raise RuntimeError(f"尚未获得有效 YOLO 箱体中心: updated={updated}")

                self._box_base_position = self._derive_target_base_position(
                    self._current_box_target
                )
                self._box_global_position = self._current_box_target.get("map_position")
                self._publish_box_map_pose()
                box_distance_m = math.hypot(
                    self._box_global_position["x"] - self._current_pose.x,
                    self._box_global_position["y"] - self._current_pose.y,
                )
                if box_distance_m <= self.target_distance_m:
                    self.ros_node.get_logger().info(
                        f"[{self.config_label}] YOLO 箱体距离 {box_distance_m:.3f}m "
                        f"<= 目标距离 {self.target_distance_m:.3f}m，跳过粗导航"
                    )
                    self._store_navigation_target_pose(
                        self._current_pose.x,
                        self._current_pose.y,
                        self._current_pose.yaw,
                    )
                    self._publish_navigation_visualization(
                        self._current_pose.x,
                        self._current_pose.y,
                        self._current_pose.yaw,
                        skipped=True,
                    )
                    self._store_result(need_navigation=False, box_distance_m=box_distance_m)
                    self.blackboard.final_pose = {
                        "x": self._current_pose.x,
                        "y": self._current_pose.y,
                        "yaw": self._current_pose.yaw,
                    }
                    self.ros_node.clear_live_runtime()
                    return self._success_status()

                self._target_pose = build_yolo_approach_pose(
                    self._current_pose,
                    self._box_global_position["x"],
                    self._box_global_position["y"],
                    self.target_distance_m,
                )
                self._box_distance_m = box_distance_m
                self._log_info(
                    "YOLO粗导航求解",
                    "选中箱map坐标=%s 底盘当前map位姿=(%.3f, %.3f, %.3f) "
                    "求解导航目标map=(%.3f, %.3f, %.3f)"
                    % (
                        self._format_position(self._box_global_position),
                        self._current_pose.x,
                        self._current_pose.y,
                        self._current_pose.yaw,
                        self._target_pose.x,
                        self._target_pose.y,
                        self._target_pose.yaw,
                    ),
                    "green",
                )
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] YOLO 粗靠近目标: "
                    f"箱子base坐标={self._format_position(self._box_base_position)}, "
                    f"记忆启用={self.use_box_memory}, "
                    f"导航目标=({self._target_pose.x:.3f}, {self._target_pose.y:.3f}, {self._target_pose.yaw:.3f})"
                )
                self._publish_navigation_visualization(
                    self._target_pose.x,
                    self._target_pose.y,
                    self._target_pose.yaw,
                    skipped=False,
                )
                self._phase = "CREATE_NAVIGATION"
                return Status.RUNNING

            if self._phase == "CREATE_NAVIGATION":
                self._task_id = str(uuid.uuid4())
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "YOLO_APPROACH",
                    f"创建 YOLO 粗靠近导航: ({self._target_pose.x:.3f}, "
                    f"{self._target_pose.y:.3f}, {self._target_pose.yaw:.3f})",
                )
                self._navigation_response = post_chassis_navigation(
                    self.chassis_config,
                    self._task_id,
                    self._target_pose.x,
                    self._target_pose.y,
                    self._target_pose.yaw,
                )
                self._task_instance_id = extract_navigation_task_id(self._navigation_response)
                self._store_navigation_target_pose(
                    self._target_pose.x,
                    self._target_pose.y,
                    self._target_pose.yaw,
                )
                self._next_poll_at = now
                self._next_memory_update_at = now + self.memory_update_interval_sec
                self._phase = "POLL_NAVIGATION"
                return Status.RUNNING

            if self._phase == "POLL_NAVIGATION":
                self._update_memory_while_navigation(now)
                if self._next_poll_at is not None and now < self._next_poll_at:
                    return Status.RUNNING
                self._navigation_status_response = post_navigation_task_status(
                    self.chassis_config,
                    self._task_instance_id,
                )
                task_status = extract_task_status(self._navigation_status_response)
                status_label = TASK_STATUS_LABELS.get(task_status, "未知状态")
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] YOLO 粗靠近导航状态: "
                    f"task_instance_id={self._task_instance_id}, task_status={task_status}({status_label})"
                )
                if task_status == TASK_STATUS_SUCCEEDED:
                    return self._finish_success()
                if task_status in TERMINAL_FAILED_TASK_STATUSES:
                    raise RuntimeError(f"YOLO 粗靠近导航失败: {self._navigation_status_response}")
                self._next_poll_at = now + self.poll_interval_sec
                return Status.RUNNING

            raise RuntimeError(f"unsupported YOLO approach phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] YOLO 粗靠近失败: {exc}")
            return Status.FAILURE

    def _finish_success(self):
        self._store_result(need_navigation=True, box_distance_m=self._box_distance_m)
        self.blackboard.final_pose = {
            "x": self._target_pose.x,
            "y": self._target_pose.y,
            "yaw": self._target_pose.yaw,
        }
        self.ros_node.clear_live_runtime()
        return self._success_status()

    def _log_no_valid_yolo_warning(self, updated, throttle_sec=1.0):
        now = time.monotonic()
        if (
            self._no_valid_yolo_warning_at is not None
            and now - self._no_valid_yolo_warning_at < throttle_sec
        ):
            return
        self._no_valid_yolo_warning_at = now
        frame_summary = self._format_detection_frame_summary(self._latest_detection_frame)
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 当前帧未获得可用 YOLO 箱体，继续等待: "
            f"updated={updated}, valid={len(self._detected_box_targets)}, "
            f"filtered={len(self._filtered_box_targets)}, {frame_summary}"
        )

    def _success_status(self):
        # 关键步骤：部分流程里 YOLO 只负责后台粗靠近，成功后保持 RUNNING，避免抢先结束外层并行。
        if self.keep_running_after_success:
            self._phase = "FINISHED"
            return Status.RUNNING
        return Status.SUCCESS

    def _get_services(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            raise RuntimeError(f"robot services missing on blackboard: key={self.services_key}")
        if not hasattr(services, "yolo_detector"):
            raise RuntimeError("robot services 缺少 yolo_detector")
        return services

    def _update_memory_while_navigation(self, now):
        """YOLO 粗靠近过程中持续用本轮检测刷新记忆，直到 FP 接管。"""
        if not self.use_box_memory or not self.memory_update_during_navigation:
            return
        if self._next_memory_update_at is not None and now < self._next_memory_update_at:
            return

        self._next_memory_update_at = now + self.memory_update_interval_sec
        try:
            # 关键步骤：机器人移动中 base 坐标持续变化，刷新当前底盘位姿后再换算 map 坐标。
            self._current_pose = get_chassis_current_pose(self.chassis_config)
            services = self._get_services()
            self._update_yolo_targets(services)
            self._choose_current_target_from_yolo()
            if self._current_box_target is not None:
                self._box_base_position = self._derive_target_base_position(
                    self._current_box_target
                )
                self._box_global_position = self._current_box_target.get("map_position")
                self._publish_box_map_pose()
        except Exception as exc:
            self._log_info(
                "导航中记忆刷新失败",
                "错误=%s，保留上一轮导航和记忆状态继续等待" % exc,
                "yellow",
            )

    def _update_yolo_targets(self, services):
        """读取 YOLO 多目标结果，并转换成本轮可用于导航和记忆的目标列表。"""
        detector = getattr(services, "yolo_detector", None)
        detection_frame = detector.get_latest_detection_frame() if detector is not None else None
        self._latest_detection_frame = detection_frame
        frame_generation = int(detection_frame.get("generation", 0)) if detection_frame else 0
        updated = bool(detection_frame) and frame_generation != self._last_yolo_frame_generation
        raw_detection = self._extract_boxes_from_detection_frame(detection_frame)
        candidate_targets = self._build_targets_from_raw_detection(services, raw_detection)

        # 关键步骤：先让所有YOLO目标进入过滤与可视化，再从有效集合中选最近箱。
        # 这样当最近目标属于非箱体类别时，它会以灰色过滤框显示，但不会挡住
        # 后面的合法 class_id=0/1/2 箱体。
        target_candidates = candidate_targets

        if detection_frame is not None:
            self._last_yolo_frame_generation = frame_generation

        self._log_info(
            "YOLO检测",
            "是否更新=%s 原始数量=%d 区域过滤启用=%s 3D重叠阈值=%.3fm"
            % (
                updated,
                len(raw_detection),
                bool(self.valid_box_map_polygon),
                self.min_detected_box_3d_distance_m,
            ),
            "magenta",
        )
        self._detected_box_targets = []
        self._filtered_box_targets = []
        self._visualization_box_targets = []
        filtered_count = 0
        for index, target in enumerate(target_candidates):
            base_position = dict(target.get("_base_position") or {})
            map_position = target.get("map_position") or {}
            source_box = target.get("box") or {}
            if not self._is_class_allowed(source_box):
                filtered_count += 1
                filtered_target = {
                    "id": "",
                    "map_position": map_position,
                    "box": dict(source_box),
                    "geometry": target.get("geometry"),
                    "_base_position": base_position,
                    "filter_reason": "class_id_not_allowed",
                    "filter_text": "类别不在允许抓取ID内",
                }
                self._filtered_box_targets.append(filtered_target)
                self._visualization_box_targets.append(filtered_target)
                self._log_info(
                    "YOLO目标过滤",
                    "序号=%d/%d 过滤类型=类别不允许 class_id=%s allowed=%s "
                    "base坐标=%s map坐标=%s"
                    % (
                        index + 1,
                        len(target_candidates),
                        source_box.get("class_id", "?"),
                        sorted(self.allowed_class_ids),
                        self._format_position(base_position),
                        self._format_position(map_position),
                    ),
                    "yellow",
                )
                continue
            if not self._is_map_position_allowed(map_position):
                filtered_count += 1
                filtered_target = {
                    "id": "",
                    "map_position": map_position,
                    "box": dict(target.get("box") or {}),
                    "geometry": target.get("geometry"),
                    "_base_position": base_position,
                    "filter_reason": "outside_valid_box_map_polygon",
                    "filter_text": "指定区域外",
                }
                self._filtered_box_targets.append(filtered_target)
                self._visualization_box_targets.append(filtered_target)
                self._log_info(
                    "YOLO目标过滤",
                    "序号=%d/%d 过滤类型=指定区域外 base坐标=%s map坐标=%s"
                    % (
                        index + 1,
                        len(target_candidates),
                        self._format_position(base_position),
                        self._format_position(map_position),
                    ),
                    "yellow",
                )
                continue

            valid_target = {
                "id": "",
                "map_position": map_position,
                "box": dict(target.get("box") or {}),
                "geometry": target.get("geometry"),
                "_base_position": base_position,
            }
            overlap_target, overlap_index, overlap_distance = self._find_overlapped_detected_target(
                valid_target
            )
            if overlap_target is not None:
                filtered_count += 1
                valid_target["filter_reason"] = "overlapped_detected_target"
                valid_target["filter_text"] = "与已有箱子3D重叠"
                valid_target["overlap_distance"] = overlap_distance
                self._filtered_box_targets.append(valid_target)
                self._visualization_box_targets.append(valid_target)
                self._log_info(
                    "YOLO目标过滤",
                    "序号=%d/%d 过滤类型=与已有箱子3D重叠 3D距离=%.3fm "
                    "重叠对象序号=%d base坐标=%s map坐标=%s 重叠对象=%s"
                    % (
                        index + 1,
                        len(target_candidates),
                        overlap_distance,
                        overlap_index + 1,
                        self._format_position(base_position),
                        self._format_position(map_position),
                        self._format_target(overlap_target),
                    ),
                    "yellow",
                )
                continue

            self._log_info(
                "YOLO目标有效",
                "序号=%d/%d base坐标=%s map坐标=%s"
                % (
                    index + 1,
                    len(target_candidates),
                    self._format_position(base_position),
                    self._format_position(map_position),
                ),
                "magenta",
            )
            self._detected_box_targets.append(valid_target)
            self._visualization_box_targets.append(valid_target)
        self._log_info(
            "YOLO检测统计",
            "原始数量=%d 有效数量=%d 过滤数量=%d"
            % (
                len(raw_detection),
                len(self._detected_box_targets),
                filtered_count,
            ),
            "magenta",
        )
        self._publish_detection_only_visualization()
        self._log_target_list("YOLO有效目标列表", self._detected_box_targets)
        return updated

    def _is_class_allowed(self, source_box):
        """class白名单过滤；未配置白名单时所有类别都可参与计算。"""
        if not self.allowed_class_ids:
            return True
        try:
            class_id = int(source_box.get("class_id"))
        except (TypeError, ValueError):
            return False
        return class_id in self.allowed_class_ids

    def _build_targets_from_raw_detection(self, services, raw_detection):
        targets = []
        for source_box in raw_detection:
            target = self._build_target_from_raw_box(services, source_box)
            if target is not None:
                targets.append(target)
        return targets

    def _build_target_from_raw_box(self, services, source_box):
        if not isinstance(source_box, dict):
            return None
        center = source_box.get("center", [0.0, 0.0, 0.0])
        if len(center) < 3:
            return None
        source_frame = source_box.get("frame_id") or BASE_LINK_FRAME
        transform_result = self._build_time_aligned_map_transform(
            source_frame,
            source_box.get("stamp"),
        )
        if transform_result is None:
            return None
        base_from_source, map_from_source = transform_result
        center_xyz = [float(center[0]), float(center[1]), float(center[2])]
        base_xyz = self._matrix_dot_xyz(base_from_source, center_xyz)
        map_xyz = self._matrix_dot_xyz(map_from_source, center_xyz)
        base_position = {
            "x": float(base_xyz[0]),
            "y": float(base_xyz[1]),
            "z": float(base_xyz[2]),
            "_stamp_sec": float(source_box.get("stamp", 0.0) or 0.0),
        }
        map_position = {
            "x": float(map_xyz[0]),
            "y": float(map_xyz[1]),
            "z": float(map_xyz[2]),
        }
        map_box = self._build_map_box(source_box, map_position)
        geometry = self._build_box_geometry_from_map_transform(map_from_source, source_box, map_xyz)
        return {
            "map_position": map_position,
            "box": map_box,
            "geometry": geometry,
            "_base_position": base_position,
        }

    @staticmethod
    def _choose_nearest_raw_target(targets):
        nearest_target = None
        nearest_distance = None
        for target in targets or []:
            base_position = target.get("_base_position") or {}
            distance = math.sqrt(
                float(base_position.get("x", 0.0)) ** 2
                + float(base_position.get("y", 0.0)) ** 2
                + float(base_position.get("z", 0.0)) ** 2
            )
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_target = target
        return nearest_target

    def _load_preselected_map_target(self):
        """按master内部box dict格式读取上游锁定目标，兼容旧map点作为回退。"""
        selected_box = None
        if self.selected_box_key and self.blackboard.exists(self.selected_box_key):
            raw_box = self.blackboard.get(self.selected_box_key)
            if isinstance(raw_box, dict):
                center = raw_box.get("center")
                if isinstance(center, (list, tuple)) and len(center) >= 3:
                    selected_box = dict(raw_box)

        if selected_box is None:
            if not self.selected_map_point_key or not self.blackboard.exists(
                self.selected_map_point_key
            ):
                raise RuntimeError(
                    "blackboard 缺少上游选箱结果: "
                    f"box_key={self.selected_box_key}, point_key={self.selected_map_point_key}"
                )
            point = self.blackboard.get(self.selected_map_point_key)
            if not isinstance(point, (list, tuple)) or len(point) < 3:
                raise RuntimeError(
                    f"上游选箱map点格式错误: key={self.selected_map_point_key}, value={point!r}"
                )
            selected_box = {
                "frame_id": MAP_FRAME,
                "stamp": self._ros_stamp_to_seconds(self.ros_node.now()),
                "center": [float(point[0]), float(point[1]), float(point[2])],
                "quat": [0.0, 0.0, 0.0, 1.0],
                "size": [0.0, 0.0, 0.0],
                "score": 0.0,
                "class_id": -1,
            }

        center = selected_box["center"]
        map_position = {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2]),
        }
        self._current_box_target = {
            "id": "",
            "map_position": map_position,
            "box": self._build_map_box(selected_box, map_position),
        }
        self._detected_box_targets = [self._current_box_target]
        self._current_target_source = (
            f"blackboard:{self.selected_box_key or self.selected_map_point_key}"
        )
        self._publish_detection_only_visualization()
        self._log_current_target()
        return True

    def _transform_base_position_to_map_position(self, services, base_position, source_frame):
        """优先按 source -> base_link -> odom/map 将 YOLO 点转换到 map。"""
        target_stamp_sec = None
        if isinstance(base_position, dict):
            target_stamp_sec = base_position.get("_stamp_sec")
        odom_msg = self._get_time_aligned_odom(target_stamp_sec)
        if self.use_tf_3d_transform:
            map_position, fallback_exc = (
                self.odom_transformer.transform_point_to_map_with_pose2d_fallback(
                    getattr(services, "tf_listener", None),
                    base_position,
                    source_frame=source_frame,
                    fallback_pose=self._current_pose,
                    timeout=self.tf_timeout_sec,
                    odom_msg=odom_msg,
                )
            )
            if fallback_exc is not None:
                self._log_info(
                    "YOLO坐标转换",
                    "source->base->odom/map 3D转换失败，回退到2D yaw近似: %s, "
                    "yolo_stamp=%.3f, matched_odom=%s"
                    % (
                        fallback_exc,
                        float(target_stamp_sec or 0.0),
                        "yes" if odom_msg is not None else "no",
                    ),
                    "yellow",
                )
            return map_position

        return self.odom_transformer.transform_base_point_to_map_with_pose2d(
            base_position,
            self._current_pose,
        )

    def _get_time_aligned_odom(self, yolo_stamp_sec):
        target_stamp_sec = None
        if yolo_stamp_sec is not None:
            target_stamp_sec = float(yolo_stamp_sec) + self.odom_match_time_offset_sec
        odom_msg = self.odom_transformer.get_nearest_odom_by_stamp_sec(target_stamp_sec)
        if odom_msg is None:
            return None
        if self.odom_match_max_delta_sec is None or target_stamp_sec is None:
            return odom_msg
        matched_odom_stamp_sec = self._ros_stamp_to_seconds(odom_msg.header.stamp)
        if abs(matched_odom_stamp_sec - target_stamp_sec) > self.odom_match_max_delta_sec:
            self._log_info(
                "YOLO时间对齐超窗",
                "yolo_stamp=%.3f target_stamp=%.3f matched_odom_stamp=%.3f "
                "delta_ms=%.1f limit_ms=%.1f"
                % (
                    float(yolo_stamp_sec or 0.0),
                    float(target_stamp_sec),
                    float(matched_odom_stamp_sec),
                    abs(matched_odom_stamp_sec - target_stamp_sec) * 1000.0,
                    self.odom_match_max_delta_sec * 1000.0,
                ),
                "yellow",
            )
            return None
        return odom_msg

    @staticmethod
    def _build_map_box(source_box, map_position):
        """用 map 坐标重建内部主 box，保证保存和发布都以 map frame 为准。"""
        box = dict(source_box)
        box["frame_id"] = MAP_FRAME
        box["center"] = [
            float(map_position["x"]),
            float(map_position["y"]),
            float(map_position.get("z", 0.0)),
        ]
        # 当前粗靠近只可靠转换中心点，朝向不从相机系硬带到 map 系。
        box["quat"] = [0.0, 0.0, 0.0, 1.0]
        return box

    def _update_visualization_targets(self, services, detection_frame):
        """仅为RViz维护一份全部YOLO候选，不参与粗导航选目标逻辑。"""
        if not detection_frame:
            return

        raw_detection = self._extract_boxes_from_detection_frame(detection_frame)
        if not raw_detection:
            return

        visualization_targets = []
        for target_pose in raw_detection:
            center = target_pose.get("center", [0.0, 0.0, 0.0])
            if len(center) < 3:
                continue
            base_position = {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "_stamp_sec": float(target_pose.get("stamp", 0.0)),
            }
            source_frame = target_pose.get("frame_id") or BASE_LINK_FRAME
            map_position = self._transform_base_position_to_map_position(
                services,
                base_position,
                source_frame,
            )
            visualization_targets.append(
                {
                    "id": "",
                    "map_position": map_position,
                    "box": self._build_map_box(target_pose, map_position),
                    "geometry": self._build_box_geometry(target_pose, source_frame, services),
                    "_base_position": base_position,
                }
            )
        if visualization_targets:
            self._visualization_box_targets = visualization_targets

    @staticmethod
    def _extract_boxes_from_detection_frame(detection_frame):
        if not isinstance(detection_frame, dict):
            return []
        return [dict(box) for box in detection_frame.get("boxes", [])]

    def _build_box_geometry(self, source_box, source_frame, services):
        """尽量把 YOLO 原始3D箱子的8个角点转换到 map，供粗靠近阶段可视化。"""
        if not isinstance(source_box, dict):
            return None

        center = source_box.get("center")
        size = source_box.get("size")
        quat = source_box.get("quat")
        if (
            not isinstance(center, (list, tuple))
            or not isinstance(size, (list, tuple))
            or not isinstance(quat, (list, tuple))
            or len(center) < 3
            or len(size) < 3
            or len(quat) < 4
        ):
            return None

        sx, sy, sz = [abs(float(value)) for value in size[:3]]
        if min(sx, sy, sz) < 0.01:
            return None

        try:
            rotation = tf_trans.quaternion_matrix(
                [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            )[:3, :3]
        except Exception:
            return None

        center_xyz = [float(center[0]), float(center[1]), float(center[2])]
        stamp_sec = float(source_box.get("stamp", 0.0) or 0.0)
        half = [sx * 0.5, sy * 0.5, sz * 0.5]
        corners = []
        for dx in (-half[0], half[0]):
            for dy in (-half[1], half[1]):
                for dz in (-half[2], half[2]):
                    rotated = rotation.dot([dx, dy, dz])
                    source_point = {
                        "x": center_xyz[0] + float(rotated[0]),
                        "y": center_xyz[1] + float(rotated[1]),
                        "z": center_xyz[2] + float(rotated[2]),
                        "_stamp_sec": stamp_sec,
                    }
                    map_point = self._transform_base_position_to_map_position(
                        services,
                        source_point,
                        source_frame,
                    )
                    corners.append(
                        [
                            float(map_point["x"]),
                            float(map_point["y"]),
                            float(map_point.get("z", 0.0)),
                        ]
                    )
        return {
            "corners": corners,
            "z_max": max(corner[2] for corner in corners) if corners else float(center_xyz[2]),
            "z_min": min(corner[2] for corner in corners) if corners else float(center_xyz[2]),
        }

    def _build_box_geometry_from_map_transform(self, map_from_source, source_box, map_xyz):
        """按 SelectAndPublishHighestYoloBox 的方式直接由 map<-source 变换整框。"""
        if not isinstance(source_box, dict):
            return None
        try:
            size = [abs(float(value)) for value in source_box.get("size", [])]
            quat = [float(value) for value in source_box.get("quat", [])]
            if len(size) != 3 or len(quat) != 4:
                return None
            if min(size) < 0.01:
                return None
            quat_norm = math.sqrt(sum(value * value for value in quat))
            if quat_norm < 1e-6:
                return None
            quat = [value / quat_norm for value in quat]

            map_box_rotation = tf_trans.concatenate_matrices(
                map_from_source,
                tf_trans.quaternion_matrix(quat),
            )
            map_box_rotation[0:3, 3] = [0.0, 0.0, 0.0]
            map_quat = tf_trans.quaternion_from_matrix(map_box_rotation)

            map_from_box = tf_trans.quaternion_matrix(map_quat)
            map_from_box[0:3, 3] = map_xyz
            half = [value * 0.5 for value in size]
            corners = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = map_from_box.dot([sx * half[0], sy * half[1], sz * half[2], 1.0])
                        corners.append([float(corner[0]), float(corner[1]), float(corner[2])])

            return {
                "corners": corners,
                "z_max": max(corner[2] for corner in corners),
                "z_min": min(corner[2] for corner in corners),
            }
        except Exception:
            return None

    def _build_time_aligned_map_transform(self, source_frame, stamp_sec):
        """按 YOLO 时间戳组合 map<-source 变换，和选箱节点保持一致。"""
        base_from_source = self._lookup_transform_matrix(self.tf_base_frame, source_frame)
        if base_from_source is None:
            return None
        target_stamp_sec = float(stamp_sec or 0.0) + self.odom_match_time_offset_sec
        odom_msg = self.odom_transformer.get_nearest_odom_by_stamp_sec(target_stamp_sec)
        if odom_msg is None:
            return None
        if self.odom_match_max_delta_sec is not None:
            matched_odom_stamp_sec = self._ros_stamp_to_seconds(odom_msg.header.stamp)
            odom_delta_sec = abs(float(matched_odom_stamp_sec) - target_stamp_sec)
            if odom_delta_sec > self.odom_match_max_delta_sec:
                self._log_info(
                    "YOLO时间对齐超窗",
                    "yolo_stamp=%.3f target_stamp=%.3f matched_odom_stamp=%.3f "
                    "delta_ms=%.1f limit_ms=%.1f"
                    % (
                        float(stamp_sec or 0.0),
                        float(target_stamp_sec),
                        float(matched_odom_stamp_sec),
                        odom_delta_sec * 1000.0,
                        self.odom_match_max_delta_sec * 1000.0,
                    ),
                    "yellow",
                )
                return None
        map_from_base = self._map_from_odom_message(odom_msg)
        return base_from_source, tf_trans.concatenate_matrices(map_from_base, base_from_source)

    def _lookup_transform_matrix(self, target_frame, source_frame):
        if target_frame == source_frame:
            return tf_trans.identity_matrix()
        try:
            return lookup_transform_matrix(
                getattr(self._get_services(), "tf_listener", None),
                self.ros_node,
                target_frame,
                source_frame,
                timeout=self.tf_timeout_sec,
            )
        except Exception as exc:
            self._log_info(
                "TF查询失败",
                "%s <- %s: %s" % (target_frame, source_frame, exc),
                "yellow",
            )
            return None

    @staticmethod
    def _matrix_dot_xyz(matrix, xyz):
        transformed = matrix.dot([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0])
        return [float(transformed[0]), float(transformed[1]), float(transformed[2])]

    @staticmethod
    def _map_from_odom_message(odom_msg):
        odom_position = odom_msg.pose.pose.position
        odom_orientation = odom_msg.pose.pose.orientation
        return tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(
                [
                    float(odom_position.x),
                    float(odom_position.y),
                    float(odom_position.z),
                ]
            ),
            tf_trans.quaternion_matrix(
                [
                    float(odom_orientation.x),
                    float(odom_orientation.y),
                    float(odom_orientation.z),
                    float(odom_orientation.w),
                ]
            ),
        )

    @staticmethod
    def _strip_runtime_target_fields(target):
        """移除只在当前 tick 内使用的派生字段，避免写入 blackboard/记忆。"""
        if target is None:
            return None
        stripped = dict(target)
        stripped.pop("_base_position", None)
        stripped.pop("base_position", None)
        return stripped

    def _choose_current_target_from_yolo(self):
        """选择本轮抓取目标，并把其它 YOLO 目标写入箱子记忆。"""
        self._current_target_source = "无"
        if not self.use_box_memory:
            nearest_target = self._get_nearest_detected_target()
            if nearest_target is not None:
                self._current_box_target = nearest_target
                self._current_target_source = "单目标检测"
                self._log_current_target()
            return

        current_target = self._read_current_box_target()
        detected_current = self._match_current_target(current_target)
        if detected_current is None and current_target is None:
            detected_current = self._get_nearest_detected_target()
            if detected_current is not None:
                self._current_target_source = "最近检测目标"
        elif detected_current is not None:
            self._current_target_source = "匹配当前目标"

        if detected_current is not None:
            self._current_box_target = detected_current
            self._current_box_target["id"] = self._ensure_target_id(
                current_target,
                self._current_box_target,
            )
        elif current_target is not None and self._is_target_allowed(current_target):
            self._current_box_target = current_target
            self._current_target_source = "记忆回退"
        else:
            if current_target is not None:
                self._log_info(
                    "YOLO当前目标过滤",
                    "过滤类型=当前目标在指定区域外 目标=%s"
                    % self._format_target(current_target),
                    "yellow",
                )
            self._current_box_target = None

        if self._current_box_target is not None:
            self.blackboard.set(
                self.current_box_target_key,
                self._strip_runtime_target_fields(self._current_box_target),
                overwrite=True,
            )
            self._log_current_target()
        else:
            self._log_info(
                "YOLO当前目标",
                "来源=无 有效检测数量=%d 记忆数量=%d"
                % (len(self._detected_box_targets), len(self._read_box_memory())),
                "yellow",
            )
        self._refresh_box_memory()

    def _derive_target_base_position(self, target):
        """从 map 主数据临时派生当前 base_link 下的位置，不写回目标对象。"""
        if target is None:
            return None
        base_position = target.get("_base_position")
        if base_position is not None:
            return dict(base_position)

        map_position = target.get("map_position")
        if map_position is None or self._current_pose is None:
            return None

        base_position = transform_global_point_to_base(
            self._current_pose,
            float(map_position["x"]),
            float(map_position["y"]),
        )
        # 关键步骤：内部目标保存 map frame；没有完整 TF 时不把 map z 伪装成 base z。
        base_position["z"] = 0.0
        return base_position

    def _read_current_box_target(self):
        """从 blackboard 读取当前任务目标。"""
        if self.blackboard.exists(self.current_box_target_key):
            target = self.blackboard.get(self.current_box_target_key)
            if target:
                return target
        return None

    def _read_box_memory(self):
        """读取箱子记忆列表，缺失时返回空列表。"""
        if self.blackboard.exists(self.box_memory_key):
            memory = self.blackboard.get(self.box_memory_key)
            if isinstance(memory, list):
                return list(memory)
        return []

    def _match_current_target(self, current_target):
        """在本次 YOLO 检测中匹配当前记忆目标。"""
        if current_target is None or not self._detected_box_targets:
            return None

        current_position = current_target.get("map_position")
        if current_position is None:
            return None

        nearest_target = None
        nearest_distance = None
        for detected_target in self._detected_box_targets:
            distance = self._target_distance(current_position, detected_target.get("map_position"))
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_target = detected_target

        if nearest_distance is not None and nearest_distance <= self.memory_match_distance_m:
            return nearest_target
        return None

    def _get_nearest_detected_target(self):
        """当前没有记忆目标时，选择离机器人最近的 YOLO 箱子。"""
        nearest_target = None
        nearest_distance = None
        for detected_target in self._detected_box_targets:
            base_position = self._derive_target_base_position(detected_target)
            if base_position is None:
                continue
            distance = math.sqrt(
                base_position["x"] * base_position["x"]
                + base_position["y"] * base_position["y"]
                + base_position["z"] * base_position["z"]
            )
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_target = detected_target
        return nearest_target

    def _ensure_target_id(self, old_target, new_target):
        """保留旧目标 id；新目标没有 id 时按 map 位置生成稳定字符串。"""
        if old_target is not None and old_target.get("id"):
            return old_target["id"]
        if new_target.get("id"):
            return new_target["id"]
        map_position = new_target.get("map_position", {})
        return "box_{:.2f}_{:.2f}".format(
            float(map_position.get("x", 0.0)),
            float(map_position.get("y", 0.0)),
        )

    def _refresh_box_memory(self):
        """把本次检测到的非当前目标刷新为候选下一个箱子。"""
        old_memory = self._read_box_memory()
        # 关键步骤：永远相信本轮 YOLO 检测，不把历史记忆合并回来。
        memory = []
        memory_before_count = len(old_memory)
        self._log_info(
            "记忆刷新开始",
            "刷新策略=observed 最大保留数量=%d 刷新前数量=%d 有效检测数量=%d 当前目标=%s"
            % (
                self.max_memory_targets,
                memory_before_count,
                len(self._detected_box_targets),
                self._format_target(self._current_box_target),
            ),
            "cyan",
        )
        self._log_target_list("记忆刷新前列表", old_memory)
        for detected_target in self._detected_box_targets:
            if not self._is_target_allowed(detected_target):
                self._log_info(
                    "记忆跳过",
                    "跳过类型=指定区域外 目标=%s" % self._format_target(detected_target),
                    "yellow",
                )
                continue
            current_target_distance = self._get_current_target_distance(detected_target)
            if current_target_distance <= self.memory_match_distance_m:
                self._log_info(
                    "记忆跳过",
                    "跳过类型=当前抓取目标 3D距离=%.3fm 阈值=%.3fm 目标=%s 当前目标=%s"
                    % (
                        current_target_distance,
                        self.memory_match_distance_m,
                        self._format_target(detected_target),
                        self._format_target(self._current_box_target),
                    ),
                    "cyan",
                )
                continue
            finished_target, finished_target_distance = self._find_finished_target_match(
                detected_target
            )
            if finished_target is not None:
                self._log_info(
                    "记忆跳过",
                    "跳过类型=已完成目标 3D距离=%.3fm 阈值=%.3fm 目标=%s 已完成目标=%s"
                    % (
                        finished_target_distance,
                        self.memory_match_distance_m,
                        self._format_target(detected_target),
                        self._format_target(finished_target),
                    ),
                    "yellow",
                )
                continue
            self._merge_memory_target(memory, detected_target)

        memory = self._limit_memory_targets(memory)
        self.blackboard.set(self.box_memory_key, memory, overwrite=True)
        self._log_info(
            "记忆刷新完成",
            "刷新策略=observed 最大保留数量=%d 刷新前数量=%d 刷新后数量=%d 当前目标=%s"
            % (
                self.max_memory_targets,
                memory_before_count,
                len(memory),
                self._format_target(self._current_box_target),
            ),
            "cyan",
        )
        self._log_target_list("候选下一个目标列表", memory)

    def _limit_memory_targets(self, memory):
        """只保留滚动决策需要的少量候选目标，默认仅保留下一个箱子。"""
        if self.max_memory_targets == 0 or len(memory) <= self.max_memory_targets:
            return memory

        # 关键步骤：下一个箱子按当前底盘位置排序，避免保留完整历史队列造成旧误检残留。
        sorted_memory = sorted(memory, key=self._target_distance_to_current_pose)
        dropped_targets = sorted_memory[self.max_memory_targets:]
        kept_targets = sorted_memory[: self.max_memory_targets]
        for index, target in enumerate(dropped_targets):
            self._log_info(
                "记忆裁剪",
                "裁剪序号=%d/%d 目标=%s"
                % (index + 1, len(dropped_targets), self._format_target(target)),
                "yellow",
            )
        return kept_targets

    def _target_distance_to_current_pose(self, target):
        """计算目标到当前底盘 map 位姿的平面距离，用于挑选下一个候选箱子。"""
        map_position = target.get("map_position") if target is not None else None
        if map_position is None or self._current_pose is None:
            return float("inf")
        return math.hypot(
            float(map_position.get("x", 0.0)) - self._current_pose.x,
            float(map_position.get("y", 0.0)) - self._current_pose.y,
        )

    def _merge_memory_target(self, memory, target):
        """按 map 坐标 3D 距离去重合并一个记忆目标。"""
        target_position = target.get("map_position")
        if target_position is None:
            self._log_info(
                "记忆跳过",
                "跳过类型=缺少map坐标 目标=%s" % self._format_target(target),
                "yellow",
            )
            return
        if not self._is_map_position_allowed(target_position):
            self._log_info(
                "记忆跳过",
                "跳过类型=指定区域外 目标=%s" % self._format_target(target),
                "yellow",
            )
            return

        target = self._strip_runtime_target_fields(target)
        target["id"] = self._ensure_target_id(None, target)
        for index, memory_target in enumerate(memory):
            distance = self._target_distance(target_position, memory_target.get("map_position"))
            if distance <= self.memory_match_distance_m:
                target["id"] = memory_target.get("id") or target["id"]
                memory[index] = target
                self._log_info(
                    "记忆更新",
                    "序号=%d/%d 3D距离=%.3fm 旧目标=%s 新目标=%s"
                    % (
                        index + 1,
                        len(memory),
                        distance,
                        self._format_target(memory_target),
                        self._format_target(target),
                    ),
                    "cyan",
                )
                return

        memory.append(target)
        self._log_info(
            "记忆新增",
            "序号=%d/%d 目标=%s" % (len(memory), len(memory), self._format_target(target)),
            "cyan",
        )

    def _get_current_target_distance(self, target):
        """计算检测目标到当前抓取目标的 3D 距离。"""
        if self._current_box_target is None:
            return float("inf")
        return self._target_distance(
            target.get("map_position"),
            self._current_box_target.get("map_position"),
        )

    def _find_finished_target_match(self, target):
        """查找与检测目标匹配的已完成箱子。"""
        if not self.blackboard.exists(self.finished_box_targets_key):
            return None, None

        finished_targets = self.blackboard.get(self.finished_box_targets_key)
        if not isinstance(finished_targets, list):
            return None, None

        target_position = target.get("map_position")
        for finished_target in finished_targets:
            distance = self._target_distance(
                target_position,
                finished_target.get("map_position"),
            )
            if distance <= self.memory_match_distance_m:
                return finished_target, distance
        return None, None

    def _is_target_allowed(self, target):
        """判断目标是否落在配置的有效 map 区域内。"""
        if target is None:
            return False
        return self._is_map_position_allowed(target.get("map_position"))

    def _is_map_position_allowed(self, map_position):
        """判断 map 坐标是否允许参与 YOLO 选择和记忆。"""
        return is_map_position_in_polygon(map_position, self.valid_box_map_polygon)

    def _find_overlapped_detected_target(self, target):
        """在本轮已保留目标中查找 3D 距离过近的重复检测。"""
        if self.min_detected_box_3d_distance_m <= 0.0:
            return None, None, None

        target_position = target.get("map_position")
        for index, detected_target in enumerate(self._detected_box_targets):
            distance = self._target_distance(
                target_position,
                detected_target.get("map_position"),
            )
            if distance <= self.min_detected_box_3d_distance_m:
                return detected_target, index, distance
        return None, None, None

    @staticmethod
    def _target_distance(left_position, right_position):
        """计算两个 map 目标的 3D 距离，用于区分叠放箱子。"""
        if left_position is None or right_position is None:
            return float("inf")
        dx = float(left_position.get("x", 0.0)) - float(right_position.get("x", 0.0))
        dy = float(left_position.get("y", 0.0)) - float(right_position.get("y", 0.0))
        dz = float(left_position.get("z", 0.0)) - float(right_position.get("z", 0.0))
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _log_current_target(self):
        """输出当前 YOLO 选中目标。"""
        memory_count = len(self._read_box_memory()) if self.use_box_memory else 0
        self._log_info(
            "YOLO当前目标",
            "来源=%s 目标=%s 有效检测数量=%d 记忆数量=%d"
            % (
                self._current_target_source,
                self._format_target(self._current_box_target),
                len(self._detected_box_targets),
                memory_count,
            ),
            "green",
        )

    def _log_target_list(self, tag, targets):
        """按列表完整打印箱子目标，便于复盘记忆新增和残留来源。"""
        self._log_info(
            tag,
            "数量=%d" % len(targets),
            "cyan",
        )
        for index, target in enumerate(targets):
            self._log_info(
                tag,
                "序号=%d/%d %s"
                % (index + 1, len(targets), self._format_target(target)),
                "cyan",
            )

    def _log_info(self, tag, message, color):
        """输出带固定前缀和可选颜色的调试日志。"""
        text = f"[{self.config_label}] [{tag}] {message}"
        self.ros_node.get_logger().info(self._color_text(text, color))
        self._write_memory_file_log(text)

    def _write_memory_file_log(self, text):
        """把 YOLO 和箱子记忆日志追加写入独立文件，方便单独检查。"""
        if not self.enable_memory_file_log:
            return

        try:
            # 关键步骤：现场可能未提前创建目录，这里保证日志目录存在。
            os.makedirs(self.memory_log_dir, exist_ok=True)
            log_path = os.path.join(self.memory_log_dir, self.memory_log_file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp} {text}\n")
        except Exception as exc:
            if not self._memory_file_log_warning_reported:
                self._memory_file_log_warning_reported = True
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 写入箱子记忆日志失败: {exc}"
                )

    def _color_text(self, text, color):
        """按配置给日志添加 ANSI 颜色。"""
        if not self.enable_colored_log:
            return text

        color_codes = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "cyan": "\033[96m",
            "magenta": "\033[95m",
        }
        color_code = color_codes.get(color, "")
        reset_code = "\033[0m" if color_code else ""
        return f"{color_code}{text}{reset_code}"

    @staticmethod
    def _format_target(target):
        """格式化箱子目标，便于日志排查。"""
        if target is None:
            return "None"
        return "id=%s map坐标=%s" % (
            target.get("id", ""),
            MoveBoxYoloApproachToBox._format_position(target.get("map_position")),
        )

    @staticmethod
    def _format_position(position):
        """格式化三维/二维坐标。"""
        if position is None:
            return "None"
        return "(%.3f, %.3f, %.3f)" % (
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        )

    def _store_result(self, need_navigation, box_distance_m):
        try:
            flow_result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            flow_result = None
        if flow_result is None:
            flow_result = {}
        flow_result["yoloApproach"] = {
            "needNavigation": need_navigation,
            "currentPose": {
                "x": self._current_pose.x,
                "y": self._current_pose.y,
                "yaw": self._current_pose.yaw,
            },
            "boxBasePosition": self._box_base_position,
            "boxGlobalPosition": self._box_global_position,
            "boxMemoryEnabled": self.use_box_memory,
            "currentBoxTarget": self._strip_runtime_target_fields(self._current_box_target),
            "boxDistanceM": box_distance_m,
            "targetDistanceM": self.target_distance_m,
            "targetPose": None
            if self._target_pose is None
            else {
                "x": self._target_pose.x,
                "y": self._target_pose.y,
                "yaw": self._target_pose.yaw,
            },
            "navigation": self._navigation_response,
            "navigationStatus": self._navigation_status_response,
        }
        self.blackboard.flow_result = flow_result

    def _store_navigation_target_pose(self, x, y, yaw):
        """把 YOLO 粗靠近导航目标写入黑板，供并行距离监听节点读取。"""
        if not self.navigation_target_key:
            return

        # 关键步骤：导航任务真正创建后才写入目标，避免监听节点在发车前抢跑。
        self.blackboard.set(
            self.navigation_target_key,
            {
                "x": x,
                "y": y,
                "yaw": yaw,
            },
            overwrite=True,
        )

    def _clear_navigation_target_pose(self):
        """清空上一轮 YOLO 导航目标，避免监听节点误用旧目标。"""
        if not self.navigation_target_key:
            return
        self.blackboard.set(self.navigation_target_key, None, overwrite=True)

    def _publish_box_map_pose(self):
        """发布 map 坐标系下的 YOLO 箱体三维框 String。"""
        if self.box_map_pose_pub is None:
            return

        source_box = {}
        if self._current_box_target is not None:
            source_box = dict(self._current_box_target.get("box") or {})

        # 关键步骤：FP 初始化需要 map 下中心点，同时继续沿用 YOLO 的尺寸、分数和类别。
        box = dict(source_box)
        box["frame_id"] = MAP_FRAME
        box["stamp"] = self._ros_stamp_to_seconds(self.ros_node.now())
        box["center"] = [
            float(self._box_global_position["x"]),
            float(self._box_global_position["y"]),
            float(
                self._box_global_position.get(
                    "z",
                    (source_box.get("center") or [0.0, 0.0, 0.0])[2],
                )
            ),
        ]
        # 当前粗靠近链路只做中心点 map 转换，朝向没有完整 map 变换时用单位四元数占位。
        box["quat"] = [0.0, 0.0, 0.0, 1.0]
        payload = serialize_yolo_box(box, frame_id=MAP_FRAME, stamp=box["stamp"])
        self.box_map_pose_pub.publish(payload)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布 map 下 YOLO 箱体 String: "
            f"topic={self.box_map_pose_topic}, "
            f"center=({box['center'][0]:.3f}, {box['center'][1]:.3f}, {box['center'][2]:.3f}), "
            f"size={box.get('size')}, score={box.get('score')}, class_id={box.get('class_id')}"
        )

    def _publish_navigation_visualization(self, target_x, target_y, target_yaw, skipped):
        """在map下显示用于导航的箱子、候选箱、目标站位、连线和最终朝向。"""
        if self.navigation_visualization_pub is None or self._box_global_position is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        visualization_targets = (
            self._visualization_box_targets
            if self._visualization_box_targets
            else self._detected_box_targets
        )
        selected_id = id(self._current_box_target) if self._current_box_target is not None else None
        marker_id = 1
        for index, target in enumerate(visualization_targets):
            map_position = target.get("map_position") or {}
            corners = (target.get("geometry") or {}).get("corners") or self._fallback_box_corners(target)
            is_selected = id(target) == selected_id
            if (not is_selected) and self._current_box_target is not None:
                target_map = target.get("map_position")
                current_map = self._current_box_target.get("map_position")
                if self._target_distance(target_map, current_map) <= self.min_detected_box_3d_distance_m:
                    is_selected = True
            is_filtered = bool(target.get("filter_reason"))
            color = (
                (0.65, 0.65, 0.65)
                if is_filtered
                else ((1.0, 0.88, 0.0) if is_selected else (0.20, 0.75, 1.0))
            )

            outline = self._new_navigation_marker(
                marker_id, "yolo_navigation_box_outline", Marker.LINE_LIST
            )
            marker_id += 1
            outline.scale.x = 0.032 if is_selected else 0.014
            self._set_navigation_marker_color(
                outline,
                color[0],
                color[1],
                color[2],
                0.35 if is_filtered else (0.95 if is_selected else 0.70),
            )
            for start_index, end_index in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_index]))
                outline.points.append(self._point_message(corners[end_index]))
            marker_array.markers.append(outline)

            center_marker = self._new_navigation_marker(
                marker_id, "yolo_navigation_box_center_all", Marker.SPHERE
            )
            marker_id += 1
            center_marker.pose.position = Point(
                x=float(map_position.get("x", 0.0)),
                y=float(map_position.get("y", 0.0)),
                z=float(map_position.get("z", 0.0)),
            )
            center_marker.pose.orientation.w = 1.0
            center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = (
                0.10 if is_selected else 0.05
            )
            self._set_navigation_marker_color(
                center_marker, color[0], color[1], color[2], 0.65 if is_filtered else 1.0
            )
            marker_array.markers.append(center_marker)

            top_z = self._target_top_height(target, corners)
            text_marker = self._new_navigation_marker(
                marker_id, "yolo_navigation_box_text_all", Marker.TEXT_VIEW_FACING
            )
            marker_id += 1
            text_marker.pose.position.x = float(map_position.get("x", 0.0))
            text_marker.pose.position.y = float(map_position.get("y", 0.0))
            text_marker.pose.position.z = top_z + 0.10
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.10 if is_selected else 0.075
            self._set_navigation_marker_color(
                text_marker, color[0], color[1], color[2], 1.0
            )
            source_box = target.get("box") or {}
            filter_suffix = ""
            if is_filtered:
                filter_suffix = f" FILTERED[{target.get('filter_text') or target.get('filter_reason')}]"
            text_marker.text = (
                f"ROUGH YOLO #{index}{' SELECTED' if is_selected else ''}{filter_suffix}\n"
                f"map=({float(map_position.get('x', 0.0)):.2f}, "
                f"{float(map_position.get('y', 0.0)):.2f}, "
                f"{float(map_position.get('z', 0.0)):.2f})\n"
                f"class={source_box.get('class_id', '?')} "
                f"score={float(source_box.get('score', 0.0)):.2f}"
            )
        marker_array.markers.append(text_marker)

        box_point = Point(
            x=float(self._box_global_position["x"]),
            y=float(self._box_global_position["y"]),
            z=float(self._box_global_position.get("z", 0.0)),
        )
        target_point = Point(x=float(target_x), y=float(target_y), z=0.05)

        box_marker = self._new_navigation_marker(marker_id, "yolo_navigation_box", Marker.SPHERE)
        marker_id += 1
        box_marker.pose.position = box_point
        box_marker.pose.orientation.w = 1.0
        box_marker.scale.x = box_marker.scale.y = box_marker.scale.z = 0.14
        self._set_navigation_marker_color(box_marker, 1.0, 0.2, 0.1, 1.0)
        marker_array.markers.append(box_marker)

        connection = self._new_navigation_marker(marker_id, "yolo_navigation_relation", Marker.LINE_LIST)
        marker_id += 1
        connection.scale.x = 0.025
        connection.points = [box_point, target_point]
        self._set_navigation_marker_color(connection, 1.0, 0.65, 0.0, 0.95)
        marker_array.markers.append(connection)

        arrow = self._new_navigation_marker(marker_id, "yolo_navigation_goal", Marker.ARROW)
        marker_id += 1
        arrow.pose.position.x = float(target_x)
        arrow.pose.position.y = float(target_y)
        arrow.pose.position.z = 0.08
        yaw_rad = math.radians(float(target_yaw))
        arrow.pose.orientation.z = math.sin(yaw_rad * 0.5)
        arrow.pose.orientation.w = math.cos(yaw_rad * 0.5)
        arrow.scale.x = 0.65
        arrow.scale.y = 0.13
        arrow.scale.z = 0.13
        self._set_navigation_marker_color(arrow, 0.1, 1.0, 0.2, 1.0)
        marker_array.markers.append(arrow)

        selected_top_z = self._target_top_height(
            self._current_box_target,
            (self._current_box_target.get("geometry") or {}).get("corners")
            if isinstance(self._current_box_target, dict)
            else None,
        )
        source_box = (
            self._current_box_target.get("box", {})
            if isinstance(self._current_box_target, dict)
            else {}
        )
        text = self._new_navigation_marker(marker_id, "yolo_navigation_text", Marker.TEXT_VIEW_FACING)
        text.pose.position.x = float(target_x)
        text.pose.position.y = float(target_y)
        text.pose.position.z = max(0.55, selected_top_z + 0.12)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.11
        self._set_navigation_marker_color(text, 1.0, 1.0, 1.0, 1.0)
        text.text = (
            f"YOLO NAV {'SKIPPED' if skipped else 'GOAL'}\n"
            f"source={self._current_target_source}\n"
            f"box_map=({box_point.x:.2f}, {box_point.y:.2f}, {box_point.z:.2f})\n"
            f"class={source_box.get('class_id', '?')} score={source_box.get('score', 0.0):.2f}\n"
            f"goal=({target_x:.2f}, {target_y:.2f}) yaw={target_yaw:.1f}deg"
        )
        marker_array.markers.append(text)

        self.navigation_visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布YOLO导航RViz标记: "
            f"topic={self.navigation_visualization_topic}, "
            f"source={self._current_target_source}, "
            f"box=({box_point.x:.3f}, {box_point.y:.3f}, {box_point.z:.3f}), "
            f"goal=({target_x:.3f}, {target_y:.3f}, {target_yaw:.3f})"
        )

    def _publish_detection_only_visualization(self):
        """在尚未生成导航目标时发布 YOLO 箱体检测结果。"""
        if self.navigation_visualization_pub is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        visualization_targets = (
            self._visualization_box_targets
            if self._visualization_box_targets
            else self._detected_box_targets
        )
        selected_id = id(self._current_box_target) if self._current_box_target is not None else None
        marker_id = 1
        for index, target in enumerate(visualization_targets):
            map_position = target.get("map_position") or {}
            corners = (target.get("geometry") or {}).get("corners") or self._fallback_box_corners(target)
            is_selected = id(target) == selected_id
            is_filtered = bool(target.get("filter_reason"))
            color = (
                (0.65, 0.65, 0.65)
                if is_filtered
                else ((1.0, 0.9, 0.0) if is_selected else (0.20, 0.75, 1.0))
            )

            outline = self._new_navigation_marker(
                marker_id, "yolo_detection_only_outline", Marker.LINE_LIST
            )
            marker_id += 1
            outline.scale.x = 0.032 if is_selected else 0.014
            self._set_navigation_marker_color(
                outline,
                color[0],
                color[1],
                color[2],
                0.35 if is_filtered else (0.95 if is_selected else 0.75),
            )
            for start_index, end_index in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_index]))
                outline.points.append(self._point_message(corners[end_index]))
            marker_array.markers.append(outline)

            top_z = self._target_top_height(target, corners)
            text_marker = self._new_navigation_marker(
                marker_id, "yolo_detection_only_text", Marker.TEXT_VIEW_FACING
            )
            marker_id += 1
            text_marker.pose.position.x = float(map_position.get("x", 0.0))
            text_marker.pose.position.y = float(map_position.get("y", 0.0))
            text_marker.pose.position.z = top_z + 0.10
            text_marker.scale.z = 0.10 if is_selected else 0.08
            self._set_navigation_marker_color(
                text_marker, color[0], color[1], color[2], 1.0
            )
            source_box = target.get("box") or {}
            filter_suffix = ""
            if is_filtered:
                filter_suffix = f" FILTERED[{target.get('filter_text') or target.get('filter_reason')}]"
            text_marker.text = (
                f"ROUGH YOLO #{index}{' SELECTED' if is_selected else ''}{filter_suffix}\n"
                f"map=({float(map_position.get('x', 0.0)):.2f}, "
                f"{float(map_position.get('y', 0.0)):.2f}, "
                f"{float(map_position.get('z', 0.0)):.2f})\n"
                f"class={source_box.get('class_id', '?')} "
                f"score={float(source_box.get('score', 0.0)):.2f}"
            )
            marker_array.markers.append(text_marker)

        self.navigation_visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布YOLO检测阶段RViz标记: "
            f"topic={self.navigation_visualization_topic}, boxes={len(visualization_targets)}, "
            f"filtered={len(self._filtered_box_targets)}"
        )

    def _clear_navigation_visualization(self):
        if self.navigation_visualization_pub is None:
            return
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.navigation_visualization_pub.publish(marker_array)

    def _new_navigation_marker(self, marker_id, namespace, marker_type):
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
    def _set_navigation_marker_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
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
        return (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )

    def _fallback_box_corners(self, target):
        map_position = target.get("map_position") or {}
        source_box = target.get("box") or {}
        size = source_box.get("size", [0.3, 0.3, 0.3])
        if len(size) != 3 or min(abs(float(value)) for value in size) < 0.01:
            size = [0.3, 0.3, 0.3]
        half = [abs(float(value)) * 0.5 for value in size]
        cx = float(map_position.get("x", 0.0))
        cy = float(map_position.get("y", 0.0))
        cz = float(map_position.get("z", 0.0))
        return [
            [cx + sx * half[0], cy + sy * half[1], cz + sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]

    @staticmethod
    def _target_top_height(target, corners=None):
        geometry = (target or {}).get("geometry") or {}
        if geometry.get("z_max") is not None:
            return float(geometry["z_max"])
        if corners:
            return max(float(corner[2]) for corner in corners)
        map_position = (target or {}).get("map_position") or {}
        return float(map_position.get("z", 0.0))

    @staticmethod
    def _ros_stamp_to_seconds(stamp):
        """把 ROS1/ROS2 时间戳转换成浮点秒。"""
        if hasattr(stamp, "secs"):
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveBoxYoloApproachToBox start: "
            f"services_key={self.services_key}, "
            f"odom_topic={self.odom_topic}, "
            f"odom_match_time_offset_sec={self.odom_match_time_offset_sec}, "
            f"odom_match_max_delta_sec={self.odom_match_max_delta_sec}, "
            f"allowed_class_ids={sorted(self.allowed_class_ids) if self.allowed_class_ids else '<all>'}, "
            f"navigation_target_key={self.navigation_target_key}, "
            f"selected_box_key={self.selected_box_key or '<disabled>'}, "
            f"selected_map_point_key={self.selected_map_point_key or '<disabled>'}"
        )

    @staticmethod
    def _format_detection_frame_summary(detection_frame):
        if not isinstance(detection_frame, dict):
            return "yolo_generation=0, yolo_stamp=0.000, raw_boxes=0"
        return (
            f"yolo_generation={int(detection_frame.get('generation', 0))}, "
            f"yolo_stamp={float(detection_frame.get('stamp', 0.0)):.3f}, "
            f"raw_boxes={len(detection_frame.get('boxes', []) or [])}"
        )
