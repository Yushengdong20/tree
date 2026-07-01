"""使用 YOLO 箱体中心进行远距离粗靠近。"""

import math
import os
import time
import uuid
from datetime import datetime

import py_trees
from geometry_msgs.msg import PoseStamped
from py_trees.common import Status

from tree.constants import (
    BASE_LINK_FRAME,
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
from tree.utils.geometry import get_odom_pose_transformer


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
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.valid_box_map_polygon = parse_map_polygon(
            params.get("valid_box_map_polygon", [])
        )
        self.valid_box_polygon_required = self._to_bool(
            params.get("valid_box_polygon_required", False)
        )
        if self.valid_box_polygon_required and not self.valid_box_map_polygon:
            raise ValueError("valid_box_polygon_required=True 时必须配置 valid_box_map_polygon")
        self.enable_colored_log = self._to_bool(params.get("enable_colored_log", True))
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
            self.box_map_pose_pub = self.ros_node.create_publisher(
                self.box_map_pose_topic,
                PoseStamped,
                queue_size=1,
                latch=True,
            )
        # 关键步骤：YOLO 目标转 map 不直接查完整 TF 链，而是复用工具里的 odom 位姿缓存。
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
            target_frame=self.tf_target_frame,
            base_frame=self.tf_base_frame,
        )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=FINAL_POSE_KEY, access=py_trees.common.Access.WRITE)
        if self.navigation_target_key:
            self.blackboard.register_key(key=self.navigation_target_key, access=py_trees.common.Access.WRITE)
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

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def initialise(self):
        super().initialise()
        self._reset_state()
        self._clear_navigation_target_pose()
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
                services = self._get_services()
                updated = self._update_yolo_targets(services)
                self._choose_current_target_from_yolo()
                if self._current_box_target is None:
                    raise RuntimeError(f"尚未获得有效 YOLO 箱体中心: updated={updated}")

                self._box_base_position = self._current_box_target.get("base_position")
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
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] YOLO 粗靠近目标: "
                    f"箱子base坐标=({self._box_base_position['x']:.3f}, {self._box_base_position['y']:.3f}, "
                    f"{self._box_base_position['z']:.3f}), "
                    f"记忆启用={self.use_box_memory}, "
                    f"导航目标=({self._target_pose.x:.3f}, {self._target_pose.y:.3f}, {self._target_pose.yaw:.3f})"
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
                self._box_base_position = self._current_box_target.get("base_position")
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
        if self.use_box_memory:
            updated = services.yolo_detector.update_latest_target_poses()
            # 关键步骤：记忆模式只用本轮新 YOLO 结果刷新记忆，避免旧缓存覆盖箱子位置。
            target_poses = services.yolo_detector.get_latest_target_poses() if updated else []
        else:
            updated = services.yolo_detector.update_latest_target_pose()
            target_pose = services.yolo_detector.get_latest_target_pose()
            target_poses = [] if target_pose is None else [target_pose]

        self._log_info(
            "YOLO检测",
            "是否更新=%s 原始数量=%d 区域过滤启用=%s 3D重叠阈值=%.3fm"
            % (
                updated,
                len(target_poses),
                bool(self.valid_box_map_polygon),
                self.min_detected_box_3d_distance_m,
            ),
            "magenta",
        )
        self._detected_box_targets = []
        filtered_count = 0
        for index, target_pose in enumerate(target_poses):
            base_position = {
                "x": float(target_pose.pose.position.x),
                "y": float(target_pose.pose.position.y),
                "z": float(target_pose.pose.position.z),
            }
            source_frame = getattr(target_pose.header, "frame_id", "") or BASE_LINK_FRAME
            map_position = self._transform_base_position_to_map_position(
                services,
                base_position,
                source_frame,
            )
            if not self._is_map_position_allowed(map_position):
                filtered_count += 1
                self._log_info(
                    "YOLO目标过滤",
                    "序号=%d/%d 过滤类型=指定区域外 base坐标=%s map坐标=%s"
                    % (
                        index + 1,
                        len(target_poses),
                        self._format_position(base_position),
                        self._format_position(map_position),
                    ),
                    "yellow",
                )
                continue

            target = {
                "id": "",
                "base_position": base_position,
                "map_position": map_position,
            }
            overlap_target, overlap_index, overlap_distance = self._find_overlapped_detected_target(
                target
            )
            if overlap_target is not None:
                filtered_count += 1
                self._log_info(
                    "YOLO目标过滤",
                    "序号=%d/%d 过滤类型=与已有箱子3D重叠 3D距离=%.3fm "
                    "重叠对象序号=%d base坐标=%s map坐标=%s 重叠对象=%s"
                    % (
                        index + 1,
                        len(target_poses),
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
                    len(target_poses),
                    self._format_position(base_position),
                    self._format_position(map_position),
                ),
                "magenta",
            )
            self._detected_box_targets.append(target)
        self._log_info(
            "YOLO检测统计",
            "原始数量=%d 有效数量=%d 过滤数量=%d"
            % (
                len(target_poses),
                len(self._detected_box_targets),
                filtered_count,
            ),
            "magenta",
        )
        self._log_target_list("YOLO有效目标列表", self._detected_box_targets)
        return updated

    def _transform_base_position_to_map_position(self, services, base_position, source_frame):
        """优先按 source -> base_link -> odom/map 将 YOLO 点转换到 map。"""
        if self.use_tf_3d_transform:
            map_position, fallback_exc = (
                self.odom_transformer.transform_point_to_map_with_pose2d_fallback(
                    getattr(services, "tf_listener", None),
                    base_position,
                    source_frame=source_frame,
                    fallback_pose=self._current_pose,
                    timeout=self.tf_timeout_sec,
                )
            )
            if fallback_exc is not None:
                self._log_info(
                    "YOLO坐标转换",
                    "source->base->odom/map 3D转换失败，回退到2D yaw近似: %s"
                    % fallback_exc,
                    "yellow",
                )
            return map_position

        return self.odom_transformer.transform_base_point_to_map_with_pose2d(
            base_position,
            self._current_pose,
        )

    def _choose_current_target_from_yolo(self):
        """选择本轮抓取目标，并把其它 YOLO 目标写入箱子记忆。"""
        self._current_target_source = "无"
        if not self.use_box_memory:
            if self._detected_box_targets:
                self._current_box_target = self._detected_box_targets[0]
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
            self._current_box_target.setdefault("base_position", None)
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
            self._refresh_current_base_position()
            self.blackboard.set(
                self.current_box_target_key,
                self._current_box_target,
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

    def _refresh_current_base_position(self):
        """根据当前底盘位姿刷新记忆目标在 base_link 下的位置。"""
        map_position = self._current_box_target.get("map_position")
        if map_position is None:
            return

        base_position = transform_global_point_to_base(
            self._current_pose,
            float(map_position["x"]),
            float(map_position["y"]),
        )
        base_position["z"] = float(map_position.get("z", 0.0))
        self._current_box_target["base_position"] = base_position

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
            base_position = detected_target.get("base_position")
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

        target = dict(target)
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
        self._log_info(
            "YOLO当前目标",
            "来源=%s 目标=%s 有效检测数量=%d 记忆数量=%d"
            % (
                self._current_target_source,
                self._format_target(self._current_box_target),
                len(self._detected_box_targets),
                len(self._read_box_memory()),
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
        return "id=%s map坐标=%s base坐标=%s" % (
            target.get("id", ""),
            MoveBoxYoloApproachToBox._format_position(target.get("map_position")),
            MoveBoxYoloApproachToBox._format_position(target.get("base_position")),
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
            "currentBoxTarget": self._current_box_target,
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
        """发布 map 坐标系下的 YOLO 箱体中心位姿。"""
        if self.box_map_pose_pub is None:
            return

        box_pose = PoseStamped()
        box_pose.header.stamp = self.ros_node.now()
        box_pose.header.frame_id = MAP_FRAME
        box_pose.pose.position.x = self._box_global_position["x"]
        box_pose.pose.position.y = self._box_global_position["y"]
        # 关键步骤：启用 3D TF 时 z 也是 map 下高度；回退路径中 z 仍是 base_link 高度近似值。
        box_pose.pose.position.z = self._box_global_position.get("z", self._box_base_position["z"])
        # YOLO 只提供箱体中心位置，没有可靠朝向，使用单位四元数占位。
        box_pose.pose.orientation.w = 1.0
        self.box_map_pose_pub.publish(box_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布 map 下 YOLO 箱体位姿: "
            f"topic={self.box_map_pose_topic}, "
            f"position=({box_pose.pose.position.x:.3f}, "
            f"{box_pose.pose.position.y:.3f}, {box_pose.pose.position.z:.3f})"
        )

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxYoloApproachToBox start"
