"""请求抓取服务并把 multi 物体抓取结果写入 blackboard。"""

import time

import py_trees
from py_trees.common import Status

from tree.constants import (
    BASE_LINK_FRAME,
    CAMERA_FRAME,
    CHASSIS_FRAME,
    MAP_FRAME,
    ROBOT_SERVICES_KEY,
)

from ..base import TimedMockAction

from .helper.grasp_request_client import GraspRequestClient
from .helper.grasp_request_errors import NoGraspObjectError
from .helper.grasp_request_parser import GraspObjectPayloadParser


class RequestGraspPoses(GraspRequestClient, GraspObjectPayloadParser, TimedMockAction):
    """调用抓取服务，获取按距离排序后的物体抓取结果。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_url = str(
            params.get("grasp_url", "http://127.0.0.1:13603/grasp")
        ).strip()
        self.max_attempts = int(params.get("max_attempts", 3))
        self.http_timeout_sec = float(params.get("http_timeout_sec", 30.0))
        self.retry_interval_sec = float(params.get("retry_interval_sec", 1.0))
        self.request_delay_sec = float(params.get("request_delay_sec", 0.0))
        self.target_class_id = int(params.get("target_class_id", 0))
        self.grasp_mode = str(params.get("grasp_mode", "nearest")).strip().lower()
        self.no_grasp_object_status = str(
            params.get("no_grasp_object_status", "running")
        ).strip().lower()
        self.reuse_sorted_objects = self._to_bool(params.get("reuse_sorted_objects", False))
        self.distance_metric = str(params.get("distance_metric", "horizontal")).strip().lower()
        self.sorted_grasp_objects_key = str(
            params.get("sorted_grasp_objects_key", "grasp_object_sorted_grasp_objects")
        ).strip()
        self.next_grasp_object_index_key = str(
            params.get("next_grasp_object_index_key", "grasp_object_next_grasp_object_index")
        ).strip()
        self.source_frame = str(params.get("source_frame", CAMERA_FRAME)).strip()
        self.target_frame = str(params.get("target_frame", BASE_LINK_FRAME)).strip()
        self.map_frame = str(params.get("map_frame", MAP_FRAME)).strip()
        self.chassis_frame = str(params.get("chassis_frame", CHASSIS_FRAME)).strip()
        self.odom_topic = str(params.get("odom_topic", self.chassis_frame)).strip()
        self.odom_transformer = self.get_odom_pose_transformer(
            odom_topic=self.odom_topic,
            target_frame=self.map_frame,
            base_frame=BASE_LINK_FRAME,
        )
        self.tf_timeout_sec = float(params.get("tf_timeout_sec", 2.0))
        self.filter_downward_grasp_poses = self._to_bool(
            params.get("filter_downward_grasp_poses", False)
        )
        self.downward_grasp_max_angle_deg = float(
            params.get("downward_grasp_max_angle_deg", 45.0)
        )
        self.services_key = ROBOT_SERVICES_KEY
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if self.grasp_mode not in ("nearest", "multi"):
            raise ValueError("grasp_mode 必须是 nearest 或 multi")
        if self.no_grasp_object_status not in ("running", "failure"):
            raise ValueError("no_grasp_object_status 必须是 running 或 failure")
        if self.distance_metric not in ("horizontal", "xyz"):
            raise ValueError("distance_metric 必须是 horizontal 或 xyz")
        if not self.sorted_grasp_objects_key:
            raise ValueError("sorted_grasp_objects_key 不能为空")
        if not self.next_grasp_object_index_key:
            raise ValueError("next_grasp_object_index_key 不能为空")
        if not self.source_frame:
            raise ValueError("source_frame 不能为空")
        if not self.target_frame:
            raise ValueError("target_frame 不能为空")
        if not self.map_frame:
            raise ValueError("map_frame 不能为空")
        if not self.odom_topic:
            raise ValueError("odom_topic 不能为空")
        if self.downward_grasp_max_angle_deg < 0.0 or self.downward_grasp_max_angle_deg > 90.0:
            raise ValueError("downward_grasp_max_angle_deg 必须在 [0, 90] 范围内")
        self.blackboard.register_key(
            key=self.services_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.sorted_grasp_objects_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.sorted_grasp_objects_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.next_grasp_object_index_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.next_grasp_object_index_key,
            access=py_trees.common.Access.WRITE,
        )
        self._attempt_count = 0
        self._next_attempt_at = 0.0
        self._request_delay_until = None
        self._request_delay_done = False

    def initialise(self):
        super().initialise()
        self._attempt_count = 0
        self._next_attempt_at = 0.0
        self._request_delay_until = None
        self._request_delay_done = False

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        now = time.monotonic()
        if now < self._next_attempt_at:
            return Status.RUNNING

        if self.reuse_sorted_objects and self._has_unselected_cached_object():
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 复用已缓存的 multi 抓取物体列表: "
                f"key={self.sorted_grasp_objects_key}, "
                f"next_index={self.blackboard.get(self.next_grasp_object_index_key)}"
            )
            return Status.SUCCESS

        if not self._request_delay_done and self.request_delay_sec > 0.0:
            if self._request_delay_until is None:
                # 关键步骤：缓存未命中后才等待，避免复用已有物体列表时无意义地卡 1 秒。
                self._request_delay_until = now + self.request_delay_sec
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "WAIT_GRASP_REQUEST",
                    f"准备请求抓取位姿前等待 {self.request_delay_sec:.2f}s",
                )
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 缓存未命中，"
                    f"请求抓取位姿前等待 {self.request_delay_sec:.2f}s"
                )
                return Status.RUNNING
            if now < self._request_delay_until:
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "WAIT_GRASP_REQUEST",
                    f"等待抓取位姿请求延时结束: "
                    f"{self._request_delay_until - now:.2f}s",
                )
                return Status.RUNNING
            self._request_delay_done = True

        self._attempt_count += 1
        self.ros_node.set_live_runtime(
            self.config_label,
            "GRASP_REQUEST",
            f"Calling {self.grasp_url}, mode={self.grasp_mode}, "
            f"attempt {self._attempt_count}/{self.max_attempts}",
        )
        try:
            payload = self._request_grasp_payload()
            source_frame = str(payload.get("frame_id") or self.source_frame).strip()
            grasp_objects = self._build_sorted_grasp_objects(payload, source_frame)
        except Exception as exc:
            if isinstance(exc, NoGraspObjectError):
                if self.no_grasp_object_status == "failure":
                    # 关键步骤：抓放任务要求无物体时直接失败，让上层 task_manager 返回明确原因。
                    self.feedback_message = f"未检测到可抓取物体: {exc}"
                    self.ros_node.clear_live_runtime()
                    self.ros_node.get_logger().error(
                        f"[{self.config_label}] {self.feedback_message}"
                    )
                    return Status.FAILURE

                # 关键步骤：交互式等待场景仍可保持 RUNNING，让行为树停在请求节点继续等待。
                self._next_attempt_at = time.monotonic() + self.retry_interval_sec
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "WAIT_GRASP_OBJECT",
                    f"等待抓取目标: {exc}",
                )
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 未发现可抓取目标，继续等待: {exc}"
                )
                return Status.RUNNING

            if self._attempt_count >= self.max_attempts:
                self.feedback_message = str(exc)
                self.ros_node.clear_live_runtime()
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 抓取位姿请求连续失败 "
                    f"{self._attempt_count} 次: {exc}"
                )
                return Status.FAILURE

            self._next_attempt_at = time.monotonic() + self.retry_interval_sec
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 抓取位姿请求失败 "
                f"({self._attempt_count}/{self.max_attempts}): {exc}"
            )
            return Status.RUNNING

        # 关键步骤：这里只缓存按物体分组后的完整结果；当前物体的扁平 pose key 由 SelectNextGraspObject 写入。
        self.blackboard.set(self.sorted_grasp_objects_key, grasp_objects, overwrite=True)
        self.blackboard.set(self.next_grasp_object_index_key, 0, overwrite=True)
        self.ros_node.clear_live_runtime()
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已获取 {len(grasp_objects)} 个物体，"
            f"已按 {self.target_frame} 距离排序并写入 blackboard: "
            f"objects={self.sorted_grasp_objects_key}"
        )
        return Status.SUCCESS

    def _has_unselected_cached_object(self):
        if not self.blackboard.exists(self.sorted_grasp_objects_key):
            return False
        if not self.blackboard.exists(self.next_grasp_object_index_key):
            return False
        objects = self.blackboard.get(self.sorted_grasp_objects_key)
        if not isinstance(objects, list) or not objects:
            return False
        try:
            next_index = int(self.blackboard.get(self.next_grasp_object_index_key))
        except (TypeError, ValueError):
            return False
        return 0 <= next_index < len(objects)

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] RequestGraspPoses start: "
            f"url={self.grasp_url}, mode={self.grasp_mode}, "
            f"target_class_id={self.target_class_id}, max_attempts={self.max_attempts}, "
            f"request_delay_sec={self.request_delay_sec:.2f}, "
            f"no_grasp_object_status={self.no_grasp_object_status}, "
            f"objects_key={self.sorted_grasp_objects_key}, "
            f"frame={self.target_frame}/{self.map_frame}<-{self.source_frame}, "
            f"odom_topic={self.odom_topic}"
        )
