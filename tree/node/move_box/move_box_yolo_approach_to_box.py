"""使用 YOLO 箱体中心进行远距离粗靠近。"""

import math
import time
import uuid

import py_trees
from geometry_msgs.msg import PoseStamped
from py_trees.common import Status

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
    transform_base_point_to_global,
)


class MoveBoxYoloApproachToBox(TimedMockAction):
    """远距离使用 YOLO 检测箱体中心，并导航到箱体前方指定距离。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
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
        self.box_map_pose_pub = None
        if self.box_map_pose_topic:
            self.box_map_pose_pub = self.ros_node.create_publisher(
                self.box_map_pose_topic,
                PoseStamped,
                queue_size=1,
                latch=True,
            )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="flow_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="final_pose", access=py_trees.common.Access.WRITE)
        if self.navigation_target_key:
            self.blackboard.register_key(key=self.navigation_target_key, access=py_trees.common.Access.WRITE)
        self._reset_state()

    def _reset_state(self):
        self._phase = "IDLE"
        self._current_pose = None
        self._box_base_position = None
        self._box_global_position = None
        self._target_pose = None
        self._task_id = None
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = None
        self._next_poll_at = None

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def initialise(self):
        super().initialise()
        self._reset_state()
        self._phase = "GET_POSE"
        self._deadline = time.monotonic() + self.navigation_timeout_sec

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            now = time.monotonic()
            if self._phase == "FINISHED":
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
                updated = services.yolo_detector.update_latest_target_pose()
                target_pose = services.yolo_detector.get_latest_target_pose()
                if target_pose is None:
                    raise RuntimeError(f"尚未获得有效 YOLO 箱体中心: updated={updated}")

                self._box_base_position = {
                    "x": float(target_pose.pose.position.x),
                    "y": float(target_pose.pose.position.y),
                    "z": float(target_pose.pose.position.z),
                }
                self._box_global_position = transform_base_point_to_global(
                    self._current_pose,
                    self._box_base_position["x"],
                    self._box_base_position["y"],
                )
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
                self._store_navigation_target_pose(
                    self._target_pose.x,
                    self._target_pose.y,
                    self._target_pose.yaw,
                )
                self._box_distance_m = box_distance_m
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] YOLO 粗靠近目标: "
                    f"box_base=({self._box_base_position['x']:.3f}, {self._box_base_position['y']:.3f}, "
                    f"{self._box_base_position['z']:.3f}), "
                    f"target=({self._target_pose.x:.3f}, {self._target_pose.y:.3f}, {self._target_pose.yaw:.3f})"
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
                self._next_poll_at = now
                self._phase = "POLL_NAVIGATION"
                return Status.RUNNING

            if self._phase == "POLL_NAVIGATION":
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
            raise RuntimeError(f"move_box services missing on blackboard: key={self.services_key}")
        if not hasattr(services, "yolo_detector"):
            raise RuntimeError("move_box services 缺少 yolo_detector")
        return services

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

        # 关键步骤：目标一旦计算出来就写入黑板，让 FP reset 可以在导航途中提前触发。
        self.blackboard.set(
            self.navigation_target_key,
            {
                "x": x,
                "y": y,
                "yaw": yaw,
            },
            overwrite=True,
        )

    def _publish_box_map_pose(self):
        """发布 map 坐标系下的 YOLO 箱体中心位姿。"""
        if self.box_map_pose_pub is None:
            return

        box_pose = PoseStamped()
        box_pose.header.stamp = self.ros_node.now()
        box_pose.header.frame_id = "map"
        box_pose.pose.position.x = self._box_global_position["x"]
        box_pose.pose.position.y = self._box_global_position["y"]
        # 关键步骤：当前只对 x/y 做二维 map 转换，z 直接沿用 base_link 下的 YOLO 检测值。
        box_pose.pose.position.z = self._box_base_position["z"]
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
