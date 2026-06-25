"""使用启动时保存的 YOLO 箱体 global 坐标进行远距离粗靠近。"""

import math
import time
import uuid

import py_trees
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
)


class MoveBoxStoredYoloApproachToBox(TimedMockAction):
    """读取 blackboard 中的箱体 global 坐标，并导航到箱体前方指定距离。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.current_target_key = str(params.get("current_target_key", "move_box_current_yolo_target")).strip()
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
        self.blackboard.register_key(key=self.current_target_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="flow_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="final_pose", access=py_trees.common.Access.WRITE)
        if self.navigation_target_key:
            self.blackboard.register_key(key=self.navigation_target_key, access=py_trees.common.Access.WRITE)
        self._reset_state()

    def _reset_state(self):
        self._phase = "IDLE"
        self._current_target = None
        self._current_pose = None
        self._box_base_position = None
        self._box_global_position = None
        self._box_distance_m = None
        self._target_pose = None
        self._task_id = None
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = None
        self._next_poll_at = None

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
            if now > self._deadline:
                raise TimeoutError(
                    f"保存目标 YOLO 靠近导航超时: task_instance_id={self._task_instance_id}, "
                    f"latest_response={self._navigation_status_response}"
                )

            if self._phase == "GET_POSE":
                self.ros_node.set_live_runtime(self.config_label, "YOLO_APPROACH", "读取底盘当前位姿")
                self._current_pose = get_chassis_current_pose(self.chassis_config)
                self._phase = "READ_STORED_TARGET"
                return Status.RUNNING

            if self._phase == "READ_STORED_TARGET":
                self.ros_node.set_live_runtime(self.config_label, "YOLO_APPROACH", "读取已保存 YOLO 箱体目标")
                self._current_target = self._get_current_target()
                self._box_global_position = self._get_global_position(self._current_target)
                self._box_base_position = self._current_target.get("basePositionAtSnapshot")
                self._box_distance_m = math.hypot(
                    self._box_global_position["x"] - self._current_pose.x,
                    self._box_global_position["y"] - self._current_pose.y,
                )
                if self._box_distance_m <= self.target_distance_m:
                    self.ros_node.get_logger().info(
                        f"[{self.config_label}] 保存箱体距离 {self._box_distance_m:.3f}m "
                        f"<= 目标距离 {self.target_distance_m:.3f}m，跳过粗导航"
                    )
                    self._store_navigation_target_pose(
                        self._current_pose.x,
                        self._current_pose.y,
                        self._current_pose.yaw,
                    )
                    self._store_result(need_navigation=False)
                    self.blackboard.final_pose = {
                        "x": self._current_pose.x,
                        "y": self._current_pose.y,
                        "yaw": self._current_pose.yaw,
                    }
                    self.ros_node.clear_live_runtime()
                    return Status.SUCCESS

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
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 保存 YOLO 目标粗靠近: "
                    f"id={self._current_target.get('id')}, "
                    f"box_global=({self._box_global_position['x']:.3f}, "
                    f"{self._box_global_position['y']:.3f}), "
                    f"target=({self._target_pose.x:.3f}, {self._target_pose.y:.3f}, "
                    f"{self._target_pose.yaw:.3f})"
                )
                self._phase = "CREATE_NAVIGATION"
                return Status.RUNNING

            if self._phase == "CREATE_NAVIGATION":
                self._task_id = str(uuid.uuid4())
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "YOLO_APPROACH",
                    f"创建保存目标 YOLO 粗靠近导航: ({self._target_pose.x:.3f}, "
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
                    f"[{self.config_label}] 保存目标 YOLO 粗靠近导航状态: "
                    f"task_instance_id={self._task_instance_id}, task_status={task_status}({status_label})"
                )
                if task_status == TASK_STATUS_SUCCEEDED:
                    return self._finish_success()
                if task_status in TERMINAL_FAILED_TASK_STATUSES:
                    raise RuntimeError(f"保存目标 YOLO 粗靠近导航失败: {self._navigation_status_response}")
                self._next_poll_at = now + self.poll_interval_sec
                return Status.RUNNING

            raise RuntimeError(f"unsupported stored YOLO approach phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] 保存目标 YOLO 粗靠近失败: {exc}")
            return Status.FAILURE

    def _get_current_target(self):
        target = self.blackboard.get(self.current_target_key) if self.blackboard.exists(self.current_target_key) else None
        if not isinstance(target, dict):
            raise RuntimeError(f"当前 YOLO 目标不存在或格式错误: key={self.current_target_key}")
        return target

    @staticmethod
    def _get_global_position(target):
        """解析当前目标中的 global 坐标。"""
        global_position = target.get("globalPosition")
        if not isinstance(global_position, dict):
            raise RuntimeError(f"当前 YOLO 目标缺少 globalPosition: {target}")
        if "x" not in global_position or "y" not in global_position:
            raise RuntimeError(f"当前 YOLO 目标 globalPosition 缺少 x/y: {global_position}")
        return {
            "x": float(global_position["x"]),
            "y": float(global_position["y"]),
            "z": float(global_position.get("z", 0.0)),
        }

    def _finish_success(self):
        self._store_result(need_navigation=True)
        self.blackboard.final_pose = {
            "x": self._target_pose.x,
            "y": self._target_pose.y,
            "yaw": self._target_pose.yaw,
        }
        self.ros_node.clear_live_runtime()
        return Status.SUCCESS

    def _store_result(self, need_navigation):
        try:
            flow_result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            flow_result = None
        if flow_result is None:
            flow_result = {}

        flow_result["yoloApproach"] = {
            "source": "storedSnapshot",
            "needNavigation": need_navigation,
            "currentTarget": self._current_target,
            "currentPose": {
                "x": self._current_pose.x,
                "y": self._current_pose.y,
                "yaw": self._current_pose.yaw,
            },
            "boxBasePosition": self._box_base_position,
            "boxGlobalPosition": self._box_global_position,
            "boxDistanceM": self._box_distance_m,
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

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxStoredYoloApproachToBox start"
