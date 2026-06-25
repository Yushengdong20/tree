"""抓取前对正 HTTP 节点。

这个节点是当前业务树里最复杂的叶子节点之一，它会把真实业务中的：
- 取当前位姿
- 调 detect_box
- 判断是否已经对正
- 必要时发起微调导航

拆成多次 tick 可观测的 phase 状态机。

当前会经历的分解动作如下：

1. DISABLED_POSE
   如果配置里关闭了对正，只读取当前位姿并直接结束。

2. GET_POSE
   读取当前底盘位姿，作为本轮检测与修正的起点。

3. DETECT_BOX
   调用 detect_box，拿到箱体中心和朝向向量。

4. EVALUATE_ALIGNMENT
   把检测结果转换成横向/纵向/角度误差，判断是否已经对正。

5. ADJUST_NAVIGATION_CREATE
   如未对正，则根据误差计算修正位姿并创建一次微调导航任务。

6. ADJUST_NAVIGATION_POLL
   周期性查询这次微调导航状态，成功后回到下一轮 GET_POSE。
"""

import time
import uuid

from py_trees import common

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    DEFAULT_GRAB_URL,
    DEFAULT_NAVIGATION_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_FAILED_TASK_STATUSES,
    build_align_config,
    build_chassis_config,
    build_grab_config,
    extract_box_alignment_measurement,
    extract_navigation_task_id,
    extract_task_status,
    get_chassis_current_pose,
    is_box_aligned,
    make_box_adjust_pose,
    post_chassis_navigation,
    post_detect_box,
    post_navigation_task_status,
)


class HttpAlignWithBox(TimedMockAction):
    """异步检测并对正箱体。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 对正节点会读取/追加流程结果，并写入最终位姿。
        self.blackboard.register_key(key="flow_result", access=common.Access.READ)
        self.blackboard.register_key(key="flow_result", access=common.Access.WRITE)
        self.blackboard.register_key(key="final_pose", access=common.Access.WRITE)
        chassis_url = str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip()
        grab_url = str(params.get("grab_url", DEFAULT_GRAB_URL)).strip()
        self.chassis_config = build_chassis_config(base_url=chassis_url)
        self.grab_config = build_grab_config(
            base_url=grab_url,
            arm=params.get("arm"),
        )
        self.align_config = build_align_config(
            enabled=self._to_bool(params.get("enabled", True)),
            tolerance_m=float(params.get("tolerance_m", 0.06)),
            target_distance_m=float(params.get("target_distance_m", 0.4)),
            longitudinal_tolerance_m=float(params.get("longitudinal_tolerance_m", 0.06)),
            yaw_tolerance_deg=float(params.get("yaw_tolerance_deg", 5.0)),
            max_attempts=int(params.get("max_attempts", 3)),
        )
        self.navigation_timeout_sec = int(
            params.get("navigation_timeout_sec", DEFAULT_NAVIGATION_TIMEOUT_SEC)
        )
        self.poll_interval_sec = float(params.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))

        self._phase = "IDLE"
        self._attempt_index = 0
        self._current_pose = None
        self._detection_response = None
        self._measurement = None
        self._pending_navigation = None

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def initialise(self):
        super().initialise()
        # 这类“对正”节点本质上是一个小状态机：
        # 取位姿 -> 检测 -> 判断 -> 必要时发起微调导航 -> 再次检测。
        self._phase = "GET_POSE" if self.align_config.enabled else "DISABLED_POSE"
        self._attempt_index = 0
        self._current_pose = None
        self._detection_response = None
        self._measurement = None
        self._pending_navigation = None
        result = self.blackboard.flow_result
        if self.align_config.enabled:
            result["boxAlignment"] = []

    def update(self):
        if self.should_use_mock_execution():
            status = self.update_mock_result()
            if status == common.Status.SUCCESS:
                result = self.blackboard.flow_result
                result["alignment"] = {
                    "simulated": True,
                    "enabled": self.align_config.enabled,
                    "max_attempts": self.align_config.max_attempts,
                }
                self.blackboard.final_pose = {
                    "x": result.get("navigation", {}).get("x", 0.0),
                    "y": result.get("navigation", {}).get("y", 0.0),
                    "yaw": result.get("navigation", {}).get("yaw", 0.0),
                }
            return status

        try:
            if self._phase == "DISABLED_POSE":
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "ALIGNMENT",
                    "Alignment disabled, fetching current pose only",
                )
                pose = get_chassis_current_pose(self.chassis_config)
                self.blackboard.final_pose = {
                    "x": pose.x,
                    "y": pose.y,
                    "yaw": pose.yaw,
                }
                self.ros_node.clear_live_runtime()
                return common.Status.SUCCESS

            if self._phase == "GET_POSE":
                # 每轮尝试都从当前底盘位姿开始，确保后面的误差计算基于最新位置。
                self._attempt_index += 1
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "ALIGNMENT",
                    f"Attempt {self._attempt_index}/{self.align_config.max_attempts}: fetching chassis pose",
                )
                self._current_pose = get_chassis_current_pose(self.chassis_config)
                self._phase = "DETECT_BOX"
                return common.Status.RUNNING

            if self._phase == "DETECT_BOX":
                # 检测请求和误差解析拆成单独 tick，方便 viewer 看到当前正在“看箱子”。
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "ALIGNMENT",
                    f"Attempt {self._attempt_index}/{self.align_config.max_attempts}: calling detect_box",
                )
                self._detection_response = post_detect_box(self.grab_config)
                self._measurement = extract_box_alignment_measurement(
                    self._detection_response,
                    self.align_config,
                )
                self._phase = "EVALUATE_ALIGNMENT"
                return common.Status.RUNNING

            if self._phase == "EVALUATE_ALIGNMENT":
                # 这一阶段不打外部接口，只根据上一步检测结果判断是否已经满足抓取姿态。
                aligned = is_box_aligned(self._measurement, self.align_config)
                result = self.blackboard.flow_result
                result["boxAlignment"].append(
                    {
                        "attempt": self._attempt_index,
                        "pose": {
                            "x": self._current_pose.x,
                            "y": self._current_pose.y,
                            "yaw": self._current_pose.yaw,
                        },
                        "lateralErrorM": self._measurement.lateral_error_m,
                        "longitudinalErrorM": self._measurement.longitudinal_error_m,
                        "yawErrorDeg": self._measurement.yaw_error_deg,
                        "aligned": aligned,
                        "detection": self._detection_response,
                    }
                )
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "ALIGNMENT",
                    "Attempt "
                    f"{self._attempt_index}: "
                    f"lateral={self._measurement.lateral_error_m:.3f}, "
                    f"longitudinal={self._measurement.longitudinal_error_m:.3f}, "
                    f"yaw={self._measurement.yaw_error_deg:.3f}, "
                    f"aligned={aligned}",
                )
                if aligned:
                    self.blackboard.final_pose = {
                        "x": self._current_pose.x,
                        "y": self._current_pose.y,
                        "yaw": self._current_pose.yaw,
                    }
                    self.ros_node.clear_live_runtime()
                    return common.Status.SUCCESS

                if self._attempt_index >= self.align_config.max_attempts:
                    raise RuntimeError(
                        "box alignment failed. "
                        f"max_attempts={self.align_config.max_attempts} "
                        f"lateral_tolerance_m={self.align_config.tolerance_m} "
                        f"longitudinal_tolerance_m={self.align_config.longitudinal_tolerance_m} "
                        f"yaw_tolerance_deg={self.align_config.yaw_tolerance_deg}"
                    )

                next_pose = make_box_adjust_pose(
                    self._current_pose,
                    self._measurement,
                    self.align_config,
                )
                result["boxAlignment"][-1]["adjustedPose"] = {
                    "x": next_pose.x,
                    "y": next_pose.y,
                    "yaw": next_pose.yaw,
                }
                self._pending_navigation = _NavigationSubtask(
                    chassis_config=self.chassis_config,
                    x=next_pose.x,
                    y=next_pose.y,
                    yaw=next_pose.yaw,
                    timeout_sec=self.navigation_timeout_sec,
                    poll_interval_sec=self.poll_interval_sec,
                )
                self._phase = "ADJUST_NAVIGATION_CREATE"
                return common.Status.RUNNING

            if self._phase.startswith("ADJUST_NAVIGATION"):
                # 微调导航本身也拆成 create/poll 两段，避免把整个导航过程塞进一个 tick。
                status = self._tick_adjust_navigation()
                if status is None:
                    return common.Status.RUNNING

                self.blackboard.flow_result["boxAlignment"][-1]["adjustNavigation"] = status
                self._phase = "GET_POSE"
                return common.Status.RUNNING

            raise RuntimeError(f"unsupported alignment phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] align failed: {exc}")
            return common.Status.FAILURE

    def _tick_adjust_navigation(self):
        task = self._pending_navigation
        now = time.monotonic()
        if now > task.deadline:
            raise TimeoutError(
                "wait navigation task success timeout. "
                f"task_instance_id={task.task_instance_id} "
                f"latest_response={task.navigation_status_response}"
            )

        if self._phase == "ADJUST_NAVIGATION_CREATE":
            self.ros_node.set_live_runtime(
                self.config_label,
                "ALIGNMENT",
                "Creating adjustment navigation to "
                f"({task.x:.3f}, {task.y:.3f}, {task.yaw:.3f})",
            )
            task.navigation_response = post_chassis_navigation(
                task.chassis_config,
                task.task_id,
                task.x,
                task.y,
                task.yaw,
                coarse=False,
            )
            task.task_instance_id = extract_navigation_task_id(task.navigation_response)
            task.next_poll_at = now
            self._phase = "ADJUST_NAVIGATION_POLL"
            return None

        if self._phase == "ADJUST_NAVIGATION_POLL":
            if task.next_poll_at is not None and now < task.next_poll_at:
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "ALIGNMENT",
                    f"Waiting to poll adjustment navigation {task.task_instance_id}",
                )
                return None

            self.ros_node.set_live_runtime(
                self.config_label,
                "ALIGNMENT",
                f"Polling adjustment navigation {task.task_instance_id}",
            )
            task.navigation_status_response = post_navigation_task_status(
                task.chassis_config,
                task.task_instance_id,
            )
            task_status = extract_task_status(task.navigation_status_response)
            if task_status == TASK_STATUS_SUCCEEDED:
                result = {
                    "taskId": task.task_id,
                    "taskInstanceId": task.task_instance_id,
                    "navigation": task.navigation_response,
                    "navigationStatus": task.navigation_status_response,
                }
                self._pending_navigation = None
                return result
            if task_status in TERMINAL_FAILED_TASK_STATUSES:
                raise RuntimeError(f"navigation task failed: {task.navigation_status_response}")

            task.next_poll_at = now + task.poll_interval_sec
            return None

        raise RuntimeError(f"unsupported adjustment navigation phase: {self._phase}")

    def describe_start(self):
        return f"[{self.config_label}] HttpAlignWithBox start"


class _NavigationSubtask:
    def __init__(self, chassis_config, x: float, y: float, yaw: float, timeout_sec: int, poll_interval_sec: float):
        self.chassis_config = chassis_config
        self.x = x
        self.y = y
        self.yaw = yaw
        self.task_id = str(uuid.uuid4())
        self.task_instance_id = None
        self.navigation_response = None
        self.navigation_status_response = None
        self.deadline = time.monotonic() + timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.next_poll_at = None
