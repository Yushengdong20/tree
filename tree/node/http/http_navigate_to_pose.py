"""导航 HTTP 节点。

这个节点把“创建导航任务 -> 轮询状态”拆成多次 tick，
从而让 viewer 能看到更细粒度的 RUNNING 过程。

当前会经历的分解动作如下：

1. CREATE_TASK
   创建底盘导航任务，拿到 task_instance_id。

2. POLL_STATUS
   周期性查询导航任务状态，直到成功或失败。
"""

import time
import uuid

from py_trees import common

from tree.constants import FLOW_RESULT_KEY
from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    DEFAULT_NAVIGATION_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    TASK_STATUS_LABELS,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_FAILED_TASK_STATUSES,
    build_chassis_config,
    extract_navigation_task_id,
    extract_task_status,
    post_chassis_navigation,
    post_navigation_task_status,
)


class HttpNavigateToPose(TimedMockAction):
    """异步导航到目标位姿。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 导航节点负责写入流程结果，供后续节点读取。
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=common.Access.WRITE)
        self.x = float(params.get("x", 0.0))
        self.y = float(params.get("y", 0.0))
        self.yaw = float(params.get("yaw", 0.0))
        self.navigation_target_key = str(params.get("navigation_target_key", "navigation_target")).strip()
        if self.navigation_target_key:
            self.blackboard.register_key(key=self.navigation_target_key, access=common.Access.WRITE)
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip(),
        )
        self.navigation_timeout_sec = int(
            params.get("navigation_timeout_sec", DEFAULT_NAVIGATION_TIMEOUT_SEC)
        )
        self.poll_interval_sec = float(params.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))

        self._phase = "IDLE"
        self._task_id = None
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = None
        self._next_poll_at = None

    def initialise(self):
        super().initialise()
        # 进入节点时只做“状态机复位”，不直接访问外部 HTTP。
        # 真正的网络调用留到 update() 中按 phase 分步推进。
        self._store_navigation_target()
        self._phase = "CREATE_TASK"
        self._task_id = str(uuid.uuid4())
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = time.monotonic() + self.navigation_timeout_sec
        self._next_poll_at = None

    def update(self):
        if self.should_use_mock_execution():
            status = self.update_mock_result()
            if status == common.Status.SUCCESS:
                self._store_navigation_result({
                    "simulated": True,
                    "targetPose": {
                        "x": self.x,
                        "y": self.y,
                        "yaw": self.yaw,
                    },
                })
            return status

        try:
            now = time.monotonic()
            if now > self._deadline:
                raise TimeoutError(
                    "wait navigation task success timeout. "
                    f"task_instance_id={self._task_instance_id} "
                    f"latest_response={self._navigation_status_response}"
                )

            if self._phase == "CREATE_TASK":
                # 第一个 tick 只负责创建导航任务，拿到 task_instance_id 后立刻返回 RUNNING。
                self._store_navigation_target()
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "NAVIGATION",
                    f"Creating navigation task for ({self.x:.3f}, {self.y:.3f}, {self.yaw:.3f})",
                )
                self._navigation_response = post_chassis_navigation(
                    self.chassis_config,
                    self._task_id,
                    self.x,
                    self.y,
                    self.yaw,
                )
                self._task_instance_id = extract_navigation_task_id(self._navigation_response)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 导航任务已创建: "
                    f"task_id={self._task_id}, task_instance_id={self._task_instance_id}"
                )
                self._phase = "POLL_STATUS"
                self._next_poll_at = now
                return common.Status.RUNNING

            if self._phase == "POLL_STATUS":
                # 后续 tick 按固定节奏轮询导航状态，这样 viewer 就能看到中间的 RUNNING 过程。
                if self._next_poll_at is not None and now < self._next_poll_at:
                    self.ros_node.set_live_runtime(
                        self.config_label,
                        "NAVIGATION",
                        f"Waiting to poll navigation status for task {self._task_instance_id}",
                    )
                    return common.Status.RUNNING

                self.ros_node.set_live_runtime(
                    self.config_label,
                    "NAVIGATION",
                    f"Polling navigation status for task {self._task_instance_id}",
                )
                self._navigation_status_response = post_navigation_task_status(
                    self.chassis_config,
                    self._task_instance_id,
                )
                task_status = extract_task_status(self._navigation_status_response)
                status_label = TASK_STATUS_LABELS.get(task_status, "未知状态")
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 导航状态: "
                    f"task_instance_id={self._task_instance_id}, "
                    f"task_status={task_status}({status_label})"
                )
                if task_status == TASK_STATUS_SUCCEEDED:
                    self._store_result()
                    self.ros_node.clear_live_runtime()
                    self.ros_node.get_logger().info(f"[{self.config_label}] 导航完成")
                    return common.Status.SUCCESS

                if task_status in TERMINAL_FAILED_TASK_STATUSES:
                    raise RuntimeError(f"navigation task failed: {self._navigation_status_response}")

                self._next_poll_at = now + self.poll_interval_sec
                return common.Status.RUNNING

            raise RuntimeError(f"unsupported navigation phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] navigation failed: {exc}")
            return common.Status.FAILURE

    def _store_result(self):
        result = {
            "taskId": self._task_id,
            "taskInstanceId": self._task_instance_id,
            "navigation": self._navigation_response,
            "navigationStatus": self._navigation_status_response,
        }
        self._store_navigation_result(result)

    def _store_navigation_result(self, navigation_result):
        """只更新流程结果中的 navigation 字段，避免覆盖其它节点结果。"""
        try:
            flow_result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            flow_result = None
        if flow_result is None:
            flow_result = {}

        flow_result["navigation"] = navigation_result
        self.blackboard.flow_result = flow_result

    def _store_navigation_target(self):
        """把当前导航目标写入黑板，供并行监控节点读取。"""
        if not self.navigation_target_key:
            return

        self.blackboard.set(
            self.navigation_target_key,
            {
                "x": self.x,
                "y": self.y,
                "yaw": self.yaw,
            },
            overwrite=True,
        )

    def describe_start(self):
        return (
            f"[{self.config_label}] HttpNavigateToPose start: "
            f"x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f}, "
            f"navigation_target_key={self.navigation_target_key or '<disabled>'}"
        )
