"""切换 FoundationPose 当前识别模型。"""

import time

import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY
from tree.utils.chassis_navigation import post_json

from ..base import TimedMockAction


DEFAULT_FOUNDATIONPOSE_BASE_URL = "http://192.168.26.12:13604"


class SwitchFoundationPoseTarget(TimedMockAction):
    """通过 HTTP 切换 FP 的 mesh，并记录当前业务目标类型。

    例如拆垛前切到 ``tuopan`` 估计托盘；完成托盘区域标定后再切回
    ``large_green``，供正常箱体检测与抓取使用。该节点只切换模型，是否 reset
    和等待新帧由后续 ``EnsureMoveBoxDetectionReady`` 明确负责，避免隐式重启。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.base_url = str(
            params.get("foundationpose_base_url", DEFAULT_FOUNDATIONPOSE_BASE_URL)
        ).rstrip("/")
        self.mesh_name = str(params.get("mesh_name", "")).strip()
        self.timeout_sec = float(params.get("timeout_sec", 5.0))
        # ``switch_target`` 在 FP 正处理/清空上一目标的队列时可能短暂阻塞。
        # 该接口对同一 mesh 的重复请求是幂等的，因此把失败后的重试做成按 tick
        # 推进的状态机，避免一次网络抖动让整棵树重新从头执行。
        self.max_retries = max(0, int(params.get("max_retries", 0)))
        self.retry_interval_sec = max(0.0, float(params.get("retry_interval_sec", 1.0)))
        self.target_key = str(
            params.get("target_key", "move_box_foundationpose_active_target")
        ).strip()
        if not self.mesh_name:
            raise ValueError("SwitchFoundationPoseTarget 必须配置 mesh_name")
        self.blackboard.register_key(key=ROBOT_SERVICES_KEY, access=py_trees.common.Access.READ)
        if self.target_key:
            self.blackboard.register_key(key=self.target_key, access=py_trees.common.Access.WRITE)
        self._attempt_count = 0
        self._next_attempt_at = None

    def initialise(self):
        super().initialise()
        self._attempt_count = 0
        self._next_attempt_at = None

    def update(self):
        if self.should_use_mock_execution():
            if self.target_key:
                self.blackboard.set(self.target_key, self.mesh_name, overwrite=True)
            return self.update_mock_result()

        now = time.monotonic()
        if self._next_attempt_at is not None and now < self._next_attempt_at:
            remaining = self._next_attempt_at - now
            self.feedback_message = f"等待 FP 模型切换重试: {remaining:.1f}s"
            return Status.RUNNING

        url = f"{self.base_url}/foundationpose/switch_target"
        self._attempt_count += 1
        try:
            result = post_json(
                url,
                {"mesh_name": self.mesh_name},
                self.timeout_sec,
                "foundationpose_switch_target",
            )
        except Exception as exc:
            self.feedback_message = str(exc)
            retries_used = self._attempt_count - 1
            if retries_used < self.max_retries:
                self._next_attempt_at = now + self.retry_interval_sec
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] FP 切换识别目标请求失败，"
                    f"将在 {self.retry_interval_sec:.1f}s 后重试: "
                    f"mesh={self.mesh_name}, attempt={self._attempt_count}/"
                    f"{self.max_retries + 1}, error={exc}"
                )
                return Status.RUNNING
            self.ros_node.get_logger().error(
                f"[{self.config_label}] FP 切换识别目标失败: mesh={self.mesh_name}, "
                f"url={url}, attempts={self._attempt_count}, error={exc}"
            )
            return Status.FAILURE
        if self.target_key:
            self.blackboard.set(self.target_key, self.mesh_name, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] FP 已切换识别目标: mesh={self.mesh_name}, "
            f"url={url}, attempt={self._attempt_count}, response={result}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] SwitchFoundationPoseTarget start: "
            f"mesh={self.mesh_name}, base_url={self.base_url}, "
            f"timeout={self.timeout_sec:.1f}s, max_retries={self.max_retries}, "
            f"retry_interval={self.retry_interval_sec:.1f}s"
        )
