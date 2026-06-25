"""等待键盘触发，再开始一轮真实抓取流程。

默认按下键盘 `s` 后，节点才会置位抓取请求并开始响应。
"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class WaitForMoveBoxGraspRequest(TimedMockAction):
    """轮询键盘输入，在按下触发键后写入抓取请求标志。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.request_key = str(params.get("request_key", "move_box_grasp_requested")).strip()
        self.trigger_key = str(params.get("trigger_key", "s")).strip().lower() or "s"
        self.blackboard.register_key(key=self.request_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.request_key, access=py_trees.common.Access.WRITE)

    def update(self):
        """未触发时持续返回 RUNNING，触发后写入 blackboard 并返回 SUCCESS。"""
        requested = self.blackboard.get(self.request_key) if self.blackboard.exists(self.request_key) else False
        if requested:
            return Status.SUCCESS

        key = None
        if hasattr(self.ros_node, "consume_key_event"):
            key = self.ros_node.consume_key_event(self.trigger_key)
        if key is not None and key.lower() == self.trigger_key:
                self.blackboard.set(self.request_key, True, overwrite=True)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 收到 {self.trigger_key}，准备执行当前最新抓取"
                )
                return Status.SUCCESS
        return Status.RUNNING
