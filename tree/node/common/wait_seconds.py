# -*- coding: utf-8 -*-
"""等待固定秒数后继续的通用节点。"""

import time

from py_trees.common import Status

from ..base import TimedMockAction


class WaitSeconds(TimedMockAction):
    """非阻塞等待指定秒数，随后返回 SUCCESS。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super(WaitSeconds, self).__init__(name, config_label, ros_node, params)
        self.duration_sec = float(params.get("duration_sec", 1.0))

    def describe_start(self):
        return f"[{self.config_label}] 等待 {self.duration_sec:.2f}s"

    def update(self):
        if self._started_at is None:
            return Status.RUNNING
        if time.monotonic() - self._started_at < self.duration_sec:
            return Status.RUNNING
        self.ros_node.get_logger().info(self.describe_finish(Status.SUCCESS))
        return Status.SUCCESS
