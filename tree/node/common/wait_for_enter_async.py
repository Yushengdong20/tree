# -*- coding: utf-8 -*-
"""WaitForEnterAsync 节点。

默认需要在终端按下 `Enter`，后台等待线程才会将节点推进到 SUCCESS。
"""

import threading

from py_trees.behaviour import Behaviour
from py_trees.common import Status


class WaitForEnterAsync(Behaviour):
    """后台等待终端按下 Enter；未按下前持续 RUNNING，按下后返回 SUCCESS。"""

    def __init__(self, name, config_label, ros_node, params):
        super(WaitForEnterAsync, self).__init__(name=name)
        self.config_label = config_label
        self.ros_node = ros_node
        self.params = params
        self.message = str(params.get("message", "按 Enter 停止循环并退出行为树..."))
        self._enter_pressed = False
        self._thread = None
        self._thread_started = False

    def initialise(self):
        if self._enter_pressed:
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._wait_for_input, daemon=True)
            self._thread.start()
        if not self._thread_started:
            self.ros_node.get_logger().info(f"[{self.config_label}] {self.message}")
            self._thread_started = True

    def update(self):
        if self._enter_pressed:
            self.ros_node.get_logger().info(f"[{self.config_label}] detected Enter, stopping tree")
            return Status.SUCCESS
        return Status.RUNNING

    def _wait_for_input(self):
        try:
            input()
        except EOFError:
            pass
        self._enter_pressed = True
