# -*- coding: utf-8 -*-
"""WaitForEnter 节点。

默认需要在终端按下 `Enter`，节点才会继续返回 SUCCESS。
"""

from py_trees.common import Status

from ..base import TimedMockAction


class WaitForEnter(TimedMockAction):
    """阻塞等待终端按下 Enter，随后返回 SUCCESS。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super(WaitForEnter, self).__init__(name, config_label, ros_node, params)
        self.message = str(params.get("message", "按 Enter 继续..."))

    def describe_start(self):
        return f"[{self.config_label}] waiting for Enter: {self.message}"

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if hasattr(self.ros_node, "consume_key_event"):
            key = self.ros_node.consume_key_event("\n", "\r")
            if key is None:
                return Status.RUNNING
        else:
            try:
                input(self.message)
            except EOFError:
                # 无交互终端时，按“已经收到继续信号”处理，避免节点永久卡住。
                pass

        self.ros_node.get_logger().info(self.describe_finish(Status.SUCCESS))
        return Status.SUCCESS
