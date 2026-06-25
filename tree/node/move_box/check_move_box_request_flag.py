"""检查指定 move_box 请求标志是否已置位。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class CheckMoveBoxRequestFlag(TimedMockAction):
    """当请求标志为 True 时返回 SUCCESS，否则返回 FAILURE。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.request_key = str(params.get("request_key", "")).strip()
        if self.request_key:
            self.blackboard.register_key(key=self.request_key, access=py_trees.common.Access.READ)

    def describe_start(self):
        # 该节点会在空闲轮询阶段被频繁 tick，不打印 started 避免刷屏。
        return ""

    def update(self):
        if not self.request_key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 params.request_key")
            return Status.FAILURE

        requested = (
            self.blackboard.get(self.request_key)
            if self.blackboard.exists(self.request_key)
            else False
        )
        return Status.SUCCESS if requested else Status.FAILURE
