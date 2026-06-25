"""在一轮抓取结束后清除抓取请求标志。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ClearMoveBoxGraspRequest(TimedMockAction):
    """重置共享抓取触发标志，让重复循环重新进入等待状态。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.request_key = str(params.get("request_key", "move_box_grasp_requested")).strip()
        self.blackboard.register_key(key=self.request_key, access=py_trees.common.Access.WRITE)

    def update(self):
        """把抓取请求标志写回 False。"""
        self.blackboard.set(self.request_key, False, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已清除请求标志: {self.request_key} -> False"
        )
        return Status.SUCCESS
