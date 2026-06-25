"""轮询键盘输入，并把抓取/复位请求写入 blackboard。

默认按下键盘 `s` 会置位抓取请求，`p` 会置位放置请求，`o` 会置位复位请求。
"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class PollMoveBoxKeyboardRequests(TimedMockAction):
    """非阻塞监听键盘，把 s/p/o 分别转换成抓取、放置和复位请求。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_request_key = str(params.get("grasp_request_key", "move_box_grasp_requested")).strip()
        self.release_request_key = str(
            params.get("release_request_key", "move_box_release_requested")
        ).strip()
        self.place_request_key = str(
            params.get("place_request_key", "move_box_place_requested")
        ).strip()
        self.grasp_trigger_key = str(params.get("grasp_trigger_key", "s")).strip().lower() or "s"
        self.place_trigger_key = str(params.get("place_trigger_key", "p")).strip().lower() or "p"
        self.release_trigger_key = (
            str(params.get("release_trigger_key", "o")).strip().lower() or "o"
        )
        self.blackboard.register_key(key=self.grasp_request_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.place_request_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key=self.release_request_key, access=py_trees.common.Access.WRITE
        )

    def update(self):
        if hasattr(self.ros_node, "consume_key_event"):
            key = self.ros_node.consume_key_event(
                self.grasp_trigger_key,
                self.place_trigger_key,
                self.release_trigger_key,
            )
            if key is not None:
                key = key.lower()
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 已消费按键: {key}"
                )
            else:
                key = None
        else:
            key = None

        if key is not None:
            if key == self.grasp_trigger_key:
                self.blackboard.set(self.grasp_request_key, True, overwrite=True)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 收到 {self.grasp_trigger_key}，置位抓取请求"
                )
            elif key == self.place_trigger_key:
                self.blackboard.set(self.place_request_key, True, overwrite=True)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 收到 {self.place_trigger_key}，置位放置请求"
                )
            elif key == self.release_trigger_key:
                self.blackboard.set(self.release_request_key, True, overwrite=True)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 收到 {self.release_trigger_key}，置位复位请求"
                )
        return Status.RUNNING
