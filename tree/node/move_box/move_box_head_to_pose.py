"""发布一次头部 yaw/pitch 目标，用于流程阶段之间的头部回正。"""

import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class MoveBoxHeadToPose(TimedMockAction):
    """让头部移动到 JSON 配置的固定角度。

    这个节点是一次性动作节点，不会持续跟踪目标点：
    - yaw/pitch 单位为 deg。
    - 默认 yaw=0, pitch=20，与 EnsureMoveBoxServices 的初始头部姿态保持一致。
    - 适合放箱后结束“盯传送带”，让头部回到下一轮抓箱前的观测姿态。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.yaw = float(params.get("yaw", 0.0))
        self.pitch = float(params.get("pitch", 20.0))
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if self.should_skip_head_motion():
            self.log_skip_head_motion()
            return Status.SUCCESS

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "head_controller"):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] services 或 head_controller 缺失: key={self.services_key}"
            )
            return Status.FAILURE

        ok = services.head_controller.set_head_target(self.yaw, self.pitch)
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveBoxHeadToPose start: "
            f"yaw={self.yaw:.1f}, pitch={self.pitch:.1f}"
        )
