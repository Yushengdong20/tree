"""检查检测器是否已经生成可用的抓取点和箱体方向。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class CheckMoveBoxGraspDataReady(TimedMockAction):
    """在抓取数据不完整时尽早失败，避免后续动作节点误执行。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(
            params.get("box_center_key", "move_box_latest_box_center")
        ).strip()
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)

    def update(self):
        """检查 blackboard 中的抓取点和箱体方向是否都已就绪。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        grasp_pair = self.blackboard.get(self.grasp_pair_key) if self.blackboard.exists(self.grasp_pair_key) else None
        box_axes = self.blackboard.get(self.box_axes_key) if self.blackboard.exists(self.box_axes_key) else None
        box_center = self.blackboard.get(self.box_center_key) if self.blackboard.exists(self.box_center_key) else None
        if grasp_pair is None or box_axes is None or box_center is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 尚未获得有效抓取数据: "
                f"grasp_pair={grasp_pair is not None}, "
                f"box_axes={box_axes is not None}, "
                f"box_center={box_center is not None}"
            )
            return Status.FAILURE
        return Status.SUCCESS
