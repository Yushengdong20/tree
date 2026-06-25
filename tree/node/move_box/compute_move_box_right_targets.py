"""在左手拉出箱体后，计算右手需要使用的补偿目标点。"""

import numpy as np
import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxRightTargets(TimedMockAction):
    """把右手流程用到的补偿后目标点写入 blackboard。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.right_edge_key = str(params.get("right_edge_key", "move_box_right_edge_point")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.left_pull_key = str(params.get("left_pull_key", "move_box_left_pull_target")).strip()
        self.target_keys = {
            "moved_right_edge": str(params.get("moved_right_edge_key", "move_box_moved_right_edge")).strip(),
            "right_above": str(params.get("right_above_key", "move_box_right_above_edge")).strip(),
            "right_below": str(params.get("right_below_key", "move_box_right_below_edge")).strip(),
            "right_lift": str(params.get("right_lift_key", "move_box_right_lift_target")).strip(),
            "return_center_left": str(
                params.get("return_center_left_key", "move_box_return_center_left_target")
            ).strip(),
            "return_center_right": str(
                params.get("return_center_right_key", "move_box_return_center_right_target")
            ).strip(),
            "lower_left": str(params.get("lower_left_key", "move_box_lower_left_target")).strip(),
            "lower_right": str(params.get("lower_right_key", "move_box_lower_right_target")).strip(),
        }
        self.blackboard.register_key(key=self.right_edge_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.left_pull_key, access=py_trees.common.Access.READ)
        for key in self.target_keys.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        """根据右边缘原始位置和左手拉出的位移，计算右手目标点。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        right_edge_point = self.blackboard.get(self.right_edge_key) if self.blackboard.exists(self.right_edge_key) else None
        box_axes = self.blackboard.get(self.box_axes_key) if self.blackboard.exists(self.box_axes_key) else None
        left_pull_target = self.blackboard.get(self.left_pull_key) if self.blackboard.exists(self.left_pull_key) else None
        if right_edge_point is None or box_axes is None or left_pull_target is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少右手目标计算所需抓取数据")
            return Status.FAILURE

        up_axis = np.array(box_axes["up"], dtype=float)
        left_axis = np.array(box_axes["left"], dtype=float)
        approach_offset = float(self.ros_node.get_param("left_approach_offset", 0.1))
        descend_below_offset = float(self.ros_node.get_param("left_descend_below_offset", 0.01))
        lift_offset = float(self.ros_node.get_param("left_lift_offset", 0.1))
        pull_left_offset = float(self.ros_node.get_param("left_pull_left_offset", 0.15))

        moved_right_edge = right_edge_point + left_axis * pull_left_offset
        above_right_edge = moved_right_edge + up_axis * approach_offset
        below_right_edge = moved_right_edge - up_axis * descend_below_offset
        right_lift_target = below_right_edge + up_axis * (descend_below_offset + lift_offset)
        return_center_offset = -left_axis * pull_left_offset
        return_center_left_target = left_pull_target + return_center_offset
        return_center_right_target = right_lift_target + return_center_offset
        lower_left_target = return_center_left_target - up_axis * lift_offset
        lower_right_target = return_center_right_target - up_axis * lift_offset

        self.blackboard.set(self.target_keys["moved_right_edge"], moved_right_edge, overwrite=True)
        self.blackboard.set(self.target_keys["right_above"], above_right_edge, overwrite=True)
        self.blackboard.set(self.target_keys["right_below"], below_right_edge, overwrite=True)
        self.blackboard.set(self.target_keys["right_lift"], right_lift_target, overwrite=True)
        self.blackboard.set(
            self.target_keys["return_center_left"],
            return_center_left_target,
            overwrite=True,
        )
        self.blackboard.set(
            self.target_keys["return_center_right"],
            return_center_right_target,
            overwrite=True,
        )
        self.blackboard.set(self.target_keys["lower_left"], lower_left_target, overwrite=True)
        self.blackboard.set(
            self.target_keys["lower_right"],
            lower_right_target,
            overwrite=True,
        )
        return Status.SUCCESS
