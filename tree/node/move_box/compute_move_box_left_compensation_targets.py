"""根据右手外拉结果计算左手补抓以及双手回中上提目标。"""

import numpy as np
import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxLeftCompensationTargets(TimedMockAction):
    """生成左手补偿目标以及后续双手回中、搬运高度和下降目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.left_edge_key = str(
            params.get("left_edge_key", "move_box_left_edge_point")
        ).strip()
        self.box_axes_key = str(
            params.get("box_axes_key", "move_box_latest_box_axes")
        ).strip()
        self.right_pull_key = str(
            params.get("right_pull_key", "move_box_right_pull_target")
        ).strip()
        self.target_keys = {
            "moved_left_edge": str(
                params.get("moved_left_edge_key", "move_box_moved_left_edge")
            ).strip(),
            "left_above": str(
                params.get("left_above_key", "move_box_left_compensation_above_edge")
            ).strip(),
            "left_below": str(
                params.get("left_below_key", "move_box_left_compensation_below_edge")
            ).strip(),
            "left_lift": str(
                params.get("left_lift_key", "move_box_left_compensation_lift_target")
            ).strip(),
            "return_center_left": str(
                params.get("return_center_left_key", "move_box_return_center_left_target")
            ).strip(),
            "return_center_right": str(
                params.get("return_center_right_key", "move_box_return_center_right_target")
            ).strip(),
            "lower_left": str(params.get("lower_left_key", "move_box_lower_left_target")).strip(),
            "lower_right": str(params.get("lower_right_key", "move_box_lower_right_target")).strip(),
        }
        self.blackboard.register_key(key=self.left_edge_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.right_pull_key, access=py_trees.common.Access.READ)
        for key in self.target_keys.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        """按右手外拉位移补偿左侧箱体边缘并生成后续目标。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        left_edge_point = (
            self.blackboard.get(self.left_edge_key)
            if self.blackboard.exists(self.left_edge_key)
            else None
        )
        box_axes = (
            self.blackboard.get(self.box_axes_key)
            if self.blackboard.exists(self.box_axes_key)
            else None
        )
        right_pull_target = (
            self.blackboard.get(self.right_pull_key)
            if self.blackboard.exists(self.right_pull_key)
            else None
        )
        if left_edge_point is None or box_axes is None or right_pull_target is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 缺少左手补抓目标计算所需抓取数据"
            )
            return Status.FAILURE

        up_axis = np.array(box_axes["up"], dtype=float)
        left_axis = np.array(box_axes["left"], dtype=float)
        approach_offset = self._get_float_param("right_approach_offset", 0.1)
        descend_below_offset = self._get_float_param("right_descend_below_offset", 0.01)
        lift_offset = self._get_float_param("right_lift_offset", 0.1)
        pull_right_offset = self._get_float_param("right_pull_right_offset", 0.15)
        carry_lift_offset = self._get_float_param("carry_lift_offset", 0.3)
        if carry_lift_offset < lift_offset:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 搬运上提高度不能小于单手上提高度: "
                f"carry_lift={carry_lift_offset:.3f}, single_lift={lift_offset:.3f}"
            )
            return Status.FAILURE

        moved_left_edge = left_edge_point - left_axis * pull_right_offset
        above_left_edge = moved_left_edge + up_axis * approach_offset
        below_left_edge = moved_left_edge - up_axis * descend_below_offset
        left_lift_target = below_left_edge + up_axis * lift_offset

        # 右拉后的回中方向是 +left_axis，与右拉方向相反。
        return_center_offset = left_axis * pull_right_offset
        additional_lift_offset = carry_lift_offset - lift_offset
        return_center_right_target = (
            right_pull_target
            + return_center_offset
            + up_axis * additional_lift_offset
        )
        return_center_left_target = (
            below_left_edge
            + return_center_offset
            + up_axis * carry_lift_offset
        )
        lower_left_target = return_center_left_target - up_axis * carry_lift_offset
        lower_right_target = return_center_right_target - up_axis * carry_lift_offset

        values = {
            "moved_left_edge": moved_left_edge,
            "left_above": above_left_edge,
            "left_below": below_left_edge,
            "left_lift": left_lift_target,
            "return_center_left": return_center_left_target,
            "return_center_right": return_center_right_target,
            "lower_left": lower_left_target,
            "lower_right": lower_right_target,
        }
        for name, value in values.items():
            self.blackboard.set(self.target_keys[name], value, overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算左手补偿及双手回中目标: "
            f"pull_right={pull_right_offset:.3f}, carry_lift={carry_lift_offset:.3f}"
        )
        return Status.SUCCESS

    def _get_float_param(self, name, default):
        return float(self.params.get(name, self.ros_node.get_param(name, default)))
