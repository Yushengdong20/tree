"""计算 move_box 直接夹取流程的双手同步目标点。"""

import numpy as np
import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxDirectGraspTargets(TimedMockAction):
    """按箱体上向生成直接夹取流程的上方、下方和上提目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.target_keys = {
            "left_above": str(params.get("left_above_key", "move_box_direct_left_above_edge")).strip(),
            "right_above": str(params.get("right_above_key", "move_box_direct_right_above_edge")).strip(),
            "left_below": str(params.get("left_below_key", "move_box_direct_left_below_edge")).strip(),
            "right_below": str(params.get("right_below_key", "move_box_direct_right_below_edge")).strip(),
            "left_lift": str(params.get("left_lift_key", "move_box_direct_left_lift_target")).strip(),
            "right_lift": str(params.get("right_lift_key", "move_box_direct_right_lift_target")).strip(),
        }
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        for key in self.target_keys.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        """根据左右边缘点和箱体上向，计算直接夹取的双手同步目标。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        grasp_pair = self.blackboard.get(self.grasp_pair_key) if self.blackboard.exists(self.grasp_pair_key) else None
        box_axes = self.blackboard.get(self.box_axes_key) if self.blackboard.exists(self.box_axes_key) else None
        if grasp_pair is None or box_axes is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少直接抓取目标计算所需数据")
            return Status.FAILURE

        left_edge_point, right_edge_point = grasp_pair
        up_axis = np.array(box_axes["up"], dtype=float)
        approach_offset = float(
            self.params.get(
                "direct_approach_offset",
                self.ros_node.get_param(
                    "direct_approach_offset",
                    self.ros_node.get_param("left_approach_offset", 0.05),
                ),
            )
        )
        descend_below_offset = float(
            self.params.get(
                "direct_descend_below_offset",
                self.ros_node.get_param(
                    "direct_descend_below_offset",
                    self.ros_node.get_param("left_descend_below_offset", 0.06),
                ),
            )
        )
        lift_offset = float(
            self.params.get(
                "direct_lift_offset",
                self.ros_node.get_param(
                    "direct_lift_offset",
                    self.ros_node.get_param("left_lift_offset", 0.2),
                ),
            )
        )

        above_left_edge = left_edge_point + up_axis * approach_offset
        above_right_edge = right_edge_point + up_axis * approach_offset
        below_left_edge = left_edge_point - up_axis * descend_below_offset
        below_right_edge = right_edge_point - up_axis * descend_below_offset
        lift_left_target = below_left_edge + up_axis * lift_offset
        lift_right_target = below_right_edge + up_axis * lift_offset

        self.blackboard.set(self.target_keys["left_above"], above_left_edge, overwrite=True)
        self.blackboard.set(self.target_keys["right_above"], above_right_edge, overwrite=True)
        self.blackboard.set(self.target_keys["left_below"], below_left_edge, overwrite=True)
        self.blackboard.set(self.target_keys["right_below"], below_right_edge, overwrite=True)
        self.blackboard.set(self.target_keys["left_lift"], lift_left_target, overwrite=True)
        self.blackboard.set(self.target_keys["right_lift"], lift_right_target, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算直接抓取目标: "
            f"approach={approach_offset:.3f}, descend={descend_below_offset:.3f}, lift={lift_offset:.3f}"
        )
        return Status.SUCCESS
