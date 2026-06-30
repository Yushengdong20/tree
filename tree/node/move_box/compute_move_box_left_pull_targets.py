"""计算分步抓箱流程中左手需要经过的目标点。"""

import numpy as np
import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxLeftPullTargets(TimedMockAction):
    """把左手靠近、下探、上提和外拉目标写入 blackboard。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_pair_key = str(
            params.get("grasp_pair_key", "move_box_latest_grasp_pair")
        ).strip()
        self.box_axes_key = str(
            params.get("box_axes_key", "move_box_latest_box_axes")
        ).strip()
        self.target_keys = {
            "left_edge": str(params.get("left_edge_key", "move_box_left_edge_point")).strip(),
            "right_edge": str(params.get("right_edge_key", "move_box_right_edge_point")).strip(),
            "left_above": str(params.get("left_above_key", "move_box_left_pull_above_edge")).strip(),
            "left_below": str(params.get("left_below_key", "move_box_left_pull_below_edge")).strip(),
            "left_lift": str(params.get("left_lift_key", "move_box_left_pull_lift_target")).strip(),
            "left_pull": str(params.get("left_pull_key", "move_box_left_pull_target")).strip(),
        }
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        for key in self.target_keys.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        """根据箱体左右抓取点和方向轴生成左手分步目标。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        grasp_pair = (
            self.blackboard.get(self.grasp_pair_key)
            if self.blackboard.exists(self.grasp_pair_key)
            else None
        )
        box_axes = (
            self.blackboard.get(self.box_axes_key)
            if self.blackboard.exists(self.box_axes_key)
            else None
        )
        if grasp_pair is None or box_axes is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 缺少左手目标计算所需抓取数据"
            )
            return Status.FAILURE

        left_edge_point, right_edge_point = grasp_pair
        up_axis = np.array(box_axes["up"], dtype=float)
        left_axis = np.array(box_axes["left"], dtype=float)
        approach_offset = self._get_float_param("left_approach_offset", 0.1)
        descend_below_offset = self._get_float_param("left_descend_below_offset", 0.01)
        lift_offset = self._get_float_param("left_lift_offset", 0.1)
        pull_left_offset = self._get_float_param("left_pull_left_offset", 0.15)

        above_left_edge = left_edge_point + up_axis * approach_offset
        below_left_edge = left_edge_point - up_axis * descend_below_offset
        lift_target = below_left_edge + up_axis * lift_offset
        pull_target = lift_target + left_axis * pull_left_offset

        self.blackboard.set(self.target_keys["left_edge"], left_edge_point, overwrite=True)
        self.blackboard.set(self.target_keys["right_edge"], right_edge_point, overwrite=True)
        self.blackboard.set(self.target_keys["left_above"], above_left_edge, overwrite=True)
        self.blackboard.set(self.target_keys["left_below"], below_left_edge, overwrite=True)
        self.blackboard.set(self.target_keys["left_lift"], lift_target, overwrite=True)
        self.blackboard.set(self.target_keys["left_pull"], pull_target, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算左手分步抓取目标: "
            f"approach={approach_offset:.3f}, descend={descend_below_offset:.3f}, "
            f"lift={lift_offset:.3f}, pull={pull_left_offset:.3f}"
        )
        return Status.SUCCESS

    def _get_float_param(self, name, default):
        """节点参数优先，未配置时回退到 ROS 参数。"""
        return float(self.params.get(name, self.ros_node.get_param(name, default)))
