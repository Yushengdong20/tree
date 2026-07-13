"""计算 move_box 直接夹取流程的双手同步目标点。"""

import numpy as np
import py_trees
from py_trees.common import Status
from visualization_msgs.msg import MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.geometry import get_odom_pose_transformer
from ..base import TimedMockAction
from .fp_grasp_visualization import publish_fp_box_and_targets


class ComputeMoveBoxDirectGraspTargets(TimedMockAction):
    """按箱体上向生成直接夹取流程的上方、下方和上提目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(params.get("box_center_key", "move_box_latest_box_center")).strip()
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.fp_grasp_visualization_enabled = self._to_bool(
            params.get("fp_grasp_visualization_enabled", True)
        )
        self.fp_grasp_visualization_topic = str(
            params.get("fp_grasp_visualization_topic", "/move_box/fp_grasp_markers")
        ).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.fp_grasp_visualization_publisher = None
        if self.fp_grasp_visualization_enabled and self.fp_grasp_visualization_topic:
            self.fp_grasp_visualization_publisher = self.ros_node.create_publisher(
                self.fp_grasp_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.target_keys = {
            "left_above": str(params.get("left_above_key", "move_box_direct_left_above_edge")).strip(),
            "right_above": str(params.get("right_above_key", "move_box_direct_right_above_edge")).strip(),
            "left_below": str(params.get("left_below_key", "move_box_direct_left_below_edge")).strip(),
            "right_below": str(params.get("right_below_key", "move_box_direct_right_below_edge")).strip(),
            "left_lift": str(params.get("left_lift_key", "move_box_direct_left_lift_target")).strip(),
            "right_lift": str(params.get("right_lift_key", "move_box_direct_right_lift_target")).strip(),
        }
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)
        for key in self.target_keys.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

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
        self._publish_visualization(
            grasp_pair,
            box_axes,
            {
                "left_above": above_left_edge,
                "right_above": above_right_edge,
                "left_below": below_left_edge,
                "right_below": below_right_edge,
                "left_lift": lift_left_target,
                "right_lift": lift_right_target,
            },
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算直接抓取目标: "
            f"approach={approach_offset:.3f}, descend={descend_below_offset:.3f}, lift={lift_offset:.3f}"
        )
        return Status.SUCCESS

    def _publish_visualization(self, grasp_pair, box_axes, target_points):
        box_center = (
            self.blackboard.get(self.box_center_key)
            if self.blackboard.exists(self.box_center_key)
            else None
        )
        services = (
            self.blackboard.get(self.services_key)
            if self.blackboard.exists(self.services_key)
            else None
        )
        publish_fp_box_and_targets(
            ros_node=self.ros_node,
            publisher=self.fp_grasp_visualization_publisher,
            topic=self.fp_grasp_visualization_topic,
            config_label=self.config_label,
            odom_transformer=self.odom_transformer,
            services=services,
            box_center=box_center,
            box_axes=box_axes,
            strategy="direct",
            grasp_pair=grasp_pair,
            target_points=target_points,
            include_grasp_targets=True,
        )
