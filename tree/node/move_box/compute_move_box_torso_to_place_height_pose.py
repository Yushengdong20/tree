"""根据放置平面高度计算躯干目标位姿，并写入 blackboard。

本节点只负责计算，不直接控制腰部；后续由 MoveTorsoPose 读取 pose_key 执行。
计算逻辑与 MoveBoxTorsoToPlaceHeight 保持一致，便于逐步替换旧业务节点。
"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxTorsoToPlaceHeightPose(TimedMockAction):
    """根据放置平面高度生成放箱前躯干目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.place_plane_height = float(params.get("place_plane_height", ros_node.get_param("place_plane_height", 0.0)))
        self.height_offset = float(
            params.get("place_torso_height_offset", ros_node.get_param("place_torso_height_offset", 0.4))
        )
        self.min_height = float(
            params.get("place_torso_min_height", ros_node.get_param("place_torso_min_height", 0.6))
        )
        self.max_height = float(
            params.get("place_torso_max_height", ros_node.get_param("place_torso_max_height", 1.4))
        )
        self.torso_x = float(params.get("place_torso_x", ros_node.get_param("place_torso_x", 0.5)))
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_place_torso_pose")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.target_pose_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "torso_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 或 torso_controller 缺失: key={self.services_key}"
            )
            return Status.FAILURE

        target_pose = list(services.torso_controller.current_pose)
        if len(target_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前躯干位姿长度异常: {target_pose}"
            )
            return Status.FAILURE

        target_z = min(
            max(self.place_plane_height + self.height_offset, self.min_height),
            self.max_height,
        )
        target_pose[0] = self.torso_x
        target_pose[2] = target_z
        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算放箱前躯干目标: "
            f"plane_z={self.place_plane_height:.3f}, pose={target_pose}, "
            f"key={self.target_pose_key}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxTorsoToPlaceHeightPose start: "
            f"place_plane_height={self.place_plane_height:.3f}, target_pose_key={self.target_pose_key}"
        )
