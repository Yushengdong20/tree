"""根据箱体中心计算躯干目标位姿，并写入 blackboard。

本节点只负责计算，不直接控制腰部；后续由 MoveTorsoPose 读取 pose_key 执行。
计算逻辑保留原抓箱流程的躯干对准约束。
"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxTorsoToBoxCenterPose(TimedMockAction):
    """根据最新箱体中心生成抓取前躯干对准目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.box_center_key = str(
            params.get("box_center_key", "move_box_latest_box_center")
        ).strip()
        self.preferred_box_center_key = str(
            params.get("preferred_box_center_key", "")
        ).strip()
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_torso_box_center_pose")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)
        if self.preferred_box_center_key:
            self.blackboard.register_key(
                key=self.preferred_box_center_key,
                access=py_trees.common.Access.READ,
            )
        self.blackboard.register_key(key=self.target_pose_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        box_center, box_center_source = self._get_box_center()
        if services is None or box_center is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services 或箱体中心缺失: "
                f"services={services is not None}, box_center={box_center is not None}"
            )
            return Status.FAILURE

        target_pose = list(services.torso_controller.current_pose)
        if len(target_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前躯干位姿长度异常: {target_pose}"
            )
            return Status.FAILURE

        target_pose[0] = min(max(float(box_center[0]) - 0.5, 0.0), 0.5)
        target_pose[2] = min(max(float(box_center[2]) + 0.2, 0.6), 1.4)
        target_pose[4] = 0.4
        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算抓取前躯干目标: "
            f"source={box_center_source}, "
            f"box=({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}), "
            f"pose={target_pose}, key={self.target_pose_key}"
        )
        return Status.SUCCESS

    def _get_box_center(self):
        """优先读取到点时保存的箱体中心，缺失时回退到最新检测结果。"""
        if self.preferred_box_center_key and self.blackboard.exists(self.preferred_box_center_key):
            box_center = self.blackboard.get(self.preferred_box_center_key)
            if box_center is not None:
                return box_center, self.preferred_box_center_key
        if self.blackboard.exists(self.box_center_key):
            return self.blackboard.get(self.box_center_key), self.box_center_key
        return None, self.box_center_key

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxTorsoToBoxCenterPose start: "
            f"box_center_key={self.box_center_key}, target_pose_key={self.target_pose_key}"
        )
