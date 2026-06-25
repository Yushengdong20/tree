"""根据最新箱体中心调整躯干前后位置和高度。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxTorsoToBoxCenter(TimedMockAction):
    """让躯干先对准箱体中心，为后续抓取留出合适操作空间。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.box_center_key = str(
            params.get("box_center_key", "move_box_latest_box_center")
        ).strip()
        self.preferred_box_center_key = str(
            params.get("preferred_box_center_key", "")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)
        if self.preferred_box_center_key:
            self.blackboard.register_key(
                key=self.preferred_box_center_key,
                access=py_trees.common.Access.READ,
            )

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
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
            return Status.SUCCESS

        target_torso_pose = list(services.torso_controller.current_pose)
        target_torso_pose[0] = min(max(float(box_center[0]) - 0.5, 0.0), 0.5)
        target_torso_pose[2] = min(max(float(box_center[2]) + 0.2, 0.6), 1.4)
        target_torso_pose[4] = 0.4
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 根据箱体中心调整躯干: "
            f"source={box_center_source}, "
            f"box=({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}), "
            f"torso_x={target_torso_pose[0]:.3f}, torso_z={target_torso_pose[2]:.3f}"
        )
        ok = services.torso_controller.move_to_pose(target_torso_pose)
        return Status.SUCCESS if ok else Status.FAILURE

    def _get_box_center(self):
        """优先读取配置指定的箱体中心，缺失时回退到默认检测结果。"""
        if self.preferred_box_center_key and self.blackboard.exists(self.preferred_box_center_key):
            box_center = self.blackboard.get(self.preferred_box_center_key)
            if box_center is not None:
                return box_center, self.preferred_box_center_key
        if self.blackboard.exists(self.box_center_key):
            return self.blackboard.get(self.box_center_key), self.box_center_key
        return None, self.box_center_key
