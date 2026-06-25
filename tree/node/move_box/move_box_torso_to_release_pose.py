"""将躯干移动到 move_box 释放/观察姿态。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxTorsoToReleasePose(TimedMockAction):
    """驱动躯干回到释放姿态，便于安全复位与重新观察。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.release_pose = params.get(
            "release_pose",
            [0.0, 0.0, 1.1, 0.0, 0.0, 0.0],
        )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
            return Status.SUCCESS
        ok = services.torso_controller.move_to_pose(list(self.release_pose), wait_done=False)
        return Status.SUCCESS if ok else Status.FAILURE
