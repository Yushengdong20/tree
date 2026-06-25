"""将躯干恢复到 move_box 初始化时记录的初始位姿。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxTorsoToInitialPose(TimedMockAction):
    """对应参考项目 pick_procedure.return_torso_to_initial_pose。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.wait_done = self._to_bool(params.get("wait_done", True))
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

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

        initial_pose = list(services.torso_controller.initial_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 躯干回到 init torso 位姿: {initial_pose}"
        )
        ok = services.torso_controller.move_to_pose(initial_pose, wait_done=self.wait_done)
        return Status.SUCCESS if ok else Status.FAILURE
