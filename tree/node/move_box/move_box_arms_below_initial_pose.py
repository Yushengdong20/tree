"""将双手移动到初始位姿下方指定距离。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.utils.geometry import transform_xyz_ypr


class MoveBoxArmsBelowInitialPose(TimedMockAction):
    """松开箱体后将双手降低到初始位姿下方，便于重新观察和安全回撤。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.lower_offset = float(params.get("lower_offset", 0.4))
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
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        arm_controller = services.arm_controller
        if hasattr(arm_controller, "initial_left_pose_in_waist") and hasattr(
            arm_controller, "initial_right_pose_in_waist"
        ):
            if (
                len(arm_controller.initial_left_pose_in_waist) != 6
                or len(arm_controller.initial_right_pose_in_waist) != 6
            ):
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] waist 初始手臂位姿长度必须为 6"
                )
                return Status.FAILURE

            left_release_pose_in_waist = list(arm_controller.initial_left_pose_in_waist)
            right_release_pose_in_waist = list(arm_controller.initial_right_pose_in_waist)
            left_release_pose_in_waist[2] -= self.lower_offset
            right_release_pose_in_waist[2] -= self.lower_offset
            try:
                left_release_pose = transform_xyz_ypr(
                    services.tf_listener,
                    left_release_pose_in_waist,
                    "waist_yaw_link",
                    "base_link",
                )
                right_release_pose = transform_xyz_ypr(
                    services.tf_listener,
                    right_release_pose_in_waist,
                    "waist_yaw_link",
                    "base_link",
                )
            except RuntimeError as err:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 放下手目标从 waist_yaw_link 转到 base_link 失败: {err}"
                )
                return Status.FAILURE
        else:
            if len(arm_controller.initial_left_pose) != 6 or len(arm_controller.initial_right_pose) != 6:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 初始手臂位姿长度必须为 6"
                )
                return Status.FAILURE

            left_release_pose = list(arm_controller.initial_left_pose)
            right_release_pose = list(arm_controller.initial_right_pose)
            left_release_pose[2] -= self.lower_offset
            right_release_pose[2] -= self.lower_offset

        ok = arm_controller.execute_arm_event(left_release_pose, right_release_pose)
        return Status.SUCCESS if ok else Status.FAILURE
