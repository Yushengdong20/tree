"""计算躯干回位后用于恢复双手 pitch 的目标位姿。

本节点只读取当前双手位姿并写入 blackboard，不直接控制手臂。
后续由 MoveBoxArmsToPose 读取目标位姿并执行。
"""

import py_trees
from py_trees.common import Status

from kuavo_humanoid_sdk.common.arm_controller import WAIST_YAW_LINK_FRAME

from ..base import TimedMockAction


class ComputeMoveBoxRestorePitchArmPoses(TimedMockAction):
    """保持双手当前位置，仅将末端姿态恢复到指定 pitch。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.pitch = float(params.get("pitch", ros_node.get_param("post_torso_return_pitch", -60.0)))
        self.left_pose_key = str(
            params.get("left_pose_key", "move_box_restore_pitch_left_pose")
        ).strip()
        self.right_pose_key = str(
            params.get("right_pose_key", "move_box_restore_pitch_right_pose")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.left_pose_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.right_pose_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services 或 arm_controller 缺失: "
                f"key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        left_pose = services.arm_controller.get_current_end_effector_pose(
            "left",
            target_frame=WAIST_YAW_LINK_FRAME,
        )
        right_pose = services.arm_controller.get_current_end_effector_pose(
            "right",
            target_frame=WAIST_YAW_LINK_FRAME,
        )
        if left_pose is None or right_pose is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 无法获取 waist_yaw_link 下的当前双手位姿"
            )
            return Status.FAILURE
        if len(left_pose) != 6 or len(right_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前双手位姿长度必须为 6"
            )
            return Status.FAILURE

        # 关键步骤：保持双手相对腰部的位置，只恢复末端姿态，避免搬箱过程中改变持箱位置。
        left_target = list(left_pose)
        right_target = list(right_pose)
        left_target[3:6] = [0.0, self.pitch, 0.0]
        right_target[3:6] = [0.0, self.pitch, 0.0]

        self.blackboard.set(self.left_pose_key, left_target, overwrite=True)
        self.blackboard.set(self.right_pose_key, right_target, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算双手 pitch 恢复目标: "
            f"pitch={self.pitch:.2f}, left_key={self.left_pose_key}, "
            f"right_key={self.right_pose_key}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxRestorePitchArmPoses start: "
            f"pitch={self.pitch:.2f}"
        )
