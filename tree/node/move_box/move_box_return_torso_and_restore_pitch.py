"""躯干回位后保持双手当前位置，并恢复末端 pitch。"""

import time

import py_trees
from py_trees.common import Status

from kuavo_humanoid_sdk.common.arm_controller import WAIST_YAW_LINK_FRAME

from ..base import TimedMockAction


class MoveBoxReturnTorsoAndRestorePitch(TimedMockAction):
    """对应参考项目 pick_procedure.return_torso_and_restore_end_effector_pitch。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.wait_done = self._to_bool(params.get("wait_done", True))
        self.post_torso_return_pitch = float(
            params.get(
                "post_torso_return_pitch",
                self.ros_node.get_param("post_torso_return_pitch", -60.0),
            )
        )
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

        initial_pose = list(services.torso_controller.initial_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 躯干回到 init torso 位姿: {initial_pose}"
        )
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            if self.should_skip_torso_motion():
                self.log_skip_torso_motion()
                return Status.SUCCESS
            ok = services.torso_controller.move_to_pose(initial_pose, wait_done=self.wait_done)
            return Status.SUCCESS if ok else Status.FAILURE

        left_target = services.arm_controller.get_current_end_effector_pose(
            "left",
            target_frame=WAIST_YAW_LINK_FRAME,
        )
        right_target = services.arm_controller.get_current_end_effector_pose(
            "right",
            target_frame=WAIST_YAW_LINK_FRAME,
        )
        if left_target is None or right_target is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前双手实际位姿无效，无法修正末端 pitch"
            )
            return Status.FAILURE
        if len(left_target) != 6 or len(right_target) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前双手实际位姿长度必须为 6，无法修正末端 pitch"
            )
            return Status.FAILURE

        # 关键步骤：先读取 waist_yaw_link 下的实际末端位姿，再发腰部目标，避免读取腰部运动中的中间状态。
        left_target[3] = 0
        left_target[4] = self.post_torso_return_pitch
        left_target[5] = 0

        right_target[3] = 0
        right_target[4] = self.post_torso_return_pitch
        right_target[5] = 0

        torso_start_time = None
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
        else:
            torso_start_time = time.monotonic()
            if not services.torso_controller.move_to_pose(initial_pose, wait_done=self.wait_done):
                return Status.FAILURE

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 基于 waist_yaw_link 保持末端相对腰部位置，"
            f"修正 pitch={self.post_torso_return_pitch:.2f} deg"
        )
        services.arm_controller.reach_time = 0.0
        ok = services.arm_controller.execute_arm_event(
            left_target,
            right_target,
            pose_frame=WAIST_YAW_LINK_FRAME,
        )
        if not ok:
            return Status.FAILURE

        if self.wait_done and torso_start_time is not None and services.torso_controller.reach_time > 0.0:
            # 关键步骤：手臂执行期间腰部已经在运动，这里只补足腰部剩余运动时间。
            elapsed_time = time.monotonic() - torso_start_time
            remaining_time = max(services.torso_controller.reach_time - elapsed_time, 0.0)
            self.ros_node.sleep(remaining_time + services.torso_controller.motion_done_padding)
        return Status.SUCCESS
