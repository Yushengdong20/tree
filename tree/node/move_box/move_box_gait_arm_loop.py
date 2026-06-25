"""move_box 专用导航摆臂循环节点。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxGaitArmLoop(TimedMockAction):
    """启动 move_box 共享 ArmController 的摆臂，节点结束时自动停止。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.services = None
        self.started = False
        self.skipped = False
        self.startup_error = None

    def initialise(self):
        super().initialise()
        self.services = None
        self.started = False
        self.skipped = False
        self.startup_error = None

        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            self.skipped = True
            return

        self.services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if self.services is None or not hasattr(self.services, "arm_controller"):
            self.startup_error = RuntimeError(
                f"move_box services 或 arm_controller 缺失: key={self.services_key}"
            )
            self.ros_node.get_logger().error(f"[{self.config_label}] {self.startup_error}")
            return

        # 关键步骤：复用 ArmController 内部的 GaitArmLoopPlayer，避免在节点里重复维护 CSV 播放逻辑。
        if self.services.arm_controller.start_gait_arm():
            self.started = True
            self.ros_node.get_logger().info(f"[{self.config_label}] 已启动 move_box 导航摆臂")
            return

        self.startup_error = RuntimeError("启动 move_box 导航摆臂失败")
        self.ros_node.get_logger().error(f"[{self.config_label}] {self.startup_error}")

    def update(self):
        if self.skipped:
            return Status.RUNNING
        if self.startup_error is not None:
            return Status.FAILURE
        return Status.RUNNING

    def terminate(self, new_status):
        # 关键步骤：Parallel 在导航成功后会中断 RUNNING 的摆臂节点，这里负责停止后台发布线程。
        if self.started and self.services is not None and hasattr(self.services, "arm_controller"):
            if not self.services.arm_controller.stop_gait_arm():
                self.ros_node.get_logger().error(f"[{self.config_label}] 停止 move_box 导航摆臂失败")
            else:
                self.ros_node.get_logger().info(f"[{self.config_label}] 已停止 move_box 导航摆臂")
        self.started = False
        super().terminate(new_status)

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxGaitArmLoop start: key={self.services_key}"
