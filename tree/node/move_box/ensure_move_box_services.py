"""在 blackboard 上创建或复用 move_box 真实共享服务。"""

import py_trees
from py_trees.common import Status

from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import (
    IK_MODEL_MOVE_BOX,
)
from tree.constants import MODEL_TYPE_KEY, ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class EnsureMoveBoxServices(TimedMockAction):
    """确保真实机器人所需的 move_box 控制实例存在，并立即进入准备姿态。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.model_type_key = MODEL_TYPE_KEY
        self.model_type = str(params.get("model_type", IK_MODEL_MOVE_BOX)).strip() or IK_MODEL_MOVE_BOX
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.model_type_key, access=py_trees.common.Access.WRITE)

    def update(self):
        """若服务不存在则创建，并在同一节点内完成头、腰、手臂的准备动作。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if not self._is_move_box_services(services):
            from tree.runtime.move_box.move_box_real import build_robot_services

            services = build_robot_services(model_type=self.model_type)
            self.blackboard.set(self.services_key, services, overwrite=True)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] created robot services: services_id={id(services)}"
            )
        else:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] reused robot services: services_id={id(services)}"
            )
        self.blackboard.set(self.model_type_key, services.model_type, overwrite=True)
        self._prepare_robot(services)
        return Status.SUCCESS

    def _is_move_box_services(self, services):
        return (
            services is not None
            and hasattr(services, "static_tf_publisher")
            and hasattr(services, "arm_controller")
            and getattr(services, "model_type", None) == self.model_type
        )

    def _prepare_robot(self, services):
        """发布静态 TF，并驱动头、腰、手臂进入初始观测和抓取准备状态。"""
        services.static_tf_publisher.publish_claw_tfs()
        if self.should_skip_head_motion():
            self.log_skip_head_motion()
        else:
            services.head_controller.set_head_target(
                float(self.ros_node.get_param("initial_head_yaw", 0.0)),
                float(self.ros_node.get_param("initial_head_pitch", 20.0)),
            )

        self.ros_node.sleep(1.0)
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
        else:
            services.arm_controller.prepare()
        self.ros_node.sleep(1.0)

        torso_ready_pose = list(services.torso_controller.current_pose)
        torso_ready_pose[0] = 0.0
        torso_ready_pose[2] = float(
            self.params.get(
                "initial_torso_height",
                self.ros_node.get_param("initial_torso_height", 1.1),
            )
        )
        torso_ready_pose[4] = 0.0
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
            return
        services.torso_controller.move_to_pose(torso_ready_pose, wait_done=False)

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return f"[{self.config_label}] EnsureMoveBoxServices start: key={self.services_key}"
