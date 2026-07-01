"""在 blackboard 上创建或复用 grasp_object 真实共享服务。"""

import py_trees
from py_trees.common import Status

from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import (
    IK_MODEL_GRASP,
)
from tree.constants import MODEL_TYPE_KEY, ROBOT_SERVICES_KEY, WAIST_YAW_LINK_FRAME

from ..base import TimedMockAction


class EnsureGraspObjectServices(TimedMockAction):
    """确保抓取任务所需的共享控制实例存在。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.model_type_key = MODEL_TYPE_KEY
        self.model_type = str(params.get("model_type", IK_MODEL_GRASP)).strip() or IK_MODEL_GRASP
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.model_type_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if not self._is_grasp_services(services):
            from tree.runtime.grasp.grasp_real import build_grasp_object_services

            services = build_grasp_object_services(model_type=self.model_type)
            self.blackboard.set(self.services_key, services, overwrite=True)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] created grasp services: services_id={id(services)}"
            )
        else:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] reused grasp services: services_id={id(services)}"
            )
        if not hasattr(services, "arm_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 中没有 arm_controller: key={self.services_key}"
            )
            return Status.FAILURE
        self.blackboard.set(self.model_type_key, services.model_type, overwrite=True)
        return Status.SUCCESS

    def _is_grasp_services(self, services):
        return (
            services is not None
            and hasattr(services, "arm_controller")
            and hasattr(services, "torso_controller")
            and getattr(services, "target_frame", None) == WAIST_YAW_LINK_FRAME
            and getattr(services, "model_type", None) == self.model_type
        )

    def describe_start(self):
        return f"[{self.config_label}] EnsureGraspObjectServices start: key={self.services_key}"
