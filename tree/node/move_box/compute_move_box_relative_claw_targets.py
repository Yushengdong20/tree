"""基于当前真实夹爪位置生成相对位移目标。"""

import numpy as np
import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction


class ComputeMoveBoxRelativeClawTargets(TimedMockAction):
    """读取当前双臂末端 TF，生成相对平移后的左右夹爪目标点。

    目标点始终在 ``base_link`` 系下。该节点特别适合“保持夹持、只让
    双手避让相机视野”这类动作：不依赖上一节点缓存的目标，因而会以实际
    已到达的夹爪位置为起点。

    参数 ``dx/dy/dz`` 是 base_link 系的米单位位移；例如 ``dz=-0.20``
    表示两只夹爪向下移动 20 cm，x/y 不变。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.dx = float(params.get("dx", 0.0))
        self.dy = float(params.get("dy", 0.0))
        self.dz = float(params.get("dz", 0.0))
        self.left_target_key = str(
            params.get("left_target_key", "move_box_relative_left_claw_target")
        ).strip()
        self.right_target_key = str(
            params.get("right_target_key", "move_box_relative_right_claw_target")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.left_target_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.right_target_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        arm_controller = getattr(services, "arm_controller", None)
        if arm_controller is None:
            self.feedback_message = f"缺少 arm_controller: services_key={self.services_key}"
            self.ros_node.get_logger().error(f"[{self.config_label}] {self.feedback_message}")
            return Status.FAILURE

        left_current = self._get_current_claw_point(arm_controller, "left")
        right_current = self._get_current_claw_point(arm_controller, "right")
        if left_current is None or right_current is None:
            self.feedback_message = "无法读取双手当前末端 TF 并换算夹爪位置"
            return Status.FAILURE

        offset = np.array([self.dx, self.dy, self.dz], dtype=float)
        left_target = (left_current + offset).tolist()
        right_target = (right_current + offset).tolist()
        self.blackboard.set(self.left_target_key, left_target, overwrite=True)
        self.blackboard.set(self.right_target_key, right_target, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已生成双夹爪相对位移目标(base_link): "
            f"offset=({self.dx:.3f},{self.dy:.3f},{self.dz:.3f}), "
            f"left=({left_target[0]:.3f},{left_target[1]:.3f},{left_target[2]:.3f}), "
            f"right=({right_target[0]:.3f},{right_target[1]:.3f},{right_target[2]:.3f})"
        )
        return Status.SUCCESS

    def _get_current_claw_point(self, arm_controller, side):
        current_eef = arm_controller.get_current_end_effector_pose(side)
        if current_eef is None or len(current_eef) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] {side} 当前末端位姿无效: {current_eef!r}"
            )
            return None
        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] {side} 末端到夹爪 TF 不可用")
            return None
        translation, _ = transform
        return np.array(current_eef[:3], dtype=float) + ypr_to_rotation_matrix(current_eef[3:6]).dot(
            np.array(translation, dtype=float)
        )

