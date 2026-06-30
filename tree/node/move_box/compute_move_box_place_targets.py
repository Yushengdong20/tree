"""计算 move_box 放置流程的双夹爪下降目标点。

参考业务代码：
/home/ysd/Desktop/leju/leju_wbc/src/kuavo_humanoid_sdk/move_box/place_procedure.py
PlaceProcedure.update_lower_claw_points()
"""

import numpy as np
import py_trees
from py_trees.common import Status

from tree.utils.geometry import ypr_to_rotation_matrix

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class ComputeMoveBoxPlaceTargets(TimedMockAction):
    """基于当前夹爪位置生成放置下降目标，只修改夹爪 z 高度。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.place_plane_height = float(params.get("place_plane_height", ros_node.get_param("place_plane_height", 0.0)))
        self.box_size_z = float(params.get("box_size_z", ros_node.get_param("box_size_z", 0.34)))
        self.left_target_key = str(params.get("left_target_key", "move_box_place_left_lower_claw_point")).strip()
        self.right_target_key = str(params.get("right_target_key", "move_box_place_right_lower_claw_point")).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.left_target_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.right_target_key, access=py_trees.common.Access.WRITE)

    def _current_claw_point(self, arm_controller, side):
        if side not in ("left", "right"):
            return None

        if hasattr(arm_controller, "get_current_end_effector_pose"):
            current_pose = arm_controller.get_current_end_effector_pose(side)
        else:
            current_pose = None

        if current_pose is None:
            current_pose = (
                arm_controller.current_left_target
                if side == "left"
                else arm_controller.current_right_target
            )
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] {side} 当前实际末端 TF 无效，回退到上一次目标缓存"
            )

        if len(current_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] {side} 当前手臂位姿长度必须为 6: {current_pose}"
            )
            return None

        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] 获取 {side} 末端到夹爪 TF 失败")
            return None

        translation, _ = transform
        rotation = ypr_to_rotation_matrix(current_pose[3:6])
        end_effector_point = np.array(current_pose[:3], dtype=float)
        return end_effector_point + rotation.dot(translation)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 或 arm_controller 缺失: key={self.services_key}"
            )
            return Status.FAILURE

        arm_controller = services.arm_controller
        left_claw_point = self._current_claw_point(arm_controller, "left")
        right_claw_point = self._current_claw_point(arm_controller, "right")
        if left_claw_point is None or right_claw_point is None:
            return Status.FAILURE

        target_claw_z = self.place_plane_height + self.box_size_z
        lower_left = np.array(left_claw_point, dtype=float)
        lower_right = np.array(right_claw_point, dtype=float)
        lower_left[2] = target_claw_z
        lower_right[2] = target_claw_z

        self.blackboard.set(self.left_target_key, lower_left, overwrite=True)
        self.blackboard.set(self.right_target_key, lower_right, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算放置下降目标: "
            f"box_size_z={self.box_size_z:.3f}, target_claw_z={target_claw_z:.3f}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPlaceTargets start: "
            f"place_plane_height={self.place_plane_height:.3f}, box_size_z={self.box_size_z:.3f}"
        )
