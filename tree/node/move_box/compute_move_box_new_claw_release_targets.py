"""计算新款上下合拢夹爪在松爪后的双臂侧向避让点。"""

import numpy as np
import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction


class ComputeMoveBoxNewClawReleaseTargets(TimedMockAction):
    """在箱体落稳、夹爪已张开后，按当前实际夹爪位置计算两臂向外撤离点。

    放置高度由 :class:`ComputeMoveBoxPlaceTargets` 统一计算；本节点只负责
    新夹爪特有的释放后横向避让，避免抓取时缓存的点位与真实落位存在误差。

    ``grasp_pair_key`` 中的 left/right 点只用于确定“向外”的左右方向；动作起点
    始终读取当前实际末端 TF 后反算夹爪位置。故即使下降时有小执行误差，撤离也会
    从真实位置开始，而不是回到抓取阶段的旧坐标。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.retract_offset_m = float(params.get("new_claw_release_retract_offset_m", 0.06))
        self.left_target_key = str(
            params.get("left_target_key", "move_box_new_claw_left_release_retract")
        ).strip()
        self.right_target_key = str(
            params.get("right_target_key", "move_box_new_claw_right_release_retract")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.left_target_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.right_target_key, access=py_trees.common.Access.WRITE)

    def _current_claw_point(self, arm_controller, side):
        current_pose = (
            arm_controller.get_current_end_effector_pose(side)
            if hasattr(arm_controller, "get_current_end_effector_pose")
            else None
        )
        if current_pose is None:
            current_pose = (
                arm_controller.current_left_target
                if side == "left"
                else arm_controller.current_right_target
            )
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] {side} 当前末端 TF 无效，侧向撤离回退至上次目标"
            )
        if current_pose is None or len(current_pose) != 6:
            raise ValueError(f"{side} 当前末端位姿无效: {current_pose!r}")
        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            raise ValueError(f"获取 {side} 末端到夹爪 TF 失败")
        translation, _ = transform
        rotation = ypr_to_rotation_matrix(current_pose[3:6])
        return np.asarray(current_pose[:3], dtype=float) + rotation.dot(translation)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        try:
            if self.retract_offset_m < 0.0:
                raise ValueError("new_claw_release_retract_offset_m 必须非负")
            services = self.blackboard.get(self.services_key)
            arm_controller = services.arm_controller
            left_grasp, right_grasp = [
                np.asarray(point, dtype=float) for point in self.blackboard.get(self.grasp_pair_key)
            ]
            side_axis = left_grasp - right_grasp
            norm = float(np.linalg.norm(side_axis))
            if norm < 1e-6:
                raise ValueError(f"抓取左右边缘点重合，无法确定侧向撤离方向: {left_grasp}")
            side_axis /= norm
            left_current = self._current_claw_point(arm_controller, "left")
            right_current = self._current_claw_point(arm_controller, "right")
            left_target = left_current + side_axis * self.retract_offset_m
            right_target = right_current - side_axis * self.retract_offset_m
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 计算新夹爪松爪后侧向避让点失败: {exc}"
            )
            return Status.FAILURE

        self.blackboard.set(self.left_target_key, left_target, overwrite=True)
        self.blackboard.set(self.right_target_key, right_target, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算新夹爪松爪后侧向避让点: "
            f"retract={self.retract_offset_m:.3f}, "
            f"left_current=({left_current[0]:.3f},{left_current[1]:.3f},{left_current[2]:.3f}), "
            f"right_current=({right_current[0]:.3f},{right_current[1]:.3f},{right_current[2]:.3f}), "
            f"left_target=({left_target[0]:.3f},{left_target[1]:.3f},{left_target[2]:.3f}), "
            f"right_target=({right_target[0]:.3f},{right_target[1]:.3f},{right_target[2]:.3f})"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxNewClawReleaseTargets start: "
            f"grasp_pair_key={self.grasp_pair_key}, retract={self.retract_offset_m:.3f}"
        )
