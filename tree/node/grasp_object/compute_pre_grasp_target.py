"""在执行前复核选中手臂并计算预抓取目标。"""

import py_trees

from .helper.grasp_compute_base import GraspComputeBase


class ComputePreGraspTarget(GraspComputeBase):
    """基于最终抓取目标计算预抓取目标。"""

    runtime_code = "PREGRASP_COMPUTE"
    runtime_message = "Computing selected-arm pregrasp target from final grasp target"

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.pregrasp_pose_key = str(
            params.get("pregrasp_pose_key", "grasp_object_pregrasp_pose")
        ).strip()
        if not self.pregrasp_pose_key:
            raise ValueError("pregrasp_pose_key 不能为空")
        self.blackboard.register_key(
            key=self.pregrasp_pose_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.computer.selected_grasp_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.computer.selected_base_grasp_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.computer.selected_map_grasp_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.computer.selected_rotate_z_180_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.computer.selected_arm_side_key,
            access=py_trees.common.Access.READ,
        )

    def run_grasp_compute(self):
        self.computer.refresh_selected_grasp_target()
        return self.computer.compute_pregrasp_target(self.pregrasp_pose_key)

    def failure_message(self, exc):
        return f"计算选中手臂预抓取目标失败: {exc}"

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputePreGraspTarget start: "
            f"pregrasp_pose_key={self.pregrasp_pose_key}"
        )
