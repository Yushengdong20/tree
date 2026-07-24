"""移动到采样位姿后复核并刷新同步双抓预抓取目标。"""

import py_trees

from .helper.grasp_compute_base import GraspComputeBase


class ComputeDualPreGraspTargets(GraspComputeBase):
    """基于同步双抓最终目标，刷新左右手预抓取位姿。"""

    runtime_code = "DUAL_PREGRASP_COMPUTE"
    runtime_message = "Refreshing synchronized dual pregrasp targets"
    failure_log_level = "warning"

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 关键步骤：这些 key 在规划节点写入，本节点负责移动后读取并复核。
        for key in (
            self.computer.left_selected_grasp_pose_key,
            self.computer.right_selected_grasp_pose_key,
            self.computer.left_selected_base_grasp_pose_key,
            self.computer.right_selected_base_grasp_pose_key,
            self.computer.left_selected_map_grasp_pose_key,
            self.computer.right_selected_map_grasp_pose_key,
            self.computer.left_selected_rotate_z_180_key,
            self.computer.right_selected_rotate_z_180_key,
            self.computer.dual_selected_pair_info_key,
        ):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def run_grasp_compute(self):
        self.computer.refresh_dual_selected_grasp_targets()
        return self.computer.compute_dual_pregrasp_targets()

    def failure_message(self, exc):
        return f"同步双抓预抓取目标复核失败: {exc}"

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeDualPreGraspTargets start: "
            f"left_pregrasp={self.computer.left_pregrasp_pose_key}, "
            f"right_pregrasp={self.computer.right_pregrasp_pose_key}"
        )
