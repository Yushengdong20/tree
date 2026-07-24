"""在当前腰部位姿下计算同步双抓目标。"""

from .helper.grasp_compute_base import GraspComputeBase


class ComputeDualGraspTargets(GraspComputeBase):
    """只使用当前腰部位姿，尝试让左右手同时抓取两个不同零件。"""

    runtime_code = "DUAL_GRASP_COMPUTE"
    runtime_message = "Computing current-torso synchronized dual grasp targets"
    failure_log_level = "warning"

    def run_grasp_compute(self):
        return self.computer.compute_current_torso_dual_targets()

    def log_grasp_success(self, selected):
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算当前腰部同步双抓目标: "
            f"left_object={selected['left'].get('object_index')}, "
            f"right_object={selected['right'].get('object_index')}"
        )

    def failure_message(self, exc):
        return f"当前腰部同步双抓目标计算失败: {exc}"

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeDualGraspTargets start: "
            f"objects_key={self.computer.sorted_grasp_objects_key}"
        )
