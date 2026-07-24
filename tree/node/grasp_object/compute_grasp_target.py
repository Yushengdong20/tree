"""在当前腰部位姿下计算双手最优抓取目标。"""

from .helper.grasp_compute_base import GraspComputeBase


class ComputeGraspTarget(GraspComputeBase):
    """只使用当前腰部位姿选择双手抓取目标，不做腰部采样。"""

    runtime_code = "GRASP_COMPUTE"
    runtime_message = "Computing dual-arm grasp target with current torso"

    def run_grasp_compute(self):
        return self.computer.compute_current_torso_target()

    def log_grasp_success(self, selected):
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算当前腰部双手抓取目标: "
            f"side={selected['arm_side']}, torso={sample['label']}, grasp={selected['grasp_target']}"
        )

    def failure_message(self, exc):
        return f"当前腰部下计算双手抓取目标失败: {exc}"

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeGraspTarget start: "
            f"base_grasp_poses_key={self.computer.base_grasp_poses_key}"
        )
