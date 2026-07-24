"""通过底盘和腰部采样计算双手最优抓取目标。"""

from .helper.grasp_compute_base import AsyncGraspComputeBase


class ComputeGraspTargetWithTorso(AsyncGraspComputeBase):
    """使用底盘和腰部采样选择可达抓取目标，并写入运动目标。"""

    runtime_code = "TORSO_SAMPLE"
    runtime_message = "Sampling chassis/torso-assisted dual-arm grasp targets"
    not_started_message = "采样任务未正确启动"

    def run_grasp_compute(self):
        return self.computer.compute_torso_sample_target()

    def log_grasp_success(self, selected):
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算双手最小移动抓取目标: "
            f"side={selected['arm_side']}, torso={sample['label']}, grasp={selected['grasp_target']}"
        )

    def failure_message(self, exc):
        return f"采样计算双手抓取目标失败: {exc}"

    def log_async_start_failure(self):
        self.ros_node.get_logger().error(
            f"[{self.config_label}] 采样计算双手抓取目标失败: {self.feedback_message}"
        )

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeGraspTargetWithTorso start: "
            f"base_grasp_poses_key={self.computer.base_grasp_poses_key}"
        )
