"""通过底盘和腰部采样计算同步双抓目标。"""

from .helper.grasp_compute_base import AsyncGraspComputeBase


class ComputeDualGraspTargetsWithTorso(AsyncGraspComputeBase):
    """使用底盘和腰部采样，尝试让左右手同时抓取两个不同零件。"""

    runtime_code = "DUAL_TORSO_SAMPLE"
    runtime_message = "Sampling chassis/torso-assisted synchronized dual grasp targets"
    failure_log_level = "warning"
    not_started_message = "同步双抓采样任务未正确启动"

    def run_grasp_compute(self):
        return self.computer.compute_torso_sample_dual_targets()

    def log_grasp_success(self, selected):
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算采样同步双抓目标: "
            f"sample={sample['label']}, "
            f"left_object={selected['left'].get('object_index')}, "
            f"right_object={selected['right'].get('object_index')}"
        )

    def failure_message(self, exc):
        return f"采样计算同步双抓目标失败: {exc}"

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeDualGraspTargetsWithTorso start: "
            f"objects_key={self.computer.sorted_grasp_objects_key}"
        )
