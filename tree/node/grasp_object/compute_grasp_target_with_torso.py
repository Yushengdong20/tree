"""通过底盘和腰部采样计算双手最优抓取目标。"""

from concurrent.futures import ThreadPoolExecutor

import py_trees
from py_trees.common import Status

from .grasp_target_utils import GraspTargetComputer
from ..base import TimedMockAction


class ComputeGraspTargetWithTorso(TimedMockAction):
    """使用底盘和腰部采样选择可达抓取目标，并写入运动目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.computer = GraspTargetComputer(
            config_label=config_label,
            ros_node=ros_node,
            blackboard=self.blackboard,
            params=params,
        )
        # 采样计算可能耗时较长，放到后台线程，避免阻塞行为树 tick 和 Web 快照刷新。
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._sample_future = None
        self._sampling_started = False

    def initialise(self):
        super().initialise()
        # 每次节点重新进入执行时都开启一轮新的采样任务。
        self._sample_future = None
        self._sampling_started = False

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        if not self._sampling_started:
            self._start_sampling_task()
            return Status.RUNNING

        if self._sample_future is None:
            self.feedback_message = "采样任务未正确启动"
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 采样计算双手抓取目标失败: {self.feedback_message}"
            )
            return Status.FAILURE

        if not self._sample_future.done():
            self.ros_node.set_live_runtime(
                self.config_label,
                "TORSO_SAMPLE",
                "Sampling chassis/torso-assisted dual-arm grasp targets",
            )
            return Status.RUNNING

        return self._finish_sampling_task()

    def _start_sampling_task(self):
        """提交后台采样任务，让当前 tick 尽快返回 RUNNING。"""
        self._sampling_started = True
        self.ros_node.set_live_runtime(
            self.config_label,
            "TORSO_SAMPLE",
            "Sampling chassis/torso-assisted dual-arm grasp targets",
        )
        # 关键步骤：耗时采样放到后台线程，后续 tick 只轮询 future 是否完成。
        self._sample_future = self._executor.submit(self.computer.compute_torso_sample_target)

    def _finish_sampling_task(self):
        """读取后台采样结果，并转换成行为树终态。"""
        try:
            selected = self._sample_future.result()
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 采样计算双手抓取目标失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.clear_live_runtime()
        self._sample_future = None
        self._sampling_started = False
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算双手最小移动抓取目标: "
            f"side={selected['arm_side']}, torso={sample['label']}, grasp={selected['grasp_target']}"
        )
        return Status.SUCCESS

    def terminate(self, new_status):
        # 节点被上层中断时尽量取消尚未开始的任务；已运行的线程无法被安全强杀。
        if new_status != Status.RUNNING:
            if self._sample_future is not None and not self._sample_future.done():
                self._sample_future.cancel()
            self._sample_future = None
            self._sampling_started = False
            self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeGraspTargetWithTorso start: "
            f"base_grasp_poses_key={self.computer.base_grasp_poses_key}"
        )
