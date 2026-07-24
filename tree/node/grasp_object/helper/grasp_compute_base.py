"""grasp_object compute 节点公共执行模板。"""

from concurrent.futures import ThreadPoolExecutor

import py_trees
from py_trees.common import Status

from .grasp_target_utils import GraspTargetComputer
from ...base import TimedMockAction


class GraspComputeBase(TimedMockAction):
    """同步 grasp 计算节点的公共基类。"""

    runtime_code = "GRASP_COMPUTE"
    runtime_message = "Computing grasp target"
    failure_log_level = "error"

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.blackboard = py_trees.blackboard.Client(name=name)
        self.computer = GraspTargetComputer(
            config_label=config_label,
            ros_node=ros_node,
            blackboard=self.blackboard,
            params=params,
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

        try:
            self.ros_node.set_live_runtime(
                self.config_label,
                self.runtime_code,
                self.runtime_message,
            )
            # 关键步骤：子类只描述具体计算任务，异常和状态转换统一在基类处理。
            result = self.run_grasp_compute()
        except Exception as exc:
            return self._finish_with_failure(exc)

        self.ros_node.clear_live_runtime()
        self.log_grasp_success(result)
        return Status.SUCCESS

    def run_grasp_compute(self):
        """执行具体抓取计算，子类必须实现。"""
        raise NotImplementedError

    def log_grasp_success(self, _result):
        """计算成功后的业务日志，子类按需覆盖。"""
        return

    def _finish_with_failure(self, exc):
        self.feedback_message = str(exc)
        self.ros_node.clear_live_runtime()
        log_method = getattr(self.ros_node.get_logger(), self.failure_log_level)
        log_method(f"[{self.config_label}] {self.failure_message(exc)}")
        return Status.FAILURE

    def failure_message(self, exc):
        """生成失败日志文本，子类按业务场景覆盖。"""
        return f"抓取目标计算失败: {exc}"

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)


class AsyncGraspComputeBase(GraspComputeBase):
    """后台执行 grasp 计算并通过行为树 tick 轮询结果。"""

    not_started_message = "抓取采样任务未正确启动"

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 关键步骤：耗时采样放到后台线程，避免阻塞行为树 tick 和 Web 快照刷新。
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
            self.feedback_message = self.not_started_message
            self.ros_node.clear_live_runtime()
            self.log_async_start_failure()
            return Status.FAILURE

        if not self._sample_future.done():
            self.ros_node.set_live_runtime(
                self.config_label,
                self.runtime_code,
                self.runtime_message,
            )
            return Status.RUNNING

        return self._finish_sampling_task()

    def _start_sampling_task(self):
        """提交后台采样任务，让当前 tick 尽快返回 RUNNING。"""
        self._sampling_started = True
        self.ros_node.set_live_runtime(
            self.config_label,
            self.runtime_code,
            self.runtime_message,
        )
        self._sample_future = self._executor.submit(self.run_grasp_compute)

    def _finish_sampling_task(self):
        """读取后台采样结果，并转换成行为树终态。"""
        try:
            result = self._sample_future.result()
        except Exception as exc:
            return self._finish_with_failure(exc)

        self.ros_node.clear_live_runtime()
        self._sample_future = None
        self._sampling_started = False
        self.log_grasp_success(result)
        return Status.SUCCESS

    def log_async_start_failure(self):
        log_method = getattr(self.ros_node.get_logger(), self.failure_log_level)
        log_method(f"[{self.config_label}] {self.not_started_message}")

    def terminate(self, new_status):
        # 节点被上层中断时尽量取消尚未开始的任务；已运行的线程无法被安全强杀。
        if new_status != Status.RUNNING:
            if self._sample_future is not None and not self._sample_future.done():
                self._sample_future.cancel()
            self._sample_future = None
            self._sampling_started = False
            self.ros_node.clear_live_runtime()
        TimedMockAction.terminate(self, new_status)
