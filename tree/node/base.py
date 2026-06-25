"""叶子节点公共基类。

这个文件定义了项目里大多数业务叶子节点共享的一套基础行为：
- 进入节点时如何打印开始日志
- 手动模式下如何等待人工结果
- 结束时如何打印完成日志

因此可以把它理解成“示例节点和真实节点共用的最小行为底座”。
"""

import time

import py_trees


class TimedMockAction(py_trees.behaviour.Behaviour):
    """项目内大多数叶子节点的公共基类。"""

    # 默认允许在 manual_result_mode 下被 s/f/r 接管；
    # 键盘请求、流程门控这类控制节点可以在子类里关闭它。
    allow_manual_result_override = True

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name)
        # name 更接近节点类型名，config_label 更接近 JSON 中的业务标签名。
        self.config_label = config_label
        self.ros_node = ros_node
        self.params = params
        # 每个叶子节点使用自己的 blackboard client；具体业务 key 由具体节点自己注册。
        self.blackboard = py_trees.blackboard.Client(name=config_label)
        self.result = str(params.get("result", "SUCCESS")).upper()
        self._started_at = None
        self._waiting_logged = False

    def _manual_result_mode_enabled(self) -> bool:
        """Read the manual mode flag from the current runner state instead of caching it once."""
        if hasattr(self.ros_node, "config"):
            return bool(getattr(self.ros_node.config, "manual_result_mode", False))
        return bool(getattr(self.ros_node, "manual_result_mode", False))

    def should_use_mock_execution(self) -> bool:
        """Use manual/mock execution whenever the runner is in manual result mode.

        这样真实业务节点在“纯软件验证流程”时也能复用同一套 s/f/r 控制逻辑，
        避免一进入节点就直接访问真实 HTTP 服务。
        """
        return self.allow_manual_result_override and self._manual_result_mode_enabled()

    @staticmethod
    def _to_bool(value) -> bool:
        """兼容 JSON 中常见的字符串布尔值。"""
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def should_skip_torso_motion(self) -> bool:
        """测试腰部异常时，允许节点保留流程但跳过真实躯干动作。"""
        return self._skip_motion_enabled("skip_torso_motion")

    def log_skip_torso_motion(self):
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 已跳过腰部动作: skip_torso_motion=True"
        )

    def should_skip_head_motion(self) -> bool:
        """测试时允许保留流程但跳过真实头部动作。"""
        return self._skip_motion_enabled("skip_head_motion")

    def log_skip_head_motion(self):
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 已跳过头部动作: skip_head_motion=True"
        )

    def should_skip_arm_motion(self) -> bool:
        """测试时允许保留流程但跳过真实手臂动作。"""
        return self._skip_motion_enabled("skip_arm_motion")

    def log_skip_arm_motion(self):
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 已跳过手臂动作: skip_arm_motion=True"
        )

    def should_skip_claw_motion(self) -> bool:
        """测试时允许保留流程但跳过真实夹爪开合动作。"""
        return self._skip_motion_enabled("skip_claw_motion")

    def log_skip_claw_motion(self):
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 已跳过夹爪动作: skip_claw_motion=True"
        )

    def _skip_motion_enabled(self, key: str) -> bool:
        """节点参数优先，其次读取 main.py/ROS 参数里的全局测试开关。"""
        if self._to_bool(self.params.get(key, False)):
            return True
        if hasattr(self.ros_node, "get_param"):
            return self._to_bool(self.ros_node.get_param(key, False))
        return False

    def initialise(self):
        # py_trees 在节点从非 RUNNING 切入执行时调用 initialise，一般用于重置本轮状态。
        self._started_at = time.monotonic()
        self._waiting_logged = False
        # 阻塞式真实业务节点会在 update() 内停留较久，先把自身标成 RUNNING，
        # 这样外部快照在中途刷新时也能知道当前活跃节点是谁。
        self.status = py_trees.common.Status.RUNNING
        start_message = self.describe_start()
        if start_message:
            self.ros_node.get_logger().info(start_message)

    def update(self):
        return self._update_from_config_or_manual()

    def _update_from_config_or_manual(self):
        # 节点一旦被 tick 到就先进入 RUNNING，并等待手动输入或直接返回配置结果。
        # 这条逻辑同时服务两类节点：
        # 1. 纯 mock 示例节点
        # 2. 真实 HTTP 节点在 manual_result_mode=True 时的“软件验证旁路”
        status_map = {
            "SUCCESS": py_trees.common.Status.SUCCESS,
            "FAILURE": py_trees.common.Status.FAILURE,
            "RUNNING": py_trees.common.Status.RUNNING,
        }
        result_name = self.result
        clear_waiting_after_finish = False
        if self._manual_result_mode_enabled() and hasattr(self.ros_node, "consume_manual_result"):
            # 先登记“这个节点正在等人工结果”，然后尝试消费是否已经有人喂了结果。
            self.ros_node.mark_manual_result_waiting(self.config_label, True)
            manual_result = self.ros_node.consume_manual_result(self.config_label)
            if manual_result is None:
                return py_trees.common.Status.RUNNING

            result_name = manual_result
            self._waiting_logged = False
            clear_waiting_after_finish = True

        status = status_map.get(result_name, py_trees.common.Status.SUCCESS)
        self.ros_node.get_logger().info(self.describe_finish(status))
        if clear_waiting_after_finish:
            self.ros_node.mark_manual_result_waiting(self.config_label, False)
        return status

    def update_mock_result(self):
        """Expose the mock/manual result path to business nodes that override update()."""
        return self._update_from_config_or_manual()

    def describe_start(self):
        return f"[{self.config_label}] started"

    def describe_finish(self, status):
        return f"[{self.config_label}] finished -> {status.name}"

    def terminate(self, new_status):
        """节点被打断或结束时，确保它不再出现在 waiting 列表里。"""
        # 如果节点被上层中断/切走，也要从 waiting 列表里清掉，避免提示残留。
        if self._manual_result_mode_enabled() and hasattr(self.ros_node, "mark_manual_result_waiting"):
            self.ros_node.mark_manual_result_waiting(self.config_label, False)
