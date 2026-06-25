"""行为树运行器。

这是整个项目最核心的运行中枢，负责把以下几件事情串起来：
- 从 JSON 配置加载行为树
- 周期性 tick 行为树
- 维护快照与运行时状态
- 接入自定义 Web Viewer
- 接入官方 py_trees_ros viewer
- 接入手动结果输入 / topic 控制
"""

import os
from typing import Optional

import py_trees

from tree.core.manual_input import ManualResultController, RawKeyInputController
from tree.core.runner_config import BehaviorTreeRunnerConfig
from tree.core.tree_factory import BehaviorTreeFactory
from tree.visualization.ros_viewer import enable_py_trees_ros_viewer_support
from tree.visualization.snapshot import BehaviorTreeSnapshotStore
from tree.visualization.web_viewer import BehaviorTreeWebViewer


class BehaviorTreeRunner:
    """ROS runtime host responsible for loading, ticking and exposing the behaviour tree.

    可以把这个类理解成整个项目的“运行中枢”：

    1. 从 JSON 配置加载行为树
    2. 用定时器周期性 tick 行为树
    3. 给自定义 Web Viewer 提供快照
    4. 给官方 py_trees_ros viewer 提供快照服务
    5. 统一接入手动结果输入 / topic 控制

    它本身尽量不实现具体业务动作，只负责把各个子模块串起来。
    """

    def __init__(self, ros, default_config: Optional[BehaviorTreeRunnerConfig] = None):
        # runner 不再继承具体 ROS 节点，而是通过 ros_interface 使用 ROS 能力。
        # 这样 ROS1/ROS2 差异会被限制在适配层里，业务树仍然面对同一个宿主对象。
        self.ros = ros

        # main.py 可以传入一份显式默认配置，同时仍然保留 ROS 参数覆盖能力。
        # 这一步之后，runner 看到的是“已经融合默认值和 ROS 参数覆盖”的最终配置。
        self.config = BehaviorTreeRunnerConfig.from_ros_interface(
            self.ros,
            defaults=default_config,
        )

        # factory 负责从 JSON 构造 py_trees 运行时对象。
        # 到这里为止，树还只是普通 py_trees 树，并没有 viewer、快照等 ROS 能力。
        self.factory = BehaviorTreeFactory(self)
        tree = self.factory.load_tree_from_json(self.config.tree_json_file)
        # 如果开启官方 viewer 支持，这里会在现有树对象外再包一层 py_trees_ros 接口。
        # 注意：包的是“ROS 通信能力”，不是重建业务树本身。
        if self.config.enable_py_trees_ros_viewer:
            tree, viewer_enabled = enable_py_trees_ros_viewer_support(
                self,
                tree,
                snapshot_period=self.config.py_trees_ros_viewer_snapshot_period,
            )
            self.config.enable_py_trees_ros_viewer = viewer_enabled
        self.tree = tree

        # 运行时辅助组件分别负责快照和终端手动输入，不再堆在入口文件里。
        # snapshot_store: 自定义 Web Viewer 读取的统一快照源
        # manual_input  : 主程序终端输入 / 外部 topic 输入的统一入口
        self.snapshot_store = BehaviorTreeSnapshotStore()
        self.manual_input = ManualResultController(self)
        self.raw_key_input = RawKeyInputController(self)
        # live_runtime 是“标准树快照”之外的补充提示信息。
        # 它不代表额外 tick，只是告诉前端：当前这个长耗时步骤正在做什么。
        self.live_runtime = None
        # tick_count 代表树已经真正推进了多少次 tick。
        # 它只应该在 tick_tree() 中递增，不应该被其他旁路刷新修改。
        self.tick_count = 0
        self.manual_result_subscriber = None
        self.waiting_nodes_publisher = None

        self.web_viewer = None
        if self.config.enable_web_viewer:
            tree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 自定义 Web Viewer 本质上是一个小型 HTTP 服务，
            # 它定期来取 get_snapshot() 返回的序列化树状态。
            self.web_viewer = BehaviorTreeWebViewer(
                snapshot_provider=self.get_snapshot,
                host=self.config.web_viewer_host,
                port=self.config.web_viewer_port,
                static_dir=os.path.join(tree_root, "visualization", "web"),
            )
            self.web_viewer.start()
            self.get_logger().info(
                f"Web viewer: http://{self.config.web_viewer_host}:{self.config.web_viewer_port}"
            )

        if self.config.enable_py_trees_ros_viewer:
            # 官方 viewer 走 py_trees_ros 的 snapshots/services 体系。
            # 这里打印的是它的服务根路径，方便用户确认 viewer 该连到哪里。
            namespace = f"{self.get_namespace().rstrip('/')}/{self.get_name()}".replace("//", "/")
            self.get_logger().info(
                f"py_trees_ros viewer services ready under: {namespace}/snapshot_streams"
            )

        # 本地终端输入和外部 topic 控制都走同一套 ManualResultController。
        # manual_result_mode=True 代表“节点结果由人工控制”，
        # enable_manual_result_input=True 只是说明“人工控制来自当前终端”。
        if self.config.manual_result_mode and self.config.enable_manual_result_input:
            self.manual_input.start_input_thread()
            self.get_logger().info(
                "Manual result mode enabled. Input: s/f/r, or '<node_label> s/f/r' when multiple nodes are waiting"
            )
        elif self.config.manual_result_mode:
            self.get_logger().info(
                f"Manual result topic enabled: {self.config.manual_result_topic}"
            )

        if self.config.manual_result_mode:
            # 外部测试脚本可以通过这个 topic 投递和终端一致的控制指令。
            # 这意味着“本地终端输入”和“外部 sender/topic 输入”最终都走同一套解析逻辑。
            self.manual_result_subscriber = self.create_subscription(
                self.config.manual_result_topic,
                self._on_manual_result_message,
                10,
            )
            self.waiting_nodes_publisher = self.create_string_publisher(
                self.config.waiting_nodes_topic,
                latch=True,
            )
            self._publish_waiting_nodes()

        if not self.config.enable_manual_result_input:
            self.raw_key_input.start_input_thread()
            self.get_logger().info("Raw key input enabled on main terminal: Enter / s / o")

        # 定时器是整棵树的主循环入口，每次触发就 tick 一次。
        # 也就是说，这个类不是 while True 主动循环，而是把节奏交给 ROS timer。
        self.timer = self.create_timer(
            self.config.tick_period_ms / 1000.0,
            self.tick_tree,
        )
        self.get_logger().info(f"Loaded tree: {self.config.tree_json_file}")

    def get_logger(self):
        """保留原 ROS node 风格日志接口，减少业务节点改动。"""
        return self.ros.get_logger()

    def get_name(self):
        """返回底层 ROS 节点名。"""
        return self.ros.get_name()

    def get_namespace(self):
        """返回底层 ROS 命名空间。"""
        return self.ros.get_namespace()

    def set_parameters(self, parameters):
        """转发参数设置给底层 ROS 运行时。"""
        if isinstance(parameters, dict):
            self.ros.set_parameters(parameters)
            return

        # 兼容少数 ROS2 风格调用：对象里有 name/value 属性时转换成普通 dict。
        converted = {}
        for parameter in parameters:
            converted[parameter.name] = parameter.value
        self.ros.set_parameters(converted)

    def get_param(self, name: str, default_value):
        """读取参数，统一交给 ros_interface 处理 ROS1/ROS2 差异。"""
        return self.ros.get_param(name, default_value)

    def sleep(self, duration_sec: float):
        """阻塞等待，统一交给 ros_interface 处理 ROS1/ROS2 差异。"""
        self.ros.sleep(duration_sec)

    def ok(self) -> bool:
        """返回 ROS 运行时是否仍然可用。"""
        return self.ros.ok()

    def now(self):
        """返回当前 ROS 时间戳。"""
        return self.ros.now()

    def zero_time(self):
        """返回 ROS 时间 0，通常用于 TF 查询最新可用变换。"""
        return self.ros.zero_time()

    def duration(self, duration_sec: float):
        """返回当前 ROS 运行时可用的 Duration 对象。"""
        return self.ros.duration(duration_sec)

    def create_timer(self, period_sec, callback):
        """创建定时器。"""
        return self.ros.create_timer(period_sec, callback)

    def create_publisher(self, topic_name, message_type, queue_size=10, latch=False):
        """创建通用发布器。"""
        return self.ros.create_publisher(
            topic_name,
            message_type,
            queue_size=queue_size,
            latch=latch,
        )

    def create_message_subscription(self, topic_name, message_type, callback, queue_size=10, latch=False):
        """创建通用消息订阅器。"""
        return self.ros.create_subscription(
            topic_name,
            message_type,
            callback,
            queue_size=queue_size,
            latch=latch,
        )

    def create_string_publisher(self, topic_name, queue_size=10, latch=False):
        """创建字符串发布器。"""
        return self.ros.create_string_publisher(
            topic_name,
            queue_size=queue_size,
            latch=latch,
        )

    def create_subscription(self, topic_name, callback, queue_size=10, latch=False):
        """创建字符串订阅器。"""
        return self.ros.create_string_subscription(
            topic_name,
            callback,
            queue_size=queue_size,
            latch=latch,
        )

    def call_trigger_service(self, service_name: str, timeout_sec: float = 5.0):
        """调用 std_srvs/Trigger 服务。"""
        return self.ros.call_trigger_service(service_name, timeout_sec=timeout_sec)

    def set_blackboard_value(self, key: str, value):
        """写入全局 blackboard，供输入线程或运行时快捷控制使用。"""
        py_trees.blackboard.Blackboard.set(key, value)

    def consume_key_event(self, *expected_keys: str):
        """Consume one raw key event captured from the main terminal."""
        return self.raw_key_input.consume_key(*expected_keys)

    def tick_tree(self):
        """Advance the tree one step and refresh visualisation snapshots."""
        # tick 先推进树，再刷新对外展示用快照。
        # 这是“标准行为树节奏”的核心位置：
        # 1. tree.tick() 真正推进一次树
        # 2. tick_count + 1
        # 3. snapshot_store 记录这次 tick 之后的状态
        self.tree.tick()
        self.tick_count += 1
        self.snapshot_store.update(
            self.tree,
            tick_count=self.tick_count,
            timer=self.timer,
            live_runtime=self.live_runtime,
        )

        root_status = self.tree.root.status
        if self.config.stop_on_terminal_state and root_status in (
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
        ):
            # 到达根节点终态后可以选择自动停表，这样树会停在最终状态供观察。
            self.timer.cancel()
            self.snapshot_store.update(
                self.tree,
                tick_count=self.tick_count,
                timer=self.timer,
                execution_state="STOPPED",
                live_runtime=self.live_runtime,
            )
            self.get_logger().info(f"Tree reached terminal status: {root_status.name}")

    def get_snapshot(self):
        """Expose the latest snapshot to the custom web viewer."""
        return self.snapshot_store.get_snapshot()

    def refresh_runtime_snapshot(self, execution_state: Optional[str] = None):
        """Refresh only the live runtime side-channel without touching tick statistics."""
        # 这里故意不重建整份 snapshot，也不修改 tick_count / last_tick_interval，
        # 否则前端会把“运行中补充提示”误认为真的发生了一次 tick。
        del execution_state
        self.snapshot_store.refresh_live_runtime(self.live_runtime)

    def set_live_runtime(self, active_node: str, phase: str, detail: str):
        """Record the currently blocking business step for web visibility."""
        # 给自定义 Web 的 Live Step 面板写入“当前活跃节点/阶段/说明”。
        # 典型用途是导航轮询、对正检测、抓箱调用这类耗时步骤。
        self.live_runtime = {
            "active_node": active_node,
            "phase": phase,
            "detail": detail,
        }
        self.refresh_runtime_snapshot(execution_state="RUNNING")

    def clear_live_runtime(self):
        # 当前长耗时步骤结束后，清掉这条补充提示。
        self.live_runtime = None
        self.refresh_runtime_snapshot()

    def _on_manual_result_message(self, data: str):
        """Accept the same command format from ROS topics as from the terminal prompt."""
        # 让“外部 sender”与“本地终端输入”复用同一套解析逻辑，避免两边行为不一致。
        self.manual_input.handle_input(data.strip())

    def consume_manual_result(self, node_label: str):
        # 这些方法只是把 ManualResultController 的能力转发给叶子节点，
        # runner 自己不重复保存手动结果数据。
        return self.manual_input.consume_result(node_label)

    def mark_manual_result_waiting(self, node_label: str, waiting: bool):
        # 当节点进入/离开“等待人工结果”状态时：
        # 1. 更新本地 waiting 集合
        # 2. 同步发布 waiting_nodes topic
        self.manual_input.mark_waiting(node_label, waiting)
        self._publish_waiting_nodes()

    def get_waiting_manual_nodes(self):
        return self.manual_input.get_waiting_nodes()

    def _publish_waiting_nodes(self):
        """Publish the current waiting list so external test terminals can render it."""
        if self.waiting_nodes_publisher is None:
            return
        # topic 里用逗号拼接一个轻量字符串，外部 sender 收到后自己再拆开渲染。
        self.waiting_nodes_publisher.publish(",".join(self.get_waiting_manual_nodes()))

    def destroy_node(self):
        """Stop auxiliary services before handing destruction back to ROS runtime."""
        # 这里优先关闭我们自己启动的附属服务，再交回给 ROS 适配层做节点销毁。
        if self.web_viewer:
            self.web_viewer.stop()
        self.raw_key_input.stop()
        self.ros.destroy_node()
