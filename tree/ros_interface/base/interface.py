"""ROS1/ROS2 通用接口定义。

这里不尝试封装完整 ROS API，只声明当前行为树运行器实际需要的最小能力。
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from tree.ros_interface.base.publisher import StringPublisher
from tree.ros_interface.base.timer import RosTimer


class RosInterface(ABC):
    """行为树运行器依赖的 ROS 能力集合。"""

    @abstractmethod
    def init(self, args: Optional[list] = None):
        """初始化 ROS 客户端库。"""

    @abstractmethod
    def shutdown(self):
        """关闭 ROS 客户端库。"""

    @abstractmethod
    def ok(self) -> bool:
        """返回 ROS 运行时是否仍然可用。"""

    @abstractmethod
    def spin(self):
        """进入 ROS 回调循环。"""

    @abstractmethod
    def start_background_spin(self):
        """在后台线程中进入 ROS 回调循环。"""

    @abstractmethod
    def destroy_node(self):
        """销毁底层 ROS 节点。"""

    @abstractmethod
    def get_logger(self):
        """返回兼容 info/warning/error 的日志对象。"""

    @abstractmethod
    def get_name(self) -> str:
        """返回当前 ROS 节点名。"""

    @abstractmethod
    def get_namespace(self) -> str:
        """返回当前 ROS 命名空间。"""

    @abstractmethod
    def get_native_node(self):
        """返回底层 ROS 节点对象，供少数 ROS 专用扩展使用。"""

    @abstractmethod
    def get_package_share_directory(self, package_name: str) -> str:
        """返回包资源目录。"""

    @abstractmethod
    def get_param(self, name: str, default_value):
        """读取参数；不存在时返回默认值。"""

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, object]):
        """批量设置参数。"""

    @abstractmethod
    def sleep(self, duration_sec: float):
        """阻塞等待指定秒数。"""

    @abstractmethod
    def now(self):
        """返回当前 ROS 时间戳对象，可直接写入消息 header.stamp。"""

    @abstractmethod
    def zero_time(self):
        """返回 ROS 时间 0，通常表示 TF 查询最新可用变换。"""

    @abstractmethod
    def duration(self, duration_sec: float):
        """返回当前 ROS 运行时可用的 Duration 对象。"""

    @abstractmethod
    def create_timer(self, period_sec: float, callback: Callable[[], None]) -> RosTimer:
        """创建周期定时器。"""

    @abstractmethod
    def create_publisher(
        self,
        topic_name: str,
        message_type,
        queue_size: int = 10,
        latch: bool = False,
    ):
        """创建通用发布器。"""

    @abstractmethod
    def create_subscription(
        self,
        topic_name: str,
        message_type,
        callback: Callable[[object], None],
        queue_size: int = 10,
        latch: bool = False,
    ):
        """创建通用消息订阅器，回调入参保留原始消息对象。"""

    @abstractmethod
    def create_string_publisher(
        self,
        topic_name: str,
        queue_size: int = 10,
        latch: bool = False,
    ) -> StringPublisher:
        """创建字符串发布器。"""

    @abstractmethod
    def create_string_subscription(
        self,
        topic_name: str,
        callback: Callable[[str], None],
        queue_size: int = 10,
        latch: bool = False,
    ):
        """创建字符串订阅器，回调入参统一为字符串。"""

    @abstractmethod
    def call_trigger_service(self, service_name: str, timeout_sec: float = 5.0):
        """调用 std_srvs/Trigger 服务，返回原始响应对象。"""

    def supports_py_trees_ros_viewer(self) -> bool:
        """当前运行时是否支持现有 py_trees_ros viewer 适配。"""
        return False
