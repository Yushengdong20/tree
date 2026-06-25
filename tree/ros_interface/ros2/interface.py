"""ROS2 rclpy 适配实现。"""

import threading

from tree.ros_interface.base import RosInterface
from tree.ros_interface.ros2.publisher import Ros2StringPublisher
from tree.ros_interface.ros2.timer import Ros2Timer


class Ros2Interface(RosInterface):
    """基于 rclpy 的运行时适配器。"""

    def __init__(self, node_name: str):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String

        self._rclpy = rclpy
        self._node_type = Node
        self._durability_policy = DurabilityPolicy
        self._qos_profile_type = QoSProfile
        self._string_type = String
        self._node_name = node_name
        self._node = None
        self._spin_thread = None

    def init(self, args=None):
        if self._node is not None:
            return
        # ROS2 需要先初始化 rclpy，再创建真正的 Node。
        self._rclpy.init(args=args)
        self._node = self._node_type(self._node_name)

    def shutdown(self):
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def ok(self) -> bool:
        return self._rclpy.ok()

    def spin(self):
        self._rclpy.spin(self._node)

    def start_background_spin(self):
        if self._spin_thread is not None:
            return
        self._spin_thread = threading.Thread(target=self.spin, daemon=True)
        self._spin_thread.start()

    def destroy_node(self):
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    def get_logger(self):
        return self._node.get_logger()

    def get_name(self) -> str:
        return self._node.get_name()

    def get_namespace(self) -> str:
        return self._node.get_namespace()

    def get_native_node(self):
        return self._node

    def get_package_share_directory(self, package_name: str) -> str:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory(package_name)

    def get_param(self, name: str, default_value):
        # ROS2 参数需要先声明，重复声明时直接读已有值。
        if not self._node.has_parameter(name):
            self._node.declare_parameter(name, default_value)
        return self._node.get_parameter(name).value

    def set_parameters(self, parameters):
        from rclpy.parameter import Parameter

        parameter_items = []
        for name, value in parameters.items():
            # ROS2 默认不允许设置未声明参数；适配层统一补齐声明步骤。
            if not self._node.has_parameter(name):
                self._node.declare_parameter(name, value)
            else:
                parameter_items.append(Parameter(name, value=value))
        if not parameter_items:
            return
        self._node.set_parameters(parameter_items)

    def sleep(self, duration_sec: float):
        import time

        time.sleep(duration_sec)

    def now(self):
        return self._node.get_clock().now().to_msg()

    def zero_time(self):
        from builtin_interfaces.msg import Time

        return Time(sec=0, nanosec=0)

    def duration(self, duration_sec: float):
        from rclpy.duration import Duration

        return Duration(seconds=float(duration_sec))

    def create_timer(self, period_sec, callback):
        return Ros2Timer(self._node.create_timer(period_sec, callback))

    def create_publisher(self, topic_name, message_type, queue_size=10, latch=False):
        qos = queue_size
        if latch:
            qos = self._qos_profile_type(
                depth=1,
                durability=self._durability_policy.TRANSIENT_LOCAL,
            )
        return self._node.create_publisher(message_type, topic_name, qos)

    def create_subscription(self, topic_name, message_type, callback, queue_size=10, latch=False):
        qos = queue_size
        if latch:
            qos = self._qos_profile_type(
                depth=1,
                durability=self._durability_policy.TRANSIENT_LOCAL,
            )
        return self._node.create_subscription(
            message_type,
            topic_name,
            callback,
            qos,
        )

    def create_string_publisher(self, topic_name, queue_size=10, latch=False):
        qos = queue_size
        if latch:
            qos = self._qos_profile_type(
                depth=1,
                durability=self._durability_policy.TRANSIENT_LOCAL,
            )
        publisher = self._node.create_publisher(self._string_type, topic_name, qos)
        return Ros2StringPublisher(publisher, self._string_type)

    def create_string_subscription(self, topic_name, callback, queue_size=10, latch=False):
        qos = queue_size
        if latch:
            qos = self._qos_profile_type(
                depth=1,
                durability=self._durability_policy.TRANSIENT_LOCAL,
            )

        def _on_message(message):
            callback(message.data.strip())

        return self._node.create_subscription(
            self._string_type,
            topic_name,
            _on_message,
            qos,
        )

    def call_trigger_service(self, service_name: str, timeout_sec: float = 5.0):
        from std_srvs.srv import Trigger

        client = self._node.create_client(Trigger, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(f"Trigger service not available: {service_name}")
        future = client.call_async(Trigger.Request())
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError(f"Trigger service call timeout: {service_name}")
        return future.result()

    def supports_py_trees_ros_viewer(self) -> bool:
        return True
