"""ROS2 字符串发布器适配。"""

from tree.ros_interface.base import StringPublisher


class Ros2StringPublisher(StringPublisher):
    """把 rclpy publisher 包装成统一字符串发布器。"""

    def __init__(self, publisher, string_type):
        self._publisher = publisher
        self._string_type = string_type

    def publish(self, data: str):
        message = self._string_type()
        message.data = data
        self._publisher.publish(message)
