"""ROS2 定时器适配。"""

from tree.ros_interface.base import RosTimer


class Ros2Timer(RosTimer):
    """把 rclpy Timer 包装成项目统一的 timer 接口。"""

    def __init__(self, timer):
        self._timer = timer

    def cancel(self):
        self._timer.cancel()

    def is_canceled(self) -> bool:
        return self._timer.is_canceled()
