"""ROS1 定时器适配。"""

from tree.ros_interface.base import RosTimer


class Ros1Timer(RosTimer):
    """把 rospy.Timer 包装成项目统一的 timer 接口。"""

    def __init__(self, timer):
        self._timer = timer
        self._canceled = False

    def cancel(self):
        self._timer.shutdown()
        self._canceled = True

    def is_canceled(self) -> bool:
        return self._canceled
