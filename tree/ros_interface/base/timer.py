"""ROS 定时器通用接口。"""

from abc import ABC, abstractmethod


class RosTimer(ABC):
    """统一 ROS 定时器接口。"""

    @abstractmethod
    def cancel(self):
        """停止定时器。"""

    @abstractmethod
    def is_canceled(self) -> bool:
        """返回定时器是否已经停止。"""
