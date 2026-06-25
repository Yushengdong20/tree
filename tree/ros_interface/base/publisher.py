"""ROS 字符串发布器通用接口。"""

from abc import ABC, abstractmethod


class StringPublisher(ABC):
    """统一字符串发布器接口。"""

    @abstractmethod
    def publish(self, data: str):
        """发布一条字符串消息。"""
