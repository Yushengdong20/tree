"""ROS 运行时适配层。

这个包只暴露项目内部需要的 ROS 能力，具体 ROS1/ROS2 差异由子模块处理。
"""

from tree.ros_interface.factory import create_ros_interface

__all__ = ["create_ros_interface"]
