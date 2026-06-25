"""ROS2 launch 入口。

这个文件提供一种“不改 Python 源码，只靠 launch 参数切换树和 viewer 配置”的启动方式。
它本身不包含业务逻辑，只负责把 launch 参数透传给 bt_runner。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # launch 文件主要做参数透传，让你不改 Python 代码也能切换树文件和 viewer 开关。
    return LaunchDescription(
        [
            DeclareLaunchArgument("tree_json_file", default_value=""),
            DeclareLaunchArgument("tick_period_ms", default_value="200"),
            DeclareLaunchArgument("enable_web_viewer", default_value="true"),
            DeclareLaunchArgument("web_viewer_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("web_viewer_port", default_value="8765"),
            DeclareLaunchArgument("enable_py_trees_ros_viewer", default_value="true"),
            Node(
                package="mercurytree",
                executable="bt_runner",
                name="pytrees_ros2_runner",
                output="screen",
                # LaunchConfiguration 会把 launch 参数映射成节点启动时的 ROS 参数。
                parameters=[
                    {
                        "tree_json_file": LaunchConfiguration("tree_json_file"),
                        "tick_period_ms": LaunchConfiguration("tick_period_ms"),
                        "enable_web_viewer": LaunchConfiguration("enable_web_viewer"),
                        "web_viewer_host": LaunchConfiguration("web_viewer_host"),
                        "web_viewer_port": LaunchConfiguration("web_viewer_port"),
                        "enable_py_trees_ros_viewer": LaunchConfiguration("enable_py_trees_ros_viewer"),
                    }
                ],
            ),
        ]
    )
