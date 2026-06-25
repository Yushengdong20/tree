"""运行器配置对象。

这个文件把 runner 需要的所有运行时参数集中到一个 dataclass 中：
- main.py 用它准备默认值
- runner.py 用它声明 / 读取 ROS 参数

这样可以避免配置项散落在多个文件里。
"""

import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BehaviorTreeRunnerConfig:
    """Runtime configuration collected from ROS parameters.

    这里既承担“默认值容器”的角色，也承担“ROS 参数读取结果”的角色。
    main.py 会先构造一份默认配置，再由 runner 用它来声明/读取实际参数。
    """

    tree_json_file: str
    tick_period_ms: int
    enable_web_viewer: bool
    web_viewer_host: str
    web_viewer_port: int
    stop_on_terminal_state: bool
    manual_result_mode: bool
    enable_manual_result_input: bool
    manual_result_topic: str
    waiting_nodes_topic: str
    enable_py_trees_ros_viewer: bool
    py_trees_ros_viewer_snapshot_period: float

    @staticmethod
    def default_tree_file(ros=None) -> str:
        """返回安装后默认树文件路径。"""
        if ros is not None:
            package_dir = ros.get_package_share_directory("mercurytree")
        else:
            # 没有运行时对象时保留源码目录兜底，方便单元测试直接构造默认配置。
            package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(
            package_dir,
            "config",
            "tree",
            "demo",
            "py_tree_complex.json",
        )

    @classmethod
    def with_defaults(
        cls,
        ros=None,
        tree_json_file: Optional[str] = None,
        tick_period_ms: int = 200,
        enable_web_viewer: bool = True,
        web_viewer_host: str = "127.0.0.1",
        web_viewer_port: int = 8765,
        stop_on_terminal_state: bool = False,
        manual_result_mode: bool = True,
        enable_manual_result_input: bool = True,
        manual_result_topic: str = "/pytrees_ros2_runner/manual_result",
        waiting_nodes_topic: str = "/pytrees_ros2_runner/waiting_nodes",
        enable_py_trees_ros_viewer: bool = True,
        py_trees_ros_viewer_snapshot_period: float = math.inf,
    ) -> "BehaviorTreeRunnerConfig":
        """创建一份既能当默认值、又能当运行配置的配置对象。"""
        return cls(
            tree_json_file=tree_json_file or cls.default_tree_file(ros=ros),
            tick_period_ms=tick_period_ms,
            enable_web_viewer=enable_web_viewer,
            web_viewer_host=web_viewer_host,
            web_viewer_port=web_viewer_port,
            stop_on_terminal_state=stop_on_terminal_state,
            manual_result_mode=manual_result_mode,
            enable_manual_result_input=enable_manual_result_input,
            manual_result_topic=manual_result_topic,
            waiting_nodes_topic=waiting_nodes_topic,
            enable_py_trees_ros_viewer=enable_py_trees_ros_viewer,
            py_trees_ros_viewer_snapshot_period=py_trees_ros_viewer_snapshot_period,
        )

    @classmethod
    def from_ros_interface(
        cls,
        ros,
        defaults: Optional["BehaviorTreeRunnerConfig"] = None,
    ) -> "BehaviorTreeRunnerConfig":
        """在一个地方完成参数声明与参数读取。"""
        defaults = defaults or cls.with_defaults(ros=ros)

        # 参数读取统一交给 ros_interface，内部分别适配 ROS1 get_param 和 ROS2 declare/get。
        return cls(
            tree_json_file=ros.get_param("tree_json_file", defaults.tree_json_file)
            or defaults.tree_json_file,
            tick_period_ms=int(ros.get_param("tick_period_ms", defaults.tick_period_ms)),
            enable_web_viewer=bool(
                ros.get_param("enable_web_viewer", defaults.enable_web_viewer)
            ),
            web_viewer_host=str(ros.get_param("web_viewer_host", defaults.web_viewer_host)),
            web_viewer_port=int(ros.get_param("web_viewer_port", defaults.web_viewer_port)),
            stop_on_terminal_state=bool(
                ros.get_param("stop_on_terminal_state", defaults.stop_on_terminal_state)
            ),
            manual_result_mode=bool(
                ros.get_param("manual_result_mode", defaults.manual_result_mode)
            ),
            enable_manual_result_input=bool(
                ros.get_param(
                    "enable_manual_result_input",
                    defaults.enable_manual_result_input,
                )
            ),
            manual_result_topic=str(
                ros.get_param("manual_result_topic", defaults.manual_result_topic)
            ),
            waiting_nodes_topic=str(
                ros.get_param("waiting_nodes_topic", defaults.waiting_nodes_topic)
            ),
            enable_py_trees_ros_viewer=bool(
                ros.get_param(
                    "enable_py_trees_ros_viewer",
                    defaults.enable_py_trees_ros_viewer,
                )
            ),
            py_trees_ros_viewer_snapshot_period=float(
                ros.get_param(
                    "py_trees_ros_viewer_snapshot_period",
                    defaults.py_trees_ros_viewer_snapshot_period,
                )
            ),
        )
