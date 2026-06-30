"""项目入口文件。

这个文件有意保持“短而显式”：
- 不承载具体业务逻辑
- 只负责挑选默认配置
- 再把控制权交给 BehaviorTreeRunner

因此当你想快速确认“当前默认跑哪棵树、是否走 mock、是否启用 viewer”时，
先看这个文件就够了。
"""

import os

from tree.core.blackboard_bootstrap import load_blackboard_from_json
from tree.core.runner import BehaviorTreeRunner
from tree.core.runner_config import BehaviorTreeRunnerConfig
from tree.ros_interface import create_ros_interface


def main(args=None):
    """程序入口：选择默认树配置，初始化 blackboard，并启动行为树运行器。"""
    # main.py 有意保持“短而显式”：
    # 这里不塞业务逻辑，只把最关键的默认选择摆在入口层，
    # 这样调试时一眼就能看到“当前默认跑哪棵树、用什么模式跑”。
    # 这里把“默认跑哪棵树”显式放在入口文件里，方便调试时直接切换。
    # 当前默认指向只靠近不抓取的往返测试树：
    # 只做底盘靠近与往返导航，头部盯箱和盯传送带。
    # tree_file_name = "tree/test/move_box_approach_shuttle_no_grasp_test_cn.json"

    # 测试完整搬箱流程头部跟踪版：抓箱前盯 YOLO 箱体，抓箱后盯 map 传送带点。
    # tree_file_name = "tree/test/move_box_full_direct_grasp_place_turn_head_track_cn.json"

    # 测试头部持续盯住 YOLO 最近箱体 + 按 Enter 双点导航时，切换到下面这棵树。
    # tree_file_name = "tree/test/head_track_yolo_box_parallel_navigation_test_cn.json"

    # 单独测试头部持续盯住 YOLO 最近箱体时，切换到下面这棵树。
    # tree_file_name = "tree/test/head_track_yolo_box_test_cn.json"

    # 单独测试头部盯住 map 系固定坐标点时，切换到下面这棵树。
    # tree_file_name = "tree/test/head_track_map_point_test_cn.json"

    # 单独测试头部在两个 map 固定坐标点之间按 Enter 循环切换时，切换到下面这棵树。
    # tree_file_name = "tree/test/head_track_map_point_switch_test_cn.json"

    # 测试通用动作节点完整搬箱流程时，切换到下面这棵树。
    # tree_file_name = "tree/test/move_box_full_direct_grasp_place_common_cn.json"

    # 完整实机搬箱流程：抓箱对齐 -> 直接抓取 -> 放箱 -> 回等待区域。
    # tree_file_name = "tree/box/move_box_full_direct_grasp_place_memory.json"

    # 当前默认：独立测试左手外拉、右手补抓、双手抬箱流程。
    tree_file_name = "tree/test/move_box_left_pull_right_grasp_test_cn.json"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # blackboard.json 用来放启动时就应该可见的共享参数。
    blackboard_json_file = os.path.join(
        project_root, "config", "blackboard", "blackboard.json"
    )

    # 入口显式指定 ROS 运行时，避免在同时装有 ROS1/ROS2 的环境里走错分支。
    ros_runtime = "ros1"
    # ros_runtime = "ros2"
    node_name = f"pytrees_{ros_runtime}_runner"

    ros = create_ros_interface(node_name=node_name, ros_version=ros_runtime)
    ros.init(args=args)
    # 实机部分执行器异常时的测试保护开关：
    # True 表示行为树节点保留流程与可视化状态，但不下发对应真实动作。
    ros.set_parameters(
        {
            "skip_torso_motion": False,
            "skip_head_motion": False,
            "skip_arm_motion": False,
            "skip_claw_motion": False,
        }
    )
    ros.get_logger().info(
        f"当前启动的 ROS 运行时为: {ros_runtime.upper()}，节点名: {node_name}"
    )
    # 启动前先把共享参数灌进全局 blackboard，供后续叶子节点复用。
    load_blackboard_from_json(ros, blackboard_json_file)
    tree_json_file = os.path.join(project_root, "config", tree_file_name)
    # 如果你想切回其他包内配置文件，只需要改上面的 tree_file_name。
    # 如果你想直接使用绝对路径，也可以改成下面这种形式：
    # tree_json_file = "/home/ysd/Desktop/ros2/pytrees_ros2/src/pytrees_ros2/config/py_tree.json"

    default_config = BehaviorTreeRunnerConfig.with_defaults(
        # 这里优先推荐只改文件名，路径会自动拼到包内 config 目录。
        tree_json_file=tree_json_file,
        # tick 周期，单位毫秒。调大后更容易观察执行过程。
        tick_period_ms=20,
        # 自定义网页可视化开关。
        enable_web_viewer=True,
        # web_viewer_host="127.0.0.1",
        web_viewer_host="0.0.0.0",
        web_viewer_port=8765,
        # 根节点到 SUCCESS/FAILURE 后是否自动停止。
        stop_on_terminal_state=True,
        
        #实机测试
        # 手动结果模式：叶子节点到达完成时机会等待你的 s/f/r 输入。
        manual_result_mode=False,
        # 是否开启终端输入线程。
        enable_manual_result_input=False,

        # #仿真测试，配合外部sender使用（src/tree/tools/manual_result_sender.py）
        # # 手动结果模式：叶子节点到达完成时机会等待你的 s/f/r 输入。
        # manual_result_mode=True,
        # # 是否开启终端输入线程。
        # enable_manual_result_input=False,

        # 外部脚本给树喂结果时使用的 topic。
        manual_result_topic=f"/{node_name}/manual_result",
        # 当前等待输入的节点列表会发布到这个 topic，便于测试终端展示。
        waiting_nodes_topic=f"/{node_name}/waiting_nodes",
        # 是否接入官方 py-trees-tree-viewer。
        enable_py_trees_ros_viewer=True,
        # 官方 viewer 的快照周期；inf 表示主要在状态变化时发布。
        py_trees_ros_viewer_snapshot_period=float("inf"),
    )

    # 入口只负责两件事：准备默认配置，以及启动/关闭 ROS 节点生命周期。
    node = BehaviorTreeRunner(ros=ros, default_config=default_config)
    try:
        ros.spin()
    finally:
        node.destroy_node()
        ros.shutdown()


if __name__ == "__main__":
    # 支持开发阶段直接使用 `python3 path/to/main.py` 启动。
    main()
