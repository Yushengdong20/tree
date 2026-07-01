"""项目入口文件。

这个文件有意保持“短而显式”：
- 不承载具体业务逻辑
- 只负责挑选默认配置
- 再把控制权交给 BehaviorTreeRunner

因此当你想快速确认“当前默认跑哪棵树、是否走 mock、是否启用 viewer”时，
先看这个文件就够了。
"""

import argparse
import os

from tree.core.blackboard_bootstrap import load_blackboard_from_json
from tree.core.runner import BehaviorTreeRunner
from tree.core.runner_config import BehaviorTreeRunnerConfig
from tree.ros_interface import create_ros_interface


# 默认行为树候选清单：测试时取消目标行注释，并注释当前启用行即可。
# 只靠近不抓取的往返测试树。
# DEFAULT_TREE_FILE_NAME = "tree/test/move_box_approach_shuttle_no_grasp_test_cn.json"
# 完整搬箱流程头部跟踪版。
# DEFAULT_TREE_FILE_NAME = "tree/test/move_box_full_direct_grasp_place_turn_head_track_cn.json"
# 头部跟踪 YOLO 最近箱体并执行双点导航。
# DEFAULT_TREE_FILE_NAME = "tree/test/head_track_yolo_box_parallel_navigation_test_cn.json"
# 单独测试头部跟踪 YOLO 最近箱体。
# DEFAULT_TREE_FILE_NAME = "tree/test/head_track_yolo_box_test_cn.json"
# 单独测试头部跟踪 map 固定坐标点。
# DEFAULT_TREE_FILE_NAME = "tree/test/head_track_map_point_test_cn.json"
# 头部在两个 map 固定坐标点之间循环切换。
# DEFAULT_TREE_FILE_NAME = "tree/test/head_track_map_point_switch_test_cn.json"
# 通用动作节点完整搬箱流程。
# DEFAULT_TREE_FILE_NAME = "tree/test/move_box_full_direct_grasp_place_common_cn.json"
# 完整实机搬箱流程：抓箱对齐、直接抓取、放箱、返回等待区域。
# DEFAULT_TREE_FILE_NAME = "tree/box/move_box_full_direct_grasp_place_memory.json"
# 独立测试左手外拉、右手补抓、双手抬箱。
DEFAULT_TREE_FILE_NAME = "tree/test/move_box_left_pull_right_grasp_test_cn.json"
# 独立测试腰部配合下的双爪同步抓箱。
# DEFAULT_TREE_FILE_NAME = "tree/test/move_box_dual_claw_grasp_test_cn.json"
# 固定点位前方两箱的 FoundationPose 左拉拆垛测试。
# DEFAULT_TREE_FILE_NAME = "tree/test/move_box_full_left_pull_right_grasp_stack_cn.json"


def parse_main_args(args=None):
    """
    解析入口自身使用的命令行参数，并保留其余参数交给 ROS。
    :param args: 命令行参数列表；None 表示读取 sys.argv。
    :return: (入口参数, ROS 参数)。
    """
    parser = argparse.ArgumentParser(
        description="启动 MercuryTree 行为树运行器",
    )
    parser.add_argument(
        "--tree",
        default=DEFAULT_TREE_FILE_NAME,
        help=(
            "要运行的行为树 JSON。相对路径按 config 目录解析，"
            "也支持直接传入绝对路径。"
        ),
    )
    return parser.parse_known_args(args)


def resolve_tree_json_file(project_root, tree_argument):
    """
    将命令行树参数解析为绝对路径。
    :param project_root: MercuryTree 项目根目录。
    :param tree_argument: --tree 传入的相对或绝对路径。
    :return: 行为树 JSON 的绝对路径。
    """
    tree_argument = os.path.expanduser(str(tree_argument).strip())
    if not tree_argument:
        tree_argument = DEFAULT_TREE_FILE_NAME

    if os.path.isabs(tree_argument):
        tree_json_file = tree_argument
    else:
        tree_json_file = os.path.join(project_root, "config", tree_argument)

    tree_json_file = os.path.abspath(tree_json_file)
    if not os.path.isfile(tree_json_file):
        raise FileNotFoundError(f"行为树配置文件不存在: {tree_json_file}")
    return tree_json_file


def main(args=None):
    """程序入口：选择默认树配置，初始化 blackboard，并启动行为树运行器。"""
    main_args, ros_args = parse_main_args(args)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree_json_file = resolve_tree_json_file(project_root, main_args.tree)
    # blackboard.json 用来放启动时就应该可见的共享参数。
    blackboard_json_file = os.path.join(
        project_root, "config", "blackboard", "blackboard.json"
    )

    # 入口显式指定 ROS 运行时，避免在同时装有 ROS1/ROS2 的环境里走错分支。
    ros_runtime = "ros1"
    # ros_runtime = "ros2"
    node_name = f"pytrees_{ros_runtime}_runner"

    ros = create_ros_interface(node_name=node_name, ros_version=ros_runtime)
    # 关键步骤：main 自己消费 --tree，其余 ROS remap/运行时参数继续交给 ROS 初始化。
    ros.init(args=ros_args)
    # 实机部分执行器异常时的测试保护开关：
    # True 表示行为树节点保留流程与可视化状态，但不下发对应真实动作。
    ros.set_parameters(
        {
            "skip_torso_motion": False,
            "skip_head_motion": False,
            "skip_arm_motion": False,
            "skip_claw_motion": False,
            "arm_event_timeout": 10.0,
        }
    )
    ros.get_logger().info(
        f"当前启动的 ROS 运行时为: {ros_runtime.upper()}，节点名: {node_name}"
    )
    ros.get_logger().info(f"当前启动的行为树配置为: {tree_json_file}")
    # 启动前先把共享参数灌进全局 blackboard，供后续叶子节点复用。
    load_blackboard_from_json(ros, blackboard_json_file)

    default_config = BehaviorTreeRunnerConfig.with_defaults(
        # CLI 负责提供默认树；ROS 参数 tree_json_file 仍可在 runner 初始化时覆盖它。
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
