"""MercuryTree HTTP server 模式入口。"""

import argparse
import os

from tree.core.blackboard_bootstrap import load_blackboard_from_json
from tree.core.runner import BehaviorTreeRunner
from tree.core.runner_config import BehaviorTreeRunnerConfig
from tree.constants import MODEL_TYPE_KEY, ROBOT_SERVICES_KEY
from tree.ros_interface import create_ros_interface
from tree.runtime.http_service.task_adapters.grasp_and_place import (
    GraspAndPlaceTaskAdapter,
)
from tree.runtime.http_service.task_adapters.direct_grasp_place_memory import (
    DirectGraspPlaceMemoryTaskAdapter,
)
from tree.runtime.http_service.task_adapters.move_box import MoveBoxTaskAdapter
from tree.runtime.http_service.task_adapters.navigation import NavigationTaskAdapter
from tree.runtime.http_service.task_http_server import TaskHttpServer
from tree.runtime.http_service.task_manager import TaskManager
from tree.runtime.http_service.task_registry import TaskDefinition, TaskRegistry


def parse_server_args(args=None):
    """
    解析 server 模式自身使用的命令行参数，并保留其余参数交给 ROS。
    :param args: 命令行参数列表；None 表示读取 sys.argv。
    :return: (入口参数, ROS 参数)。
    """
    parser = argparse.ArgumentParser(
        description="启动 MercuryTree HTTP 任务服务模式",
    )
    parser.add_argument(
        "--initial-tree",
        default="",
        help="可选：server 启动后预加载的行为树 JSON；默认不加载树、不启动 tick。",
    )
    parser.add_argument(
        "--task-host",
        default="127.0.0.1",
        help="任务 HTTP 服务监听地址。",
    )
    parser.add_argument(
        "--task-port",
        type=int,
        default=8766,
        help="任务 HTTP 服务监听端口。",
    )
    parser.add_argument(
        "--preload-services",
        choices=("none", "grasp_object", "move_box"),
        default="none",
        help="server 启动时预初始化的机器人共享服务，默认不预加载。",
    )
    return parser.parse_known_args(args)


def resolve_config_file(project_root, tree_argument):
    """
    将配置文件参数解析为绝对路径。
    :param project_root: MercuryTree 项目根目录。
    :param tree_argument: config 下相对路径或绝对路径。
    :return: 配置文件绝对路径。
    """
    tree_argument = os.path.expanduser(str(tree_argument).strip())
    if os.path.isabs(tree_argument):
        config_file = tree_argument
    else:
        config_file = os.path.join(project_root, "config", tree_argument)

    config_file = os.path.abspath(config_file)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    return config_file


def build_task_registry(project_root):
    """
    构建 HTTP endpoint 到 service tree 的映射。
    :param project_root: MercuryTree 项目根目录。
    :return: TaskRegistry 实例。
    """
    registry = TaskRegistry()
    move_box_tree = resolve_config_file(
        project_root,
        "tree/service/move_box/start_move_box.json",
    )
    direct_grasp_place_memory_tree = resolve_config_file(
        project_root,
        "tree/service/move_box/start_move_box_direct_grasp_place_memory.json",
    )
    grasp_and_place_tree = resolve_config_file(
        project_root,
        "tree/service/grasp_object/start_grasp_and_place.json",
    )
    navigation_tree = resolve_config_file(
        project_root,
        "tree/service/navigation/start_navigation.json",
    )
    # 关键步骤：客户端通过 URL 区分任务，内部 registry 再映射到任务名、tree 和 adapter。
    registry.register(
        TaskDefinition(
            name="move_box",
            endpoint="/api/start_move_box",
            tree_file=move_box_tree,
            adapter=MoveBoxTaskAdapter(),
            root_label="HTTP move_box 单次任务流程",
        )
    )
    registry.register(
        TaskDefinition(
            name="move_box_direct_grasp_place_memory",
            endpoint="/api/start_move_box_direct_grasp_place_memory",
            tree_file=direct_grasp_place_memory_tree,
            adapter=DirectGraspPlaceMemoryTaskAdapter(),
            root_label="HTTP YOLO记忆直接抓箱放箱单次任务流程",
        )
    )
    registry.register(
        TaskDefinition(
            name="grasp_and_place",
            endpoint="/api/start_grasp_and_place",
            tree_file=grasp_and_place_tree,
            adapter=GraspAndPlaceTaskAdapter(),
            root_label="HTTP grasp_and_place 单次任务流程",
        )
    )
    registry.register(
        TaskDefinition(
            name="navigation",
            endpoint="/api/start_navigation",
            tree_file=navigation_tree,
            adapter=NavigationTaskAdapter(),
            root_label="HTTP navigation 单次任务流程",
        )
    )
    return registry


def preload_robot_services(ros, service_name):
    """
    server 启动阶段预初始化机器人 SDK services。
    :param ros: ROS 接口实例，用于打印日志。
    :param service_name: 预加载服务类型，可选 none/grasp_object/move_box。
    """
    service_name = str(service_name or "none").strip().lower()
    if service_name == "none":
        ros.get_logger().info("Robot services preload skipped")
        return

    ros.get_logger().info(f"Preloading robot services: {service_name}")
    if service_name == "grasp_object":
        from tree.runtime.grasp.grasp_real import build_grasp_object_services

        # 关键步骤：grasp_and_place 的 ArmController 目标系固定为 waist_yaw_link。
        services = build_grasp_object_services()
    elif service_name == "move_box":
        from tree.runtime.move_box.move_box_real import build_robot_services

        # 关键步骤：move_box 的 ArmController 目标系固定为 base_link。
        services = build_robot_services()
    else:
        raise ValueError(f"不支持的 preload service: {service_name}")

    import py_trees

    py_trees.blackboard.Blackboard.set(ROBOT_SERVICES_KEY, services)
    if hasattr(services, "model_type"):
        py_trees.blackboard.Blackboard.set(MODEL_TYPE_KEY, services.model_type)
    ros.get_logger().info(
        f"Preloaded robot services: type={service_name}, services_id={id(services)}"
    )


def main(args=None):
    """server 模式入口：启动 runner、任务注册表和独立 HTTP 任务服务。"""
    server_args, ros_args = parse_server_args(args)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    initial_tree_json_file = ""
    if str(server_args.initial_tree).strip():
        initial_tree_json_file = resolve_config_file(
            project_root,
            server_args.initial_tree,
        )
    blackboard_json_file = os.path.join(
        project_root,
        "config",
        "blackboard",
        "blackboard.json",
    )

    ros_runtime = "ros1"
    node_name = f"pytrees_{ros_runtime}_server"

    ros = create_ros_interface(node_name=node_name, ros_version=ros_runtime)
    ros.init(args=ros_args)
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
    if initial_tree_json_file:
        ros.get_logger().info(f"server 模式初始行为树配置为: {initial_tree_json_file}")
    else:
        ros.get_logger().info("server 模式启动待命：未加载初始行为树")
    load_blackboard_from_json(ros, blackboard_json_file)
    preload_robot_services(ros, server_args.preload_services)

    default_config = BehaviorTreeRunnerConfig.with_defaults(
        tree_json_file=initial_tree_json_file,
        tick_period_ms=20,
        enable_web_viewer=True,
        web_viewer_host="0.0.0.0",
        web_viewer_port=8765,
        # server 模式按 HTTP 请求 reload 单次树；单次树终态后停表，下一次请求再恢复。
        stop_on_terminal_state=True,
        manual_result_mode=False,
        enable_manual_result_input=False,
        manual_result_topic=f"/{node_name}/manual_result",
        waiting_nodes_topic=f"/{node_name}/waiting_nodes",
        # server 模式会运行时 reload tree，首版先关闭官方 viewer 包装，避免重复注册服务。
        enable_py_trees_ros_viewer=False,
        py_trees_ros_viewer_snapshot_period=float("inf"),
    )

    runner = BehaviorTreeRunner(ros=ros, default_config=default_config)
    registry = build_task_registry(project_root)
    task_manager = TaskManager(runner=runner, registry=registry)
    task_server = TaskHttpServer(
        task_manager=task_manager,
        registry=registry,
        host=server_args.task_host,
        port=server_args.task_port,
    )
    task_server.start()
    ros.get_logger().info(
        f"Task API: http://{server_args.task_host}:{server_args.task_port}"
    )
    ros.get_logger().info(
        f"Task API docs: http://{server_args.task_host}:{server_args.task_port}/docs"
    )
    ros.get_logger().info(f"Task endpoints: {', '.join(registry.endpoints())}")

    try:
        ros.spin()
    finally:
        task_server.stop()
        runner.destroy_node()
        ros.shutdown()


if __name__ == "__main__":
    main()
