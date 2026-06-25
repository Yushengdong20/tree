"""官方 py_trees_ros viewer 适配层。

这个文件只负责一件事：
在不改动现有业务树 root 的前提下，为它挂上 py_trees_ros 的
snapshot/services 能力，让官方 `py-trees-tree-viewer` 可以连接。
"""


def enable_py_trees_ros_viewer_support(ros_node, tree, snapshot_period: float):
    """Wrap the core tree with py_trees_ros services so the official viewer can connect."""
    if not ros_node.ros.supports_py_trees_ros_viewer():
        ros_node.get_logger().warning(
            "py_trees_ros viewer support disabled because current ROS runtime is not ROS2."
        )
        return tree, False

    # 官方 viewer 依赖 py_trees_ros 的 snapshots/services 体系。
    # 这里的做法不是重建一棵树，而是在现有 root 外面包一层 py_trees_ros.BehaviourTree，
    # 这样原来的树结构和业务节点都能保留下来。
    try:
        from py_trees_ros import trees as py_trees_ros_trees
    except Exception as exc:
        ros_node.get_logger().warning(
            "py_trees_ros viewer support disabled because py_trees_ros could not be imported: "
            f"{exc}. Install missing dependencies such as python3-pydot if needed."
        )
        return tree, False

    try:
        # 复用原来的 root，只额外挂上 py_trees_ros 的 services/topics 能力。
        ros_tree = py_trees_ros_trees.BehaviourTree(root=tree.root)
        ros_tree.setup(node=ros_node.ros.get_native_node())
        # 默认 stream 关闭，由 viewer 端按需通过服务打开，避免平时额外刷流量。
        ros_node.set_parameters(
            {
                "default_snapshot_stream": False,
                "default_snapshot_period": snapshot_period,
                "default_snapshot_blackboard_data": True,
                "default_snapshot_blackboard_activity": False,
            }
        )
        return ros_tree, True
    except Exception as exc:
        ros_node.get_logger().warning(
            f"Failed to initialise py_trees_ros viewer support: {exc}"
        )
        return tree, False
