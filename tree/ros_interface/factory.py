"""ROS 运行时选择工厂。"""


def create_ros_interface(node_name: str, ros_version: str = "auto"):
    """根据显式配置或当前 Python 环境创建 ROS 适配器。"""
    version = str(ros_version).strip().lower()

    if version == "ros2":
        from tree.ros_interface.ros2 import Ros2Interface

        return Ros2Interface(node_name=node_name)

    if version == "ros1":
        from tree.ros_interface.ros1 import Ros1Interface

        return Ros1Interface(node_name=node_name)

    if version != "auto":
        raise ValueError(f"Unsupported ros_version: {ros_version}")

    try:
        from tree.ros_interface.ros2 import Ros2Interface

        return Ros2Interface(node_name=node_name)
    except ImportError as ros2_error:
        try:
            from tree.ros_interface.ros1 import Ros1Interface

            return Ros1Interface(node_name=node_name)
        except ImportError as ros1_error:
            raise RuntimeError(
                "Neither ROS2 rclpy nor ROS1 rospy could be imported. "
                f"ROS2 error: {ros2_error}; ROS1 error: {ros1_error}"
            ) from ros1_error
