"""从 blackboard 读取共享 MockHelloService，并调用它输出日志。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class UseMockHelloService(TimedMockAction):
    """使用其他节点预先创建好的共享 mock 服务实例。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 这个节点不负责创建实例，只负责读取并使用，所以只申请 READ 权限。
        self.service_key = str(
            params.get("service_key", "runtime_mock_hello_service")
        ).strip()
        self.blackboard.register_key(
            key=self.service_key,
            access=py_trees.common.Access.READ,
        )

    def update(self):
        """读取共享实例并调用 say_hello，验证实例是否被持续复用。"""
        # 从 blackboard 读取共享实例；如果为空，说明初始化节点没有先执行成功。
        service = self.blackboard.get(self.service_key) if self.blackboard.exists(self.service_key) else None
        if service is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] shared service missing on blackboard: key={self.service_key}"
            )
            return Status.FAILURE

        # 调用共享实例的方法；如果复用成功，日志里的 instance_id 会保持不变，而 call_count 会递增。
        message = service.say_hello()
        self.ros_node.get_logger().info(
            f"[{self.config_label}] {message}"
        )
        return Status.SUCCESS

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return f"[{self.config_label}] UseMockHelloService start: key={self.service_key}"
