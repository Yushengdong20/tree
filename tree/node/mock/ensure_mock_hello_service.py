"""确保 blackboard 中存在一个可复用的 MockHelloService 实例。"""

import py_trees
from py_trees.common import Status

from tree.runtime.mock.mock_hello_service import MockHelloService

from ..base import TimedMockAction


class EnsureMockHelloService(TimedMockAction):
    """第一次执行时创建 mock 服务，后续执行时直接复用。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # blackboard key 可配置，默认使用一个扁平 key，避免层级 key 在不同 py_trees 版本下的权限差异。
        self.service_key = str(
            params.get("service_key", "runtime_mock_hello_service")
        ).strip()
        # 这个节点既要“检查是否已有实例”，也要“在没有时写入实例”，所以需要读写权限。
        self.blackboard.register_key(
            key=self.service_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.service_key,
            access=py_trees.common.Access.WRITE,
        )

    def update(self):
        """检查 blackboard 中是否已有实例，没有则创建，有则记录复用。"""
        # 先尝试从 blackboard 读取共享实例；有则复用，无则创建。
        service = self.blackboard.get(self.service_key) if self.blackboard.exists(self.service_key) else None
        if service is None:
            service = MockHelloService()
            # overwrite=True 让这个示例节点在重复执行时仍然能安全更新同一个 key。
            self.blackboard.set(self.service_key, service, overwrite=True)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] created MockHelloService: "
                f"instance_id={id(service)}, creation_index={service.creation_index}, "
                f"class_created_count={MockHelloService.created_count}"
            )
        else:
            # 命中已存在实例时只记录复用日志，不再重新创建。
            self.ros_node.get_logger().info(
                f"[{self.config_label}] reused MockHelloService: "
                f"instance_id={id(service)}, creation_index={service.creation_index}, "
                f"class_created_count={MockHelloService.created_count}"
            )
        return Status.SUCCESS

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return f"[{self.config_label}] EnsureMockHelloService start: key={self.service_key}"
