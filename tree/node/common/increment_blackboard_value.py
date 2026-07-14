# -*- coding: utf-8 -*-
"""IncrementBlackboardValue 节点。

用于流程成功后推进计数，例如码垛成功后把 stack_count + 1。
"""

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status


class IncrementBlackboardValue(Behaviour):
    """读取 blackboard 数值，自增后写回。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name)
        self.config_label = config_label
        self.ros_node = ros_node
        self.key = str(params.get("key", "")).strip()
        self.step = int(params.get("step", 1))
        self.default_value = int(params.get("default_value", 0))
        self.blackboard = py_trees.blackboard.Client(name=config_label)
        if self.key:
            self.blackboard.register_key(key=self.key, access=py_trees.common.Access.READ)
            self.blackboard.register_key(key=self.key, access=py_trees.common.Access.WRITE)

    def update(self):
        if not self.key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 params.key")
            return Status.FAILURE

        try:
            current = self.blackboard.get(self.key) if self.blackboard.exists(self.key) else self.default_value
            next_value = int(current) + self.step
            self.blackboard.set(self.key, next_value, overwrite=True)
        except Exception as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 自增失败: key={self.key}, error={exc}"
            )
            return Status.FAILURE

        self.ros_node.get_logger().info(
            f"[{self.config_label}] increment blackboard {self.key}: {current!r} -> {next_value!r}"
        )
        return Status.SUCCESS
