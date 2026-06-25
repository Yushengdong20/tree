# -*- coding: utf-8 -*-
"""SetBlackboardValue 节点。"""

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from tree.utils.params import parse_param_value


class SetBlackboardValue(Behaviour):
    """把一个值写入 blackboard，可直接写常量，也可从另一个 key 复制。"""

    def __init__(self, name, config_label, ros_node, params):
        super(SetBlackboardValue, self).__init__(name=name)
        self.config_label = config_label
        self.ros_node = ros_node
        self.params = params
        self.target_key = str(self.params.get("key", ""))
        self.source_key = str(self.params.get("source_blackboard_key", "")).strip()
        self.raw_value = self.params.get("value")
        self.blackboard = py_trees.blackboard.Client(name=config_label)

        if self.target_key:
            self.blackboard.register_key(
                key=self.target_key,
                access=py_trees.common.Access.WRITE,
            )
        if self.source_key:
            self.blackboard.register_key(
                key=self.source_key,
                access=py_trees.common.Access.READ,
            )

    def update(self):
        if not self.target_key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 params.key")
            return Status.FAILURE

        try:
            if self.source_key:
                if not self.blackboard.exists(self.source_key):
                    self.ros_node.get_logger().error(
                        f"[{self.config_label}] blackboard key 不存在: {self.source_key}"
                    )
                    return Status.FAILURE
                value = self.blackboard.get(self.source_key)
            else:
                value = parse_param_value(self.raw_value)
        except Exception as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 写入 blackboard 失败: {exc}"
            )
            return Status.FAILURE

        self.blackboard.set(self.target_key, value, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] set blackboard {self.target_key}={value!r}"
        )
        return Status.SUCCESS
