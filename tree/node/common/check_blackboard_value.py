"""检查 blackboard 值是否与期望值一致。"""

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from tree.utils.params import parse_param_value


class CheckBlackboardValue(Behaviour):
    """值相等时返回 SUCCESS，供 Selector 选择业务分支。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name)
        self.config_label = config_label
        self.ros_node = ros_node
        self.key = str(params.get("key", "")).strip()
        self.expected = parse_param_value(params.get("expected"))
        self.blackboard = py_trees.blackboard.Client(name=config_label)
        if self.key:
            self.blackboard.register_key(key=self.key, access=py_trees.common.Access.READ)

    def update(self):
        if not self.key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 params.key")
            return Status.FAILURE
        if not self.blackboard.exists(self.key):
            return Status.FAILURE
        actual = self.blackboard.get(self.key)
        if actual != self.expected:
            return Status.FAILURE
        self.ros_node.get_logger().info(
            f"[{self.config_label}] blackboard条件满足: "
            f"{self.key}={actual!r}"
        )
        return Status.SUCCESS
