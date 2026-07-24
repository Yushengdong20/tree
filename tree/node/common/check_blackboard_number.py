"""检查 blackboard 数值关系是否满足条件。"""

import operator

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from tree.utils.params import parse_param_value


class CheckBlackboardNumber(Behaviour):
    """按数值比较 blackboard 中的两个值。"""

    OPERATORS = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        ">=": operator.ge,
        ">": operator.gt,
    }

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name)
        self.config_label = config_label
        self.ros_node = ros_node
        self.left_key = str(params.get("left_key", params.get("key", ""))).strip()
        self.right_key = str(params.get("right_key", "")).strip()
        self.operator_text = str(params.get("operator", ">=")).strip()
        self.right_value = parse_param_value(params.get("right_value"))
        self.blackboard = py_trees.blackboard.Client(name=config_label)

        if not self.left_key:
            raise ValueError("CheckBlackboardNumber 缺少 params.left_key")
        if not self.right_key and self.right_value is None:
            raise ValueError(
                "CheckBlackboardNumber 缺少 params.right_key 或 params.right_value"
            )
        if self.operator_text not in self.OPERATORS:
            raise ValueError(
                "CheckBlackboardNumber operator 仅支持 <, <=, ==, >=, >"
            )

        self.blackboard.register_key(
            key=self.left_key,
            access=py_trees.common.Access.READ,
        )
        if self.right_key:
            self.blackboard.register_key(
                key=self.right_key,
                access=py_trees.common.Access.READ,
            )

    def update(self):
        try:
            left_value = self._read_number(self.left_key)
            right_value = self._resolve_right_number()
        except Exception as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 数值条件读取失败: {exc}"
            )
            return Status.FAILURE

        matched = self.OPERATORS[self.operator_text](left_value, right_value)
        if not matched:
            return Status.FAILURE

        self.ros_node.get_logger().info(
            f"[{self.config_label}] blackboard数值条件满足: "
            f"{self.left_key}={left_value:g} {self.operator_text} {right_value:g}"
        )
        return Status.SUCCESS

    def _resolve_right_number(self):
        """解析右侧比较值，优先读取 blackboard key。"""
        if self.right_key:
            return self._read_number(self.right_key)
        return self._to_number(self.right_value, "right_value")

    def _read_number(self, key):
        """从 blackboard 读取并转换为 float。"""
        if not self.blackboard.exists(key):
            raise RuntimeError(f"blackboard key 不存在: {key}")
        return self._to_number(self.blackboard.get(key), key)

    @staticmethod
    def _to_number(value, label):
        """把输入值转换为 float，bool 不按数字处理。"""
        if isinstance(value, bool):
            raise RuntimeError(f"{label} 必须是数字，当前为 bool")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} 必须是数字: value={value!r}") from exc
