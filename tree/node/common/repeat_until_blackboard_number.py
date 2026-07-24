"""按 blackboard 数值条件重复执行子树。"""

import operator

import py_trees
from py_trees.common import Status


class RepeatUntilBlackboardNumber(py_trees.decorators.Decorator):
    """重复 tick 子节点，直到 blackboard 数值条件满足。"""

    OPERATORS = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        ">=": operator.ge,
        ">": operator.gt,
    }

    def __init__(self, name, child, config_label, ros_node, params):
        """
        初始化循环装饰器。
        :param name: py_trees 节点名称。
        :param child: 每轮需要执行的子树。
        :param config_label: JSON 中配置的展示名称。
        :param ros_node: BehaviorTreeRunner/ROS 节点适配对象。
        :param params: JSON 展平后的参数对象。
        """
        super().__init__(name=name, child=child)
        self.config_label = config_label
        self.ros_node = ros_node
        self.left_key = str(params.get("left_key", params.get("key", ""))).strip()
        self.right_key = str(params.get("right_key", "")).strip()
        self.operator_text = str(params.get("operator", ">=")).strip()
        self.right_value = params.get("right_value")
        self.blackboard = py_trees.blackboard.Client(name=config_label)

        if not self.left_key:
            raise ValueError("RepeatUntilBlackboardNumber 缺少 params.left_key")
        if not self.right_key and self.right_value is None:
            raise ValueError(
                "RepeatUntilBlackboardNumber 缺少 params.right_key 或 params.right_value"
            )
        if self.operator_text not in self.OPERATORS:
            raise ValueError(
                "RepeatUntilBlackboardNumber operator 仅支持 <, <=, ==, >=, >"
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

    def tick(self):
        """先检查完成条件；未达标时推进一轮子树。"""
        if self.status != Status.RUNNING:
            self.initialise()

        # 关键步骤：条件已满足时直接让根进入 SUCCESS，TaskManager 据此完成任务。
        if self._condition_matched():
            self._stop_child_if_active()
            self.status = Status.SUCCESS
            yield self
            return

        for node in self.decorated.tick():
            yield node

        if self.decorated.status == Status.FAILURE:
            self.status = Status.FAILURE
            yield self
            return

        if self.decorated.status == Status.SUCCESS:
            if self._condition_matched():
                self.status = Status.SUCCESS
                yield self
                return
            # 关键步骤：一轮成功但数量未达标时重置子树，下个 tick 从新一轮开始。
            self.decorated.stop(Status.INVALID)

        self.status = Status.RUNNING
        yield self

    def update(self):
        """Decorators API 占位；实际状态由 tick() 控制。"""
        return self.status

    def _condition_matched(self):
        """读取并比较 blackboard 数值。"""
        try:
            left_value = self._read_number(self.left_key)
            right_value = self._resolve_right_number()
            return self.OPERATORS[self.operator_text](left_value, right_value)
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 循环完成条件读取失败: {exc}"
            )
            return False

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

    def _stop_child_if_active(self):
        """达标时停止仍处于运行态的子树。"""
        if self.decorated.status != Status.INVALID:
            self.decorated.stop(Status.INVALID)

    @staticmethod
    def _to_number(value, label):
        """把输入值转换为 float，bool 不按数字处理。"""
        if isinstance(value, bool):
            raise RuntimeError(f"{label} 必须是数字，当前为 bool")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{label} 必须是数字: value={value!r}") from exc
