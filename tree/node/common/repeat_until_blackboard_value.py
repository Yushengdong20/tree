"""按 blackboard 值重复执行子树，直到满足完成条件。"""

import py_trees
from py_trees.common import Status

from tree.utils.params import parse_param_value


class RepeatUntilBlackboardValue(py_trees.decorators.Decorator):
    """重复执行子节点，直到 ``key == expected``。

    与 py_trees 自带 ``Repeat`` 不同，本装饰器在每一轮子树成功后都会检查
    blackboard，因此视觉校正已经收敛时可以立即退出，而不用跑满固定次数。
    ``max_iterations`` 是安全上限；超限返回 FAILURE，保持上层流程不释放箱子。
    """

    def __init__(self, name, child, config_label, ros_node, params):
        super().__init__(name=name, child=child)
        self.config_label = config_label
        self.ros_node = ros_node
        self.key = str(params.get("key", "")).strip()
        self.expected = parse_param_value(params.get("expected", True))
        self.max_iterations = max(int(params.get("max_iterations", 8)), 1)
        self.iterations = 0
        self.blackboard = py_trees.blackboard.Client(name=config_label)
        if not self.key:
            raise ValueError("RepeatUntilBlackboardValue 缺少 params.key")
        self.blackboard.register_key(key=self.key, access=py_trees.common.Access.READ)

    def initialise(self):
        self.iterations = 0

    def tick(self):
        if self.status != Status.RUNNING:
            self.initialise()

        if self._matched():
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
            self.iterations += 1
            if self._matched():
                self.status = Status.SUCCESS
                yield self
                return
            if self.iterations >= self.max_iterations:
                self.feedback_message = (
                    f"视觉校正未在 {self.max_iterations} 轮内收敛: "
                    f"key={self.key}, expected={self.expected!r}"
                )
                self.ros_node.get_logger().error(f"[{self.config_label}] {self.feedback_message}")
                self.status = Status.FAILURE
                yield self
                return
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 第 {self.iterations}/{self.max_iterations} 轮未收敛，重新获取 FP 并计算下一小步"
            )
            self.decorated.stop(Status.INVALID)

        self.status = Status.RUNNING
        yield self

    def update(self):
        return self.status

    def _matched(self):
        if not self.blackboard.exists(self.key):
            return False
        return self.blackboard.get(self.key) == self.expected

    def _stop_child_if_active(self):
        if self.decorated.status != Status.INVALID:
            self.decorated.stop(Status.INVALID)
