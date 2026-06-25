"""一个始终返回 RUNNING 的空转节点。"""

from py_trees.common import Status

from ..base import TimedMockAction


class IdleRunning(TimedMockAction):
    """当没有任何业务请求时保持分支存活。"""

    def update(self):
        if self.should_use_mock_execution():
            return Status.RUNNING
        return Status.RUNNING
