"""BehaviorTreeRunner server 待命启动测试。"""

from tree.core.runner import BehaviorTreeRunner
from tree.core.runner_config import BehaviorTreeRunnerConfig


class _Logger:
    """测试用日志对象。"""

    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _Timer:
    """测试用定时器。"""

    def __init__(self):
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def is_canceled(self):
        return self._canceled


class _Ros:
    """测试用最小 ROS 接口。"""

    def __init__(self):
        self.logger = _Logger()
        self.created_timers = []

    def get_param(self, _name, default_value):
        return default_value

    def get_logger(self):
        return self.logger

    def get_namespace(self):
        return "/"

    def get_name(self):
        return "test_runner"

    def create_timer(self, _period_sec, _callback):
        timer = _Timer()
        self.created_timers.append(timer)
        return timer


def _make_config(tree_json_file=""):
    """构造测试用 runner 配置。"""
    return BehaviorTreeRunnerConfig.with_defaults(
        tree_json_file=tree_json_file,
        tick_period_ms=20,
        enable_web_viewer=False,
        stop_on_terminal_state=True,
        manual_result_mode=False,
        enable_manual_result_input=True,
        enable_py_trees_ros_viewer=False,
    )


def test_runner_without_initial_tree_stays_idle_until_reload(tmp_path):
    """server 模式无初始树启动时不应创建 tick timer，任务 reload 后才启动。"""
    ros = _Ros()
    runner = BehaviorTreeRunner(ros=ros, default_config=_make_config())

    snapshot = runner.get_snapshot()
    assert runner.tree is None
    assert runner.timer is None
    assert ros.created_timers == []
    assert snapshot["execution_state"] == "IDLE"
    assert snapshot["root_status"] == "INVALID"
    assert snapshot["node_count"] == 0

    tree_file = tmp_path / "single_task.json"
    tree_file.write_text(
        (
            '{"tree": {"name": "IdleRunning", "label": "task", '
            '"params": {}, "childs": []}}'
        ),
        encoding="utf-8",
    )

    reloaded, message = runner.reload_tree(str(tree_file))

    assert reloaded is True
    assert message == "tree reloaded"
    assert runner.tree is not None
    assert runner.timer is ros.created_timers[0]
    assert runner.get_snapshot()["execution_state"] == "READY"
