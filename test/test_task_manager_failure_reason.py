"""HTTP TaskManager 失败原因透传测试。"""

from types import SimpleNamespace

from tree.runtime.http_service.task_manager import TaskManager


class _Adapter:
    """测试用任务 adapter。"""

    pass


class _Runner:
    """测试用 runner，只提供快照。"""

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def get_snapshot(self):
        return self.snapshot


def test_task_manager_exposes_deepest_failed_node_feedback_message():
    """任务失败时应优先透传最深失败节点的 feedback_message。"""
    adapter = _Adapter()
    snapshot = {
        "generated_at": 100.0,
        "root_status": "FAILURE",
        "execution_state": "STOPPED",
        "tick_count": 5,
        "nodes": {
            "root": {
                "label": "HTTP navigation 单次任务流程",
                "name": "Sequence",
                "type": "Sequence",
                "status": "FAILURE",
                "depth": 0,
                "feedback_message": "",
            },
            "root/nav": {
                "label": "导航到 HTTP 目标点",
                "name": "HttpNavigateToPose",
                "type": "HttpNavigateToPose",
                "status": "FAILURE",
                "depth": 1,
                "feedback_message": "navigation task failed: {'task_status': 6}",
            },
        },
    }
    manager = TaskManager(runner=_Runner(snapshot), registry=None)
    manager._active_task_id = "task-1"
    manager._active_definition = SimpleNamespace(
        name="navigation",
        root_label="HTTP navigation 单次任务流程",
        adapter=adapter,
    )
    manager._task_status = "RUNNING"
    manager._created_at = 1.0

    response = manager.get_task_status("task-1")

    assert response["status"] == "FAILED"
    assert response["message"] == "navigation task failed: {'task_status': 6}"
    assert response["failedNode"]["name"] == "HttpNavigateToPose"
    assert response["failedNode"]["message"] == "navigation task failed: {'task_status': 6}"


def test_task_manager_marks_success_from_root_terminal_status():
    """单次树根节点 SUCCESS 后，TaskManager 应直接标记任务成功。"""
    snapshot = {
        "generated_at": 100.0,
        "root_status": "SUCCESS",
        "execution_state": "STOPPED",
        "tick_count": 3,
        "nodes": {},
    }
    manager = TaskManager(runner=_Runner(snapshot), registry=None)
    manager._active_task_id = "task-2"
    manager._active_definition = SimpleNamespace(
        name="navigation",
        root_label="HTTP navigation 单次任务流程",
        adapter=_Adapter(),
    )
    manager._task_status = "RUNNING"
    manager._created_at = 1.0

    response = manager.get_task_status("task-2")

    assert response["status"] == "SUCCEEDED"
    assert response["message"] == "navigation task completed"
    assert response["finishedAt"] is not None


def test_task_manager_limits_task_records_and_keeps_active_task():
    """任务记录超过上限时应淘汰最老历史记录，并保留当前任务记录。"""
    manager = TaskManager(runner=_Runner({}), registry=None, max_task_records=2)
    manager._active_task_id = "task-active"

    manager._store_task_record_locked("task-old", {"taskId": "task-old"})
    manager._store_task_record_locked("task-active", {"taskId": "task-active"})
    manager._store_task_record_locked("task-new", {"taskId": "task-new"})

    assert "task-old" not in manager._task_records
    assert list(manager._task_records.keys()) == ["task-active", "task-new"]
