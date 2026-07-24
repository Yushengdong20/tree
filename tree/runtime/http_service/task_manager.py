"""通用 HTTP service tree 任务管理器。"""

from collections import OrderedDict
import threading
import time
import uuid

from tree.runtime.http_service.models import StartTaskResponseModel, TaskStatusResponseModel


def _model_to_dict(model):
    """
    转换 Pydantic model 为普通 dict。
    :param model: Pydantic model 实例。
    :return: 可 JSON 序列化 dict。
    """
    return model.model_dump()


class TaskManager:
    """管理 HTTP service tree 任务的启动、切换和状态查询。"""

    def __init__(self, runner, registry, max_task_records=1000):
        """
        初始化任务管理器。
        :param runner: BehaviorTreeRunner 实例。
        :param registry: TaskRegistry 实例。
        :param max_task_records: 内存中最多保留的历史任务记录数量。
        """
        self.runner = runner
        self.registry = registry
        self._max_task_records = max(int(max_task_records), 1)
        self._lock = threading.Lock()
        self._active_task_id = None
        self._active_definition = None
        self._task_status = "IDLE"
        self._message = "no active task"
        self._failed_node = None
        self._created_at = None
        self._finished_at = None
        self._task_records = OrderedDict()

    def start_task_by_endpoint(self, endpoint, payload):
        """
        按 HTTP endpoint 启动任务。
        :param endpoint: 请求路径，例如 /api/start_move_box。
        :param payload: HTTP JSON 请求体。
        :return: 可 JSON 序列化响应。
        """
        with self._lock:
            self._refresh_status_locked()
            if self._active_task_id and self._task_status in ("QUEUED", "RUNNING"):
                return _model_to_dict(StartTaskResponseModel(
                    success=False,
                    taskId=self._active_task_id,
                    taskName=self._active_definition.name if self._active_definition else "",
                    status=self._task_status,
                    message="task is already running",
                ))

            definition = self.registry.get_by_endpoint(endpoint)
            if definition is None:
                return _model_to_dict(StartTaskResponseModel(
                    success=False,
                    taskId="",
                    status="REJECTED",
                    message=f"unsupported task endpoint: {endpoint}",
                ))

            request = definition.adapter.parse_payload(payload)
            task_id = str(uuid.uuid4())
            now = time.time()

            # 关键步骤：先写入本次输入，再 reload 单次树，避免恢复 timer 后新树抢先 tick。
            definition.adapter.write_blackboard(task_id, request)
            reloaded, reload_message = self.runner.reload_tree(definition.tree_file)
            if not reloaded:
                return _model_to_dict(StartTaskResponseModel(
                    success=False,
                    taskId="",
                    taskName=definition.name,
                    status="REJECTED",
                    message=reload_message,
                ))

            self._active_task_id = task_id
            self._active_definition = definition
            self._task_status = "QUEUED"
            self._message = f"{definition.name} task accepted"
            self._failed_node = None
            self._created_at = now
            self._finished_at = None
            self._sync_active_record_locked()
            return _model_to_dict(StartTaskResponseModel(
                success=True,
                taskId=task_id,
                taskName=definition.name,
                status=self._task_status,
                message=self._message,
            ))

    def get_task_status(self, task_id):
        """
        查询任务状态。
        :param task_id: start 接口返回的任务 ID。
        :return: 可 JSON 序列化响应。
        """
        with self._lock:
            self._refresh_status_locked()
            if not task_id:
                return _model_to_dict(TaskStatusResponseModel(
                    success=False,
                    taskId="",
                    status="UNKNOWN",
                    message="missing taskId",
                ))
            if task_id != self._active_task_id:
                record = self._task_records.get(task_id)
                if record:
                    return dict(record)
                return _model_to_dict(TaskStatusResponseModel(
                    success=False,
                    taskId=task_id,
                    status="UNKNOWN",
                    message="unknown taskId",
                ))

            snapshot = self._snapshot()
            current_node = self._find_running_node(snapshot)
            task_name = self._active_definition.name if self._active_definition else ""
            response = _model_to_dict(TaskStatusResponseModel(
                success=True,
                taskId=self._active_task_id,
                taskName=task_name,
                status=self._task_status,
                currentNode=current_node,
                failedNode=self._failed_node,
                rootStatus=snapshot.get("root_status"),
                executionState=snapshot.get("execution_state"),
                tickCount=snapshot.get("tick_count", 0),
                liveRuntime=snapshot.get("live_runtime"),
                message=self._message,
                createdAt=self._created_at,
                finishedAt=self._finished_at,
            ))
            self._store_task_record_locked(self._active_task_id, response)
            return response

    def _refresh_status_locked(self):
        """根据当前单次任务树根节点状态刷新任务状态。"""
        if not self._active_task_id or self._active_definition is None:
            return
        if self._task_status in ("SUCCEEDED", "FAILED"):
            return

        snapshot = self._snapshot()
        root_status = snapshot.get("root_status")
        if self._created_at is not None and float(snapshot.get("generated_at", 0.0) or 0.0) < self._created_at:
            self._task_status = "QUEUED"
            self._message = f"{self._active_definition.name} task accepted, waiting for next tree tick"
            self._sync_active_record_locked()
            return

        if root_status == "FAILURE":
            failed_node = self._find_failed_node(snapshot)
            self._mark_failed_locked(
                self._failure_message(
                    f"{self._active_definition.name} tree failed",
                    failed_node,
                ),
                failed_node=failed_node,
            )
            return

        if root_status == "SUCCESS":
            if self._task_status in ("QUEUED", "RUNNING"):
                self._task_status = "SUCCEEDED"
                self._message = f"{self._active_definition.name} task completed"
                self._failed_node = None
                self._finished_at = self._finished_at or time.time()
                self._sync_active_record_locked()
            return

        # 兼容 reload 后首个 tick 之前的 READY 快照；只有真正 tick 过才进入 RUNNING。
        if root_status in (None, "INVALID"):
            self._task_status = "QUEUED"
            self._message = f"{self._active_definition.name} task accepted, waiting for next tree tick"
            self._sync_active_record_locked()
            return

        self._task_status = "RUNNING"
        self._message = f"{self._active_definition.name} task running"
        self._failed_node = None
        self._sync_active_record_locked()

    def _mark_failed_locked(self, message, failed_node=None):
        """标记当前任务失败。"""
        self._task_status = "FAILED"
        self._message = message
        self._failed_node = failed_node
        self._finished_at = self._finished_at or time.time()
        self._sync_active_record_locked()

    def _sync_active_record_locked(self):
        """把当前任务的生命周期状态同步到任务记录表。"""
        if not self._active_task_id:
            return
        task_name = self._active_definition.name if self._active_definition else ""
        self._store_task_record_locked(self._active_task_id, _model_to_dict(TaskStatusResponseModel(
            success=True,
            taskId=self._active_task_id,
            taskName=task_name,
            status=self._task_status,
            message=self._message,
            failedNode=self._failed_node,
            createdAt=self._created_at,
            finishedAt=self._finished_at,
        )))

    def _store_task_record_locked(self, task_id, record):
        """
        写入任务记录并按上限裁剪最老记录。
        :param task_id: 任务 ID。
        :param record: 任务状态响应 dict。
        """
        if not task_id:
            return
        # 关键步骤：OrderedDict 让任务记录具备明确的淘汰顺序，避免长期运行无限增长。
        self._task_records[task_id] = dict(record)
        self._task_records.move_to_end(task_id)
        self._trim_task_records_locked()

    def _trim_task_records_locked(self):
        """裁剪任务记录缓存，保留当前任务并删除最老的历史记录。"""
        while len(self._task_records) > self._max_task_records:
            oldest_task_id = next(iter(self._task_records))
            if oldest_task_id == self._active_task_id:
                self._task_records.move_to_end(oldest_task_id)
                continue
            self._task_records.popitem(last=False)

    def _snapshot(self):
        snapshot = self.runner.get_snapshot() if self.runner else {}
        return snapshot if isinstance(snapshot, dict) else {}

    @staticmethod
    def _find_running_node(snapshot):
        nodes = snapshot.get("nodes") or {}
        running_nodes = []
        for node_id, node in nodes.items():
            if isinstance(node, dict) and node.get("status") == "RUNNING":
                item = dict(node)
                item["id"] = node_id
                running_nodes.append(item)
        if not running_nodes:
            return None
        running_nodes.sort(key=lambda item: int(item.get("depth", 0)), reverse=True)
        node = running_nodes[0]
        return {
            "id": node.get("id"),
            "label": node.get("label"),
            "name": node.get("name"),
            "type": node.get("type"),
            "status": node.get("status"),
        }

    @staticmethod
    def _find_failed_node(snapshot):
        nodes = snapshot.get("nodes") or {}
        failed_nodes = []
        for node_id, node in nodes.items():
            if isinstance(node, dict) and node.get("status") == "FAILURE":
                item = dict(node)
                item["id"] = node_id
                failed_nodes.append(item)
        if not failed_nodes:
            return None
        failed_nodes.sort(key=lambda item: int(item.get("depth", 0)), reverse=True)
        node = failed_nodes[0]
        failed_node = {
            "id": node.get("id"),
            "label": node.get("label"),
            "name": node.get("name"),
            "type": node.get("type"),
            "status": node.get("status"),
        }
        feedback_message = str(node.get("feedback_message", "") or "")
        if feedback_message:
            failed_node["message"] = feedback_message
        return failed_node

    @staticmethod
    def _failure_message(default_message, failed_node):
        if isinstance(failed_node, dict):
            feedback_message = str(failed_node.get("message", "") or "")
            if feedback_message:
                return feedback_message
        return default_message

    @staticmethod
    def _find_node_status(snapshot, label):
        nodes = snapshot.get("nodes") or {}
        for node in nodes.values():
            if isinstance(node, dict) and node.get("label") == label:
                return node.get("status")
        return None
