"""HTTP service tree 任务注册表。"""

from dataclasses import dataclass


@dataclass
class TaskDefinition:
    """描述一个可由 HTTP endpoint 启动的 service tree。"""

    name: str
    endpoint: str
    tree_file: str
    adapter: object
    root_label: str = ""


class TaskRegistry:
    """维护 endpoint 到任务定义的映射。"""

    def __init__(self):
        self._tasks_by_endpoint = {}

    def register(self, definition: TaskDefinition):
        """
        注册一个 HTTP 任务。
        :param definition: 任务定义，包含 endpoint、tree 文件和 adapter。
        """
        endpoint = self.normalize_endpoint(definition.endpoint)
        self._tasks_by_endpoint[endpoint] = definition

    def get_by_endpoint(self, endpoint: str):
        """
        按 HTTP endpoint 查询任务定义。
        :param endpoint: 请求路径，例如 /api/start_move_box。
        :return: TaskDefinition 或 None。
        """
        return self._tasks_by_endpoint.get(self.normalize_endpoint(endpoint))

    def endpoints(self):
        """返回当前支持的任务 endpoint 列表。"""
        return sorted(self._tasks_by_endpoint.keys())

    @staticmethod
    def normalize_endpoint(endpoint: str) -> str:
        """统一 endpoint 表达，避免尾部斜杠造成匹配失败。"""
        endpoint = str(endpoint or "").strip()
        if not endpoint:
            return "/"
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        if len(endpoint) > 1:
            endpoint = endpoint.rstrip("/")
        return endpoint
