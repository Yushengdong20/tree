"""行为树快照存储。

这个文件负责把运行中的 py_trees 对象整理成前端更容易消费的快照结构。
核心思路是同时保存两种视图：
- 嵌套树形结构，适合前端递归渲染
- 扁平节点字典，适合前端按 id 快速查状态
"""

import threading
import time
from typing import Any, Dict, Optional


class BehaviorTreeSnapshotStore:
    """Build and cache a web-friendly snapshot of the current behaviour tree state.

    自定义 Web Viewer 不直接访问 py_trees 对象，而是读取这里整理好的快照。
    """

    def __init__(self):
        # _node_history 用来保留节点“最近一次活跃状态 / 最近一次终态”，
        # 这样前端即使看到节点当前 INVALID，也能知道它上一次跑到了哪里。
        self._lock = threading.Lock()
        self._node_history: Dict[str, Dict[str, Any]] = {}
        self._last_tick_at: Optional[float] = None
        self._snapshot = {
            "execution_state": "READY",
            "root_status": "INVALID",
            "node_count": 0,
            "last_tick_interval": None,
            "generated_at": time.time(),
            "live_runtime": None,
            "tree": None,
            "nodes": {},
        }

    def update(
        self,
        tree,
        tick_count: int,
        timer,
        execution_state: Optional[str] = None,
        live_runtime: Optional[Dict[str, Any]] = None,
    ):
        """Rebuild both hierarchical and flat tree views after a tick."""
        # 同一份源数据同时维护成树形和扁平结构：
        # 树形适合画 UI，扁平结构适合前端快速按 id 查节点状态。
        now = time.time()
        last_tick_interval = None
        if self._last_tick_at is not None:
            last_tick_interval = now - self._last_tick_at
        self._last_tick_at = now

        tree_state = self._build_runtime_tree(tree.root, depth=0, path="root")
        flat_nodes: Dict[str, Dict[str, Any]] = {}
        self._flatten_tree(tree_state, flat_nodes)

        with self._lock:
            self._snapshot = {
                "execution_state": execution_state
                or ("RUNNING" if not timer.is_canceled() else "STOPPED"),
                "root_status": tree.root.status.name,
                "node_count": len(flat_nodes),
                "tick_count": tick_count,
                "last_tick_interval": last_tick_interval,
                "generated_at": now,
                "live_runtime": dict(live_runtime) if live_runtime else None,
                "tree": tree_state,
                "nodes": flat_nodes,
            }

    def refresh_live_runtime(self, live_runtime: Optional[Dict[str, Any]]):
        """Update only the live runtime side-channel without mutating tick timing statistics."""
        with self._lock:
            self._snapshot["live_runtime"] = dict(live_runtime) if live_runtime else None

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy so external threads do not mutate internal state."""
        with self._lock:
            return dict(self._snapshot)

    def _build_runtime_tree(self, node, depth: int, path: str):
        """Convert a py_trees node into nested JSON-friendly runtime metadata."""
        # path 兼做“唯一节点 id”和“前端层级路径”。
        status = node.status.name
        label = getattr(node, "json_label", node.name)
        node_type = getattr(node, "node_type_raw", node.__class__.__name__)
        previous = self._node_history.get(path, {})

        last_active_status = previous.get("last_active_status")
        last_terminal_status = previous.get("last_terminal_status")
        if status != "INVALID":
            last_active_status = status
        if status in ("SUCCESS", "FAILURE"):
            last_terminal_status = status

        children = []
        if hasattr(node, "children"):
            for index, child in enumerate(node.children):
                child_label = getattr(child, "json_label", child.name)
                child_path = f"{path}/{child_label}[{index}]"
                children.append(self._build_runtime_tree(child, depth + 1, child_path))

        node_info = {
            "id": path,
            "label": label,
            "name": node.name,
            "type": node_type,
            "status": status,
            "depth": depth,
            "timestamp": time.time(),
            "last_active_status": last_active_status,
            "last_terminal_status": last_terminal_status,
            "children": children,
        }
        self._node_history[path] = {
            "last_active_status": last_active_status,
            "last_terminal_status": last_terminal_status,
        }
        return node_info

    def _flatten_tree(self, node_info, output):
        """Mirror the nested tree as an id-indexed map for fast frontend lookup."""
        output[node_info["id"]] = {
            "label": node_info["label"],
            "name": node_info["name"],
            "type": node_info["type"],
            "status": node_info["status"],
            "depth": node_info["depth"],
            "timestamp": node_info["timestamp"],
            "last_active_status": node_info["last_active_status"],
            "last_terminal_status": node_info["last_terminal_status"],
        }
        for child in node_info["children"]:
            self._flatten_tree(child, output)
