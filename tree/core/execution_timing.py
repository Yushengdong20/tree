"""行为树节点与子树的运行耗时统计。"""

import time
from collections import defaultdict

import py_trees


class BehaviorTreeExecutionTiming:
    """根据每次 tick 前后的状态变化统计节点真实运行耗时。"""

    TERMINAL_STATUSES = (
        py_trees.common.Status.SUCCESS,
        py_trees.common.Status.FAILURE,
    )

    def __init__(self, logger, enabled=True):
        self.logger = logger
        self.enabled = bool(enabled)
        self._tick_started_at = None
        self._statuses_before_tick = {}
        self._active_started_at = {}
        self._records = []
        self._summary_logged = False

    def before_tick(self, tree):
        if not self.enabled:
            return
        self._tick_started_at = time.monotonic()
        self._statuses_before_tick = {
            node.id: node.status for node in tree.root.iterate()
        }

    def after_tick(self, tree):
        if not self.enabled:
            return
        finished_at = time.monotonic()
        tick_started_at = self._tick_started_at or finished_at

        for node in tree.root.iterate():
            previous_status = self._statuses_before_tick.get(
                node.id, py_trees.common.Status.INVALID
            )
            current_status = node.status

            if current_status == py_trees.common.Status.RUNNING:
                if previous_status != py_trees.common.Status.RUNNING:
                    self._active_started_at[node.id] = self._node_start_time(
                        node, tick_started_at
                    )
                continue

            if current_status in self.TERMINAL_STATUSES:
                # 终态会在后续快照中保持，只有本 tick 新进入终态时打印一次。
                if previous_status == current_status:
                    continue
                started_at = self._active_started_at.pop(
                    node.id,
                    self._node_start_time(node, tick_started_at),
                )
                elapsed_sec = max(0.0, finished_at - started_at)
                self._record_and_log(node, current_status, elapsed_sec)
                continue

            # Selector/Parallel切换分支时可能把RUNNING节点置为INVALID；该次执行没有
            # 返回SUCCESS/FAILURE，因此不计入完成记录。
            if current_status == py_trees.common.Status.INVALID:
                self._active_started_at.pop(node.id, None)

        self._tick_started_at = None

    def log_summary(self):
        """根树结束时输出子树耗时和最慢叶子节点汇总。"""
        if not self.enabled or self._summary_logged:
            return
        self._summary_logged = True
        subtree_records = [record for record in self._records if record["is_subtree"]]
        if subtree_records:
            subtree_text = "; ".join(
                f"{record['label']}={record['elapsed_sec']:.3f}s({record['status']})"
                for record in subtree_records
            )
            self.logger.info(f"[耗时汇总][子树] {subtree_text}")

        leaf_groups = defaultdict(list)
        for record in self._records:
            if record["is_leaf"]:
                leaf_groups[record["label"]].append(record["elapsed_sec"])
        slowest_groups = sorted(
            leaf_groups.items(),
            key=lambda item: sum(item[1]),
            reverse=True,
        )[:10]
        if slowest_groups:
            slowest_text = "; ".join(
                f"{label}: total={sum(values):.3f}s, avg={sum(values) / len(values):.3f}s, "
                f"count={len(values)}"
                for label, values in slowest_groups
            )
            self.logger.info(f"[耗时汇总][最慢叶子TOP10] {slowest_text}")

    @staticmethod
    def _node_start_time(node, fallback):
        # TimedMockAction可提供更精确的initialise时刻；组合节点使用tick起点。
        started_at = getattr(node, "_started_at", None)
        if isinstance(started_at, (int, float)) and started_at > 0.0:
            return float(started_at)
        return fallback

    def _record_and_log(self, node, status, elapsed_sec):
        label = getattr(node, "json_label", node.name)
        node_type = getattr(node, "node_type_raw", node.__class__.__name__)
        is_subtree = bool(getattr(node, "is_subtree_root", False))
        is_leaf = not bool(getattr(node, "children", []))
        path = self._node_path(node)
        record = {
            "label": label,
            "node_type": node_type,
            "status": status.name,
            "elapsed_sec": elapsed_sec,
            "is_subtree": is_subtree,
            "is_leaf": is_leaf,
            "path": path,
        }
        self._records.append(record)

        category = "子树" if is_subtree else "节点"
        extra = ""
        if is_subtree:
            extra = f", file={getattr(node, 'subtree_file', '')}"
        self.logger.info(
            f"[耗时][{category}] status={status.name}, elapsed={elapsed_sec:.3f}s, "
            f"type={node_type}, label={label}, path={path}{extra}"
        )

    @staticmethod
    def _node_path(node):
        labels = []
        current = node
        while current is not None:
            labels.append(getattr(current, "json_label", current.name))
            current = getattr(current, "parent", None)
        return "/".join(reversed(labels))
