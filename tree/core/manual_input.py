"""手动结果控制器。

这个文件负责统一处理两类“人工干预树结果”的来源：
- 主程序终端里输入的 s/f/r
- 外部 topic 发来的控制命令

它本身不跑树，只负责维护：
- 哪些节点正在等待人工结果
- 某个节点被喂了什么结果
"""

import select
import sys
import termios
import threading
import tty
from typing import Dict, List, Optional, Set


class ManualResultController:
    """Manage terminal-driven SUCCESS/FAILURE/RUNNING overrides for leaf nodes.

    无论结果来自主程序终端输入，还是来自外部 sender topic，
    最终都会落到这个控制器里统一排队和消费。
    """

    STATUS_ALIASES = {
        "s": "SUCCESS",
        "f": "FAILURE",
        "r": "RUNNING",
        "success": "SUCCESS",
        "failure": "FAILURE",
        "running": "RUNNING",
    }

    def __init__(self, ros_node):
        # ros_node 作为宿主，提供日志、waiting_nodes 发布以及结果消费入口。
        self.ros_node = ros_node
        self._lock = threading.Lock()
        self._manual_results: Dict[str, str] = {}
        self._waiting_nodes: Set[str] = set()
        self._input_thread: Optional[threading.Thread] = None

    def start_input_thread(self):
        """Start a background prompt so ROS spinning stays responsive."""
        if self._input_thread is not None:
            return
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _input_loop(self):
        # 终端读输入会阻塞，所以这部分必须放在线程里，
        # 否则会把 ROS 主线程的 spin/tick 卡住。
        while self.ros_node.ros.ok():
            try:
                raw = input("manual result> ").strip()
            except EOFError:
                return
            except Exception as exc:
                self.ros_node.get_logger().warning(f"Manual input stopped: {exc}")
                return

            self.handle_input(raw)

    def handle_input(self, raw: str):
        """Parse one terminal command and enqueue the selected result if valid."""
        # 这里的输入语法同时服务于本地终端和外部 topic，因此尽量保持简单统一。
        # 支持：
        # 1. s/f/r
        # 2. list
        # 3. <node_label> s/f/r
        if not raw:
            return

        if raw.lower() == "list":
            waiting = self.get_waiting_nodes()
            self.ros_node.get_logger().info(
                f"Waiting nodes: {', '.join(waiting) if waiting else 'none'}"
            )
            return

        parts = raw.split()
        if len(parts) == 1:
            command = parts[0].lower()
            waiting = self.get_waiting_nodes()
            result = self.STATUS_ALIASES.get(command)
            if result is None:
                self.ros_node.get_logger().info(
                    "Usage: s/f/r, 'list', or '<node_label> s/f/r'"
                )
                return

            if not waiting:
                self.ros_node.get_logger().info("No nodes are waiting for manual input")
                return
            if len(waiting) > 1:
                self.ros_node.get_logger().info(
                    f"Multiple waiting nodes: {', '.join(waiting)}. Use '<node_label> s/f/r'"
                )
                return
            node_label = waiting[0]
        else:
            result = self.STATUS_ALIASES.get(parts[-1].lower())
            node_label = " ".join(parts[:-1])

        if result is None:
            self.ros_node.get_logger().info("Result must be one of s/f/r")
            return

        self.set_result(node_label, result)
        if not getattr(self.ros_node.config, "enable_manual_result_input", False):
            self.ros_node.get_logger().info(
                f"Manual result accepted: {node_label} -> {result}"
            )

    def set_result(self, node_label: str, result: str):
        # 手动结果按“节点 label -> 目标状态”缓存，等对应节点下一次 tick 时消费。
        with self._lock:
            self._manual_results[node_label] = result

    def consume_result(self, node_label: str) -> Optional[str]:
        with self._lock:
            return self._manual_results.pop(node_label, None)

    def mark_waiting(self, node_label: str, waiting: bool):
        """Track which nodes are currently blocked on user input."""
        # waiting 集合是“当前树已经走到这里，但结果还未定”的节点列表。
        # 它既服务于日志提示，也服务于外部 sender/topic 的交互体验。
        with self._lock:
            changed = False
            if waiting:
                if node_label not in self._waiting_nodes:
                    self._waiting_nodes.add(node_label)
                    changed = True
            else:
                if node_label in self._waiting_nodes:
                    self._waiting_nodes.discard(node_label)
                    changed = True
            current_waiting = sorted(self._waiting_nodes)

        # 节点新增等待时提示“现在等谁”，
        # 节点完成后如果仍有剩余等待节点，也继续提示当前剩余列表。
        if changed and current_waiting:
            if getattr(self.ros_node.config, "enable_manual_result_input", False):
                if len(current_waiting) == 1:
                    self.ros_node.get_logger().info(
                        f"Waiting manual result: {current_waiting[0]} | input: s/f/r"
                    )
                else:
                    self.ros_node.get_logger().info(
                        "Waiting manual result: "
                        f"{', '.join(current_waiting)} | input: <node_label> s/f/r"
                    )
            else:
                self.ros_node.get_logger().info(
                    f"Waiting manual result: {', '.join(current_waiting)}"
                )

    def get_waiting_nodes(self) -> List[str]:
        with self._lock:
            return sorted(self._waiting_nodes)


class RawKeyInputController:
    """Capture raw key presses from the main terminal without binding business meaning.

    这个控制器只负责采集原始按键事件，例如：
    - `s`
    - `o`
    - `Enter`

    它不解释这些按键的业务含义；具体如何使用由节点自己决定。
    """

    def __init__(self, ros_node):
        self.ros_node = ros_node
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._input_thread: Optional[threading.Thread] = None
        self._terminal_attr = None

    def start_input_thread(self):
        if self._input_thread is not None:
            return
        self._enable_cbreak_mode()
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def stop(self):
        self._restore_terminal_mode()

    def _input_loop(self):
        while self.ros_node.ros.ok():
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key:
                        shown = {"\n": "\\n", "\r": "\\r"}.get(key, key)
                        self.ros_node.get_logger().info(
                            f"[原始按键输入] 捕获到按键: {shown}"
                        )
                        with self._lock:
                            self._keys.append(key)
            except Exception as exc:
                self.ros_node.get_logger().warning(f"Raw key input stopped: {exc}")
                return

    def consume_key(self, *expected_keys: str) -> Optional[str]:
        """Pop and return the first matching key from the buffered input queue."""
        with self._lock:
            if not self._keys:
                return None

            if not expected_keys:
                return self._keys.pop(0)

            expected = set(expected_keys)
            for index, key in enumerate(self._keys):
                if key in expected:
                    return self._keys.pop(index)
        return None

    def _enable_cbreak_mode(self):
        if not sys.stdin.isatty():
            return
        if self._terminal_attr is not None:
            return
        self._terminal_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def _restore_terminal_mode(self):
        if self._terminal_attr is None:
            return
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal_attr)
        self._terminal_attr = None
