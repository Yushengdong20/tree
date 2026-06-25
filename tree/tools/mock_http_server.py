"""Tiny local mock servers for exercising the real HTTP behaviour-tree path.

这个 mock 文件的定位不是“简化版行为树”，而是“假设备后端”：

- 行为树节点仍然按真实 HTTP 方式发请求
- 这里只是在本机提供一组假的 HTTP 接口进行响应
- 因此它更像一个可控的服务端，而不是树内部的 mock 节点

它主要模拟两类服务：

1. 底盘服务
   - 创建导航任务
   - 查询导航任务状态
   - 获取当前位置

2. 抓取服务
   - 检测箱体
   - 抓箱
   - 放箱
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Set


TASK_STATUS_RUNNING = 1
TASK_STATUS_SUCCEEDED = 3
TASK_STATUS_FAILED = 4


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class NavigationTask:
    """Minimal navigation task record kept by the chassis mock.

    这里只保留行为树真正会关心的最小信息：
    - 任务 id
    - 目标位姿
    - 查询了几次状态
    - 是否已经完成
    """
    task_instance_id: str
    target_pose: Pose2D
    polls: int = 0
    completed: bool = False


def normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


class MockServerState:
    """Share navigation pose and mock business state across both HTTP servers.

    这个状态对象是整个 mock 服务的核心：

    - chassis mock 和 grab mock 共用它
    - 手动输入线程和 HTTP 请求处理线程也共用它

    因此这里用锁和条件变量保护共享状态，避免：
    - 一边在查询导航状态
    - 一边在手动输入 s/f
    时出现读写冲突。
    """

    def __init__(
        self,
        initial_pose: Pose2D,
        navigation_polls_before_success: int,
        target_distance_m: float,
        align_forward_error_m: float,
        align_lateral_error_m: float,
        align_yaw_error_deg: float,
        fail_navigation_create: bool,
        fail_navigation_status: bool,
        fail_detect_box: bool,
        fail_grab_box: bool,
        fail_put_down: bool,
        manual_mode: bool,
    ):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # current_pose 表示“mock 世界里当前底盘所在的位置”。
        # 导航成功时它会被更新，后续 get_current_pose / detect_box 都依赖它。
        self.current_pose = initial_pose
        self.navigation_polls_before_success = max(1, navigation_polls_before_success)
        self.target_distance_m = target_distance_m
        self.align_forward_error_m = align_forward_error_m
        self.align_lateral_error_m = align_lateral_error_m
        self.align_yaw_error_deg = align_yaw_error_deg
        self.fail_navigation_create = fail_navigation_create
        self.fail_navigation_status = fail_navigation_status
        self.fail_detect_box = fail_detect_box
        self.fail_grab_box = fail_grab_box
        self.fail_put_down = fail_put_down
        self.manual_mode = manual_mode

        self.navigation_tasks: Dict[str, NavigationTask] = {}
        # _alignment_target_pose 代表“箱体真正理想的位置/朝向”。
        # detect_box 不是凭空乱造结果，而是根据当前底盘位姿和这个目标位姿之间的差值，
        # 动态计算出前后/左右/角度误差。
        self._alignment_target_pose: Optional[Pose2D] = None
        # 手动模式下，导航和其他动作的结果不会立刻返回，
        # 而是先记在这里，等行为树下一次查询到对应步骤时再消费。
        self._manual_navigation_outcome: Optional[str] = None
        self._manual_action_results: Dict[str, Optional[str]] = {
            "detect": None,
            "grab": None,
            "put": None,
        }
        # waiting_xxx 表示“树那边已经发来了请求，mock 现在正等你给这个步骤一个 s/f 决定”。
        self._waiting_actions: Set[str] = set()
        self._waiting_navigation_task_id: Optional[str] = None

    def create_navigation_task(self, x: float, y: float, yaw: float) -> Dict[str, object]:
        """Simulate /navigation/create_navigation_task.

        这个接口只负责“创建任务”，不代表导航立即成功。
        后续是否成功，要靠 query_navigation_task_status 多次查询来决定。
        """
        with self._lock:
            if self.fail_navigation_create:
                return {"succeeded": False, "message": "mock create navigation failed"}

            task_instance_id = str(uuid.uuid4())
            self.navigation_tasks[task_instance_id] = NavigationTask(
                task_instance_id=task_instance_id,
                target_pose=Pose2D(x=x, y=y, yaw=yaw),
            )
            if self.manual_mode:
                print(
                    "[mock] navigation task created, input s/f -> "
                    f"target=({x:.3f}, {y:.3f}, {yaw:.3f})"
                )
            return {
                "succeeded": True,
                "navigationTaskInfo": {
                    "taskInstanceId": task_instance_id,
                },
            }

    def query_navigation_task_status(self, task_instance_id: str) -> Dict[str, object]:
        """Simulate /navigation/query_navigation_task_status.

        自动模式下：
        - 前几次查询返回 RUNNING
        - 到达设定轮次后返回 SUCCEEDED

        手动模式下：
        - 只有当你在 mock 终端输入 s/f 后
        - 下一次状态查询才真正消费这个决定并返回 SUCCESS/FAILURE
        """
        with self._lock:
            if self.fail_navigation_status:
                return {
                    "succeeded": True,
                    "taskStatusInfo": {
                        "taskStatus": TASK_STATUS_FAILED,
                    },
                }

            task = self.navigation_tasks.get(task_instance_id)
            if task is None:
                return {
                    "succeeded": False,
                    "message": f"unknown task_instance_id: {task_instance_id}",
                }

            if self.manual_mode:
                # 进入这里说明树那边已经开始轮询某个导航任务了，
                # 因此 mock status 会提示“当前下一步该给导航一个 s/f”。
                self._waiting_navigation_task_id = task_instance_id
                if self._manual_navigation_outcome == "fail":
                    self._manual_navigation_outcome = None
                    self._waiting_navigation_task_id = None
                    task.completed = True
                    return {
                        "succeeded": True,
                        "taskStatusInfo": {
                            "taskStatus": TASK_STATUS_FAILED,
                        },
                    }
                if self._manual_navigation_outcome != "success":
                    return {
                        "succeeded": True,
                        "taskStatusInfo": {
                            "taskStatus": TASK_STATUS_RUNNING,
                        },
                    }

                self._manual_navigation_outcome = None
                self._waiting_navigation_task_id = None
                task.completed = True
                # 导航成功后，把当前位姿直接推进到目标点，
                # 后续 get_current_pose / detect_box 都会基于这个新位置继续算。
                self.current_pose = Pose2D(
                    x=task.target_pose.x,
                    y=task.target_pose.y,
                    yaw=normalize_angle_deg(task.target_pose.yaw),
                )
                return {
                    "succeeded": True,
                    "taskStatusInfo": {
                        "taskStatus": TASK_STATUS_SUCCEEDED,
                    },
                }

            task.polls += 1
            if task.polls >= self.navigation_polls_before_success:
                task.completed = True
                self.current_pose = Pose2D(
                    x=task.target_pose.x,
                    y=task.target_pose.y,
                    yaw=normalize_angle_deg(task.target_pose.yaw),
                )
                return {
                    "succeeded": True,
                    "taskStatusInfo": {
                        "taskStatus": TASK_STATUS_SUCCEEDED,
                    },
                }

            return {
                "succeeded": True,
                "taskStatusInfo": {
                    "taskStatus": TASK_STATUS_RUNNING,
                },
            }

    def get_current_pose_payload(self) -> Dict[str, object]:
        """Simulate /localization/get_current_pose using the current mock pose."""
        with self._lock:
            return {
                "pose": {
                    "x": self.current_pose.x,
                    "y": self.current_pose.y,
                    "yaw": self.current_pose.yaw,
                }
            }

    def _ensure_alignment_target(self) -> Pose2D:
        """Create the fixed box target pose once for the whole alignment session.

        这里的“目标位姿”可以理解成：
        - 机器人理想情况下应该面向和接近箱体的那个位置
        - detect_box 返回的误差，实际上就是“当前位姿”和“目标位姿”之间的差
        """
        if self._alignment_target_pose is not None:
            return self._alignment_target_pose

        yaw_rad = math.radians(self.current_pose.yaw)
        # 这里把设定好的前后/左右误差，从车体坐标系转到世界坐标系，
        # 以便构造出一个固定的“箱体目标位姿”。
        global_dx = (
            math.cos(yaw_rad) * self.align_forward_error_m
            - math.sin(yaw_rad) * self.align_lateral_error_m
        )
        global_dy = (
            math.sin(yaw_rad) * self.align_forward_error_m
            + math.cos(yaw_rad) * self.align_lateral_error_m
        )
        self._alignment_target_pose = Pose2D(
            x=self.current_pose.x + global_dx,
            y=self.current_pose.y + global_dy,
            yaw=normalize_angle_deg(self.current_pose.yaw + self.align_yaw_error_deg),
        )
        return self._alignment_target_pose

    def detect_box_payload(self) -> Dict[str, object]:
        """Simulate /detect_box.

        这不是返回一个死值，而是：
        1. 根据当前 current_pose
        2. 和固定的 _alignment_target_pose 做差
        3. 动态生成 boxCenter/frontAxis

        因此当树内部发起一次调整导航后，下一次 detect_box 的误差会自然变小，
        从而形成真实的“检测 -> 调整 -> 再检测”闭环。
        """
        with self._lock:
            if self.fail_detect_box:
                return {"success": False, "message": "mock detect_box failed"}
            if self.manual_mode:
                decision = self._wait_for_manual_action_locked("detect")
                if decision == "fail":
                    return {"success": False, "message": "mock detect_box failed (manual)"}

            target_pose = self._ensure_alignment_target()
            yaw_rad = math.radians(self.current_pose.yaw)
            # 当前位姿和目标位姿先在世界坐标系下求差，再投影回车体坐标系，
            # 得到业务层更关心的 forward/lateral/yaw 三种误差。
            dx_global = target_pose.x - self.current_pose.x
            dy_global = target_pose.y - self.current_pose.y

            forward_error = math.cos(yaw_rad) * dx_global + math.sin(yaw_rad) * dy_global
            lateral_error = -math.sin(yaw_rad) * dx_global + math.cos(yaw_rad) * dy_global
            yaw_error_deg = normalize_angle_deg(target_pose.yaw - self.current_pose.yaw)
            yaw_error_rad = math.radians(yaw_error_deg)

            return {
                "success": True,
                "boxCenter": {
                    "x": self.target_distance_m + forward_error,
                    "y": lateral_error,
                    "z": 0.0,
                },
                "frontAxis": {
                    "x": math.cos(yaw_error_rad),
                    "y": math.sin(yaw_error_rad),
                    "z": 0.0,
                },
            }

    def grab_box_payload(self, arm: Optional[str]) -> Dict[str, object]:
        """Simulate /grab_box with optional manual or forced failure."""
        with self._lock:
            if self.fail_grab_box:
                return {"success": False, "message": "mock grab_box failed", "arm": arm}
            if self.manual_mode:
                decision = self._wait_for_manual_action_locked("grab")
                if decision == "fail":
                    return {"success": False, "message": "mock grab_box failed (manual)", "arm": arm}
            return {"success": True, "message": "mock grab_box ok", "arm": arm}

    def put_down_payload(self, arm: Optional[str]) -> Dict[str, object]:
        """Simulate /put_down with optional manual or forced failure."""
        with self._lock:
            if self.fail_put_down:
                return {"success": False, "message": "mock put_down failed", "arm": arm}
            if self.manual_mode:
                decision = self._wait_for_manual_action_locked("put")
                if decision == "fail":
                    return {"success": False, "message": "mock put_down failed (manual)", "arm": arm}
            return {"success": True, "message": "mock put_down ok", "arm": arm}

    def _wait_for_manual_action_locked(self, action: str) -> str:
        """Block the HTTP handler until the operator provides s/f for this action.

        只有手动模式才会进这里。典型流程是：
        - 行为树发来一个 detect/grab/put 请求
        - mock 打印“waiting for xxx, input s/f”
        - 终端输入把结果写入 _manual_action_results
        - 这里被唤醒并消费结果
        """
        first_wait = action not in self._waiting_actions
        self._waiting_actions.add(action)
        if first_wait:
            print(f"[mock] waiting for {action}, input s/f")
        self._condition.notify_all()
        while self._manual_action_results[action] is None:
            self._condition.wait(timeout=0.5)
        decision = self._manual_action_results[action] or "success"
        self._manual_action_results[action] = None
        self._waiting_actions.discard(action)
        return decision

    def set_manual_navigation_outcome(self, outcome: str) -> None:
        """Queue the next navigation status result for manual mode."""
        with self._lock:
            self._manual_navigation_outcome = outcome
            self._condition.notify_all()

    def set_manual_action_result(self, action: str, outcome: str) -> None:
        """Queue the next detect/grab/put result for manual mode."""
        with self._lock:
            if action not in self._manual_action_results:
                raise ValueError(f"unsupported action: {action}")
            self._manual_action_results[action] = outcome
            self._condition.notify_all()

    def format_status(self) -> str:
        """Render a human-friendly status block for the mock terminal."""
        with self._lock:
            active_tasks = [
                task
                for task in self.navigation_tasks.values()
                if not task.completed
            ]
            lines = [
                "Mock status:",
                f"  current pose: ({self.current_pose.x:.3f}, {self.current_pose.y:.3f}, {self.current_pose.yaw:.3f})",
            ]
            if active_tasks:
                lines.append("  active navigation tasks:")
                for task in active_tasks:
                    lines.append(
                        "    - "
                        f"{task.task_instance_id} -> "
                        f"target=({task.target_pose.x:.3f}, {task.target_pose.y:.3f}, {task.target_pose.yaw:.3f})"
                    )
            else:
                lines.append("  active navigation tasks: none")

            if self._waiting_navigation_task_id is not None:
                lines.append("  next step: input 's' or 'f' for current navigation")
            elif "detect" in self._waiting_actions:
                lines.append("  next step: input 's' or 'f' for detect_box")
            elif "grab" in self._waiting_actions:
                lines.append("  next step: input 's' or 'f' for grab_box")
            elif "put" in self._waiting_actions:
                lines.append("  next step: input 's' or 'f' for put_down")
            else:
                lines.append("  next step: no pending manual decision")

            if self._manual_navigation_outcome is not None:
                lines.append(f"  queued nav decision: {self._manual_navigation_outcome}")
            queued_actions = [
                f"{name}={outcome}"
                for name, outcome in self._manual_action_results.items()
                if outcome is not None
            ]
            if queued_actions:
                lines.append(f"  queued action decisions: {', '.join(queued_actions)}")
            return "\n".join(lines)

    def infer_pending_manual_target(self) -> Optional[str]:
        """Return the currently waiting manual step in priority order.

        mock 终端支持只输入 s/f，因此这里需要判断“当前这次结果应该喂给谁”。
        """
        with self._lock:
            if self._waiting_navigation_task_id is not None:
                return "nav"
            for action in ("detect", "grab", "put"):
                if action in self._waiting_actions:
                    return action
            return None


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, object]:
    """Read and decode one JSON POST body."""
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _write_json(handler: BaseHTTPRequestHandler, payload: Dict[str, object], status: int = 200) -> None:
    """Write one JSON response with explicit headers."""
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(state: MockServerState, role: str):
    """Create a request handler bound to one shared state object and one role.

    role="chassis" 时处理底盘相关接口；
    role="grab" 时处理检测/抓取/放下接口。
    """
    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            try:
                payload = _read_json_body(self)
            except json.JSONDecodeError as exc:
                _write_json(
                    self,
                    {"success": False, "message": f"invalid json body: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            if role == "chassis":
                self._handle_chassis_post(payload)
                return
            if role == "grab":
                self._handle_grab_post(payload)
                return

            _write_json(self, {"success": False, "message": f"unknown role: {role}"}, status=500)

        def log_message(self, format: str, *args):  # noqa: A003
            # 保持 mock 服务输出简洁，避免 http.server 默认访问日志刷屏。
            return

        def _handle_chassis_post(self, payload: Dict[str, object]) -> None:
            # 这里不做复杂路由框架，直接按 path 手动分发，
            # 因为 mock 只需要覆盖当前项目真正会访问到的几个接口。
            if self.path == "/navigation/create_navigation_task":
                poi = payload.get("poi", {})
                x = float(poi.get("x", 0.0))
                y = float(poi.get("y", 0.0))
                yaw = float(poi.get("yaw", 0.0))
                _write_json(self, state.create_navigation_task(x=x, y=y, yaw=yaw))
                return

            if self.path == "/navigation/query_navigation_task_status":
                task_info = payload.get("navigationTaskInfo", {})
                task_instance_id = str(task_info.get("taskInstanceId", ""))
                _write_json(self, state.query_navigation_task_status(task_instance_id))
                return

            if self.path == "/localization/get_current_pose":
                _write_json(self, state.get_current_pose_payload())
                return

            _write_json(
                self,
                {"success": False, "message": f"unsupported chassis path: {self.path}"},
                status=HTTPStatus.NOT_FOUND,
            )

        def _handle_grab_post(self, payload: Dict[str, object]) -> None:
            # 抓取侧接口同样只覆盖当前项目所需的最小集合。
            arm = payload.get("arm")
            if arm is not None:
                arm = str(arm)

            if self.path == "/detect_box":
                _write_json(self, state.detect_box_payload())
                return

            if self.path == "/grab_box":
                _write_json(self, state.grab_box_payload(arm=arm))
                return

            if self.path == "/put_down":
                _write_json(self, state.put_down_payload(arm=arm))
                return

            _write_json(
                self,
                {"success": False, "message": f"unsupported grab path: {self.path}"},
                status=HTTPStatus.NOT_FOUND,
            )

    return MockHandler


def _parse_pose(text: str) -> Pose2D:
    """Parse CLI pose text like '1.0,2.0,90'."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pose must be in 'x,y,yaw' format")
    try:
        x, y, yaw = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid pose value: {exc}") from exc
    return Pose2D(x=x, y=y, yaw=yaw)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for automatic and manual mock runs."""
    parser = argparse.ArgumentParser(
        description="Start local mock HTTP servers for behaviour-tree software simulation.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for both mock servers.")
    parser.add_argument(
        "--chassis-port",
        type=int,
        default=18081,
        help="Port for the chassis mock server.",
    )
    parser.add_argument(
        "--grab-port",
        type=int,
        default=18080,
        help="Port for the grab mock server.",
    )
    parser.add_argument(
        "--initial-pose",
        type=_parse_pose,
        default=Pose2D(0.0, 0.0, 0.0),
        help="Initial chassis pose in 'x,y,yaw' format.",
    )
    parser.add_argument(
        "--navigation-polls-before-success",
        type=int,
        default=2,
        help="How many status polls a navigation task stays RUNNING before succeeding.",
    )
    parser.add_argument(
        "--target-distance-m",
        type=float,
        default=0.4,
        help="Target box distance used to generate /detect_box responses.",
    )
    parser.add_argument(
        "--align-forward-error-m",
        type=float,
        default=0.08,
        help="Initial forward alignment error injected into detect_box.",
    )
    parser.add_argument(
        "--align-lateral-error-m",
        type=float,
        default=0.05,
        help="Initial lateral alignment error injected into detect_box.",
    )
    parser.add_argument(
        "--align-yaw-error-deg",
        type=float,
        default=8.0,
        help="Initial yaw alignment error injected into detect_box.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manually control mock progress from the mock server terminal.",
    )
    parser.add_argument(
        "--fail-navigation-create",
        action="store_true",
        help="Force /navigation/create_navigation_task to fail.",
    )
    parser.add_argument(
        "--fail-navigation-status",
        action="store_true",
        help="Force /navigation/query_navigation_task_status to report failure.",
    )
    parser.add_argument(
        "--fail-detect-box",
        action="store_true",
        help="Force /detect_box to fail.",
    )
    parser.add_argument(
        "--fail-grab-box",
        action="store_true",
        help="Force /grab_box to fail.",
    )
    parser.add_argument(
        "--fail-put-down",
        action="store_true",
        help="Force /put_down to fail.",
    )
    return parser


def _serve_forever(server: ThreadingHTTPServer) -> None:
    """Run one HTTP server loop in a background thread."""
    server.serve_forever(poll_interval=0.2)


def _print_manual_help() -> None:
    print("Manual mock commands:")
    print("  help                 show this help")
    print("  status               show current mock state")
    print("  s / f                apply success or failure to the current waiting step")
    print("  nav success|fail     complete or fail the current navigation task")
    print("  detect success|fail  release a waiting /detect_box request")
    print("  grab success|fail    release a waiting /grab_box request")
    print("  put success|fail     release a waiting /put_down request")


def _run_manual_console(state: MockServerState) -> None:
    """Interactive console for manually feeding mock results.

    这里和行为树主程序的 manual_result_mode 不一样：
    - 主程序 manual_result_mode=True 是“节点不走真实 HTTP，直接等人输 s/f/r”
    - 这里 manual=True 是“节点照常走真实 HTTP，但由 mock 服务端等待你输 s/f”
    """
    _print_manual_help()
    while True:
        try:
            line = input("mock control> ").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            return

        if not line:
            continue
        if line in ("help", "h", "?"):
            _print_manual_help()
            continue
        if line in ("status", "ls"):
            print(state.format_status())
            continue
        if line.lower() in ("s", "f"):
            # 简化后的交互：用户不需要关心是 nav/detect/grab/put，
            # 直接把结果喂给“当前唯一等待中的步骤”。
            pending_target = state.infer_pending_manual_target()
            if pending_target is None:
                print("No pending manual step. Use 'status' to inspect the current mock state.")
                continue
            outcome = "success" if line.lower() == "s" else "fail"
            if pending_target == "nav":
                state.set_manual_navigation_outcome(outcome)
            else:
                state.set_manual_action_result(pending_target, outcome)
            print(state.format_status())
            continue

        parts = line.split()
        if len(parts) != 2:
            print("Invalid command. Use 'help' to see supported commands.")
            continue

        command, outcome = parts[0].lower(), parts[1].lower()
        if outcome not in ("success", "fail"):
            print("Outcome must be 'success' or 'fail'.")
            continue

        try:
            if command == "nav":
                state.set_manual_navigation_outcome(outcome)
            elif command == "detect":
                state.set_manual_action_result("detect", outcome)
            elif command == "grab":
                state.set_manual_action_result("grab", outcome)
            elif command == "put":
                state.set_manual_action_result("put", outcome)
            else:
                print("Unsupported command. Use 'help' to see supported commands.")
                continue
        except ValueError as exc:
            print(str(exc))
            continue

        print(state.format_status())


def main(argv=None) -> None:
    """Start both mock HTTP servers and optionally an interactive manual console."""
    parser = build_parser()
    args = parser.parse_args(argv)

    state = MockServerState(
        initial_pose=args.initial_pose,
        navigation_polls_before_success=args.navigation_polls_before_success,
        target_distance_m=args.target_distance_m,
        align_forward_error_m=args.align_forward_error_m,
        align_lateral_error_m=args.align_lateral_error_m,
        align_yaw_error_deg=args.align_yaw_error_deg,
        fail_navigation_create=args.fail_navigation_create,
        fail_navigation_status=args.fail_navigation_status,
        fail_detect_box=args.fail_detect_box,
        fail_grab_box=args.fail_grab_box,
        fail_put_down=args.fail_put_down,
        manual_mode=args.manual,
    )

    chassis_server = ThreadingHTTPServer(
        (args.host, args.chassis_port),
        make_handler(state=state, role="chassis"),
    )
    grab_server = ThreadingHTTPServer(
        (args.host, args.grab_port),
        make_handler(state=state, role="grab"),
    )

    threads = [
        threading.Thread(target=_serve_forever, args=(chassis_server,), daemon=True),
        threading.Thread(target=_serve_forever, args=(grab_server,), daemon=True),
    ]
    for thread in threads:
        thread.start()

    print("Mock HTTP servers started.")
    print(f"  chassis: http://{args.host}:{args.chassis_port}")
    print(f"  grab   : http://{args.host}:{args.grab_port}")
    print(
        "  mode   : "
        f"manual={args.manual}, "
        f"nav_polls_before_success={args.navigation_polls_before_success}, "
        f"align_errors=(forward={args.align_forward_error_m}, "
        f"lateral={args.align_lateral_error_m}, yaw={args.align_yaw_error_deg})"
    )
    print("Press Ctrl-C to stop.")

    try:
        if args.manual:
            _run_manual_console(state)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        chassis_server.shutdown()
        grab_server.shutdown()
        chassis_server.server_close()
        grab_server.server_close()


if __name__ == "__main__":
    main()
