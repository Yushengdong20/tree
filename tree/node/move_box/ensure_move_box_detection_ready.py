"""确保 move_box 视觉检测结果可用，必要时通过 HTTP 重置 FoundationPose。"""

import threading
import time

import py_trees
from py_trees.common import Status
from visualization_msgs.msg import MarkerArray

from tree.constants import BASE_LINK_FRAME
from tree.constants import MAP_FRAME
from tree.constants import ROBOT_SERVICES_KEY
from ..base import TimedMockAction
from .fp_grasp_visualization import clear_marker_array, publish_fp_box_and_targets
from tree.runtime.http.move_and_grab_flow import post_json


FOUNDATIONPOSE_RESET_URL = "http://192.168.26.12:13604/foundationpose/reset"


class EnsureMoveBoxDetectionReady(TimedMockAction):
    """等待箱体检测数据有效，超时后调用 FoundationPose HTTP reset。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(params.get("box_center_key", "move_box_latest_box_center")).strip()
        self.grasp_strategy_key = str(
            params.get("grasp_strategy_key", "move_box_grasp_strategy")
        ).strip()
        self.enable_colored_log = self._to_bool(params.get("enable_colored_log", True))
        self.detect_timeout_sec = float(params.get("detect_timeout_sec", 10.0))
        self.poll_interval_sec = float(params.get("poll_interval_sec", 0.2))
        self.restart_on_timeout = self._to_bool(params.get("restart_on_timeout", True))
        self.restart_before_wait = self._to_bool(params.get("restart_before_wait", False))
        self.detection_reset_timeout_sec = float(params.get("detection_reset_timeout_sec", 5.0))
        self.fp_settle_before_refresh_enabled = self._to_bool(
            params.get("fp_settle_before_refresh_enabled", False)
        )
        self.fp_settle_before_refresh_sec = max(
            0.0,
            float(params.get("fp_settle_before_refresh_sec", 1.0)),
        )
        self.fp_grasp_visualization_enabled = self._to_bool(
            params.get("fp_grasp_visualization_enabled", True)
        )
        self.fp_grasp_visualization_topic = str(
            params.get("fp_grasp_visualization_topic", "/move_box/fp_grasp_markers")
        ).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.services = None
        self._phase = "IDLE"
        self._wait_reason = ""
        self._wait_deadline = 0.0
        self._next_poll_at = 0.0
        self._settle_deadline = 0.0
        self._settle_next_phase = ""
        self._settle_reason = ""
        self._settle_wait_reason = ""
        self._reset_thread = None
        self._reset_result = None
        self._reset_generation = 0
        self._reset_lock = threading.Lock()
        self._detection_cache_cleared = False

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.WRITE)
        if self.grasp_strategy_key:
            self.blackboard.register_key(
                key=self.grasp_strategy_key,
                access=py_trees.common.Access.READ,
            )
        self.fp_grasp_visualization_publisher = None
        if self.fp_grasp_visualization_enabled and self.fp_grasp_visualization_topic:
            self.fp_grasp_visualization_publisher = self.ros_node.create_publisher(
                self.fp_grasp_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def initialise(self):
        """初始化非阻塞检测等待状态。"""
        super().initialise()
        self.services = None
        self._reset_thread = None
        self._reset_result = None
        self._reset_generation += 1
        self._detection_cache_cleared = False
        self._settle_deadline = 0.0
        self._settle_next_phase = ""
        self._settle_reason = ""
        self._settle_wait_reason = ""
        self._clear_fp_grasp_visualization()
        if self.restart_before_wait:
            self._start_settle_before_phase("RESTART_BEFORE_WAIT", "初次视觉重置前")
        else:
            self._start_settle_before_phase(
                "WAIT_INITIAL",
                "初次视觉检测轮询前",
                wait_reason="初次等待视觉检测",
            )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        self.services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if self.services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] robot services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE
        self._clear_detection_cache_once("开始等待视觉检测")

        if self._phase in ("WAIT_INITIAL", "WAIT_AFTER_RESTART"):
            return self._update_wait_detection()
        if self._phase in (
            "SETTLE_BEFORE_WAIT",
            "SETTLE_BEFORE_RESTART",
            "SETTLE_AFTER_TIMEOUT",
        ):
            return self._update_settle_before_phase()
        if self._phase in ("RESTART_BEFORE_WAIT", "RESTART_AFTER_TIMEOUT"):
            return self._update_restart_detection()

        self.ros_node.get_logger().error(f"[{self.config_label}] 不支持的检测等待阶段: {self._phase}")
        return Status.FAILURE

    def _start_wait(self, phase, reason):
        """开始一个非阻塞检测等待阶段。"""
        self._phase = phase
        self._wait_reason = reason
        self._wait_deadline = time.monotonic() + self.detect_timeout_sec
        self._next_poll_at = 0.0

    def _update_wait_detection(self):
        """每次 tick 最多轮询一次检测结果。"""
        now = time.monotonic()
        if now > self._wait_deadline:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] {self._wait_reason}超时: "
                f"{self.detect_timeout_sec:.1f}s 内未获得完整检测数据"
            )
            if self._phase == "WAIT_AFTER_RESTART":
                self.ros_node.get_logger().error(f"[{self.config_label}] 重启视觉服务后仍未获得有效箱体检测")
                return Status.FAILURE
            if not self.restart_on_timeout:
                self.ros_node.get_logger().error(f"[{self.config_label}] 视觉检测超时，未配置自动重启")
                return Status.FAILURE

            self._reset_thread = None
            self._reset_result = None
            self._start_settle_before_phase("RESTART_AFTER_TIMEOUT", "视觉检测超时重置前")
            return Status.RUNNING

        if now < self._next_poll_at:
            return Status.RUNNING

        if self._poll_detection_ready(self._wait_reason):
            return Status.SUCCESS

        # 关键步骤：本 tick 未获得检测结果，只记录下次轮询时间并立即让出行为树 tick。
        self._next_poll_at = now + self.poll_interval_sec
        return Status.RUNNING

    def _start_settle_before_phase(self, next_phase, reason, wait_reason=""):
        """在调用 FoundationPose reset 或轮询检测前，可选等待相机/TF/底盘画面稳定。"""
        if not self.fp_settle_before_refresh_enabled or self.fp_settle_before_refresh_sec <= 0.0:
            if next_phase in ("WAIT_INITIAL", "WAIT_AFTER_RESTART"):
                self._start_wait(next_phase, wait_reason or reason)
            else:
                self._phase = next_phase
            return

        self._settle_next_phase = next_phase
        self._settle_reason = reason
        self._settle_wait_reason = wait_reason
        self._settle_deadline = time.monotonic() + self.fp_settle_before_refresh_sec
        if next_phase in ("WAIT_INITIAL", "WAIT_AFTER_RESTART"):
            self._phase = "SETTLE_BEFORE_WAIT"
        elif next_phase == "RESTART_BEFORE_WAIT":
            self._phase = "SETTLE_BEFORE_RESTART"
        else:
            self._phase = "SETTLE_AFTER_TIMEOUT"
        self.ros_node.get_logger().info(
            f"[{self.config_label}] FP刷新前等待相机/TF稳定: "
            f"reason={reason}, wait={self.fp_settle_before_refresh_sec:.3f}s, "
            f"next={next_phase}"
        )

    def _update_settle_before_phase(self):
        now = time.monotonic()
        if now < self._settle_deadline:
            return Status.RUNNING

        next_phase = self._settle_next_phase or "RESTART_BEFORE_WAIT"
        wait_reason = self._settle_wait_reason
        action_desc = (
            "开始轮询视觉检测"
            if next_phase in ("WAIT_INITIAL", "WAIT_AFTER_RESTART")
            else "开始调用视觉重置服务"
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] FP刷新前等待结束，{action_desc}: "
            f"reason={self._settle_reason or '<unset>'}, next={next_phase}"
        )
        if next_phase in ("WAIT_INITIAL", "WAIT_AFTER_RESTART"):
            self._start_wait(next_phase, wait_reason or self._settle_reason or "等待视觉检测")
        else:
            self._phase = next_phase
        self._settle_next_phase = ""
        self._settle_reason = ""
        self._settle_wait_reason = ""
        self._settle_deadline = 0.0
        return Status.RUNNING

    def _poll_detection_ready(self, reason):
        """执行一次检测刷新，并在拿到完整抓取数据后写入 blackboard。"""
        updated = self.services.box_detector.update_latest_grasp_pose(
            self.services.arm_controller.get_initial_left_ypr(),
            self.services.arm_controller.get_initial_right_ypr(),
        )
        if not updated:
            # 关键步骤：没有本轮新检测时不能读取旧缓存，否则 FP 重启后会误用上一轮目标。
            return False

        grasp_pair = self.services.box_detector.get_latest_grasp_pair()
        box_axes = self.services.box_detector.get_latest_box_axes()
        box_center = self.services.box_detector.get_latest_box_center()
        if grasp_pair is None or box_axes is None or box_center is None:
            return False

        self.blackboard.set(self.grasp_pair_key, grasp_pair, overwrite=True)
        self.blackboard.set(self.box_axes_key, box_axes, overwrite=True)
        self.blackboard.set(self.box_center_key, box_center, overwrite=True)
        grasp_strategy = "未设置"
        if self.grasp_strategy_key and self.blackboard.exists(self.grasp_strategy_key):
            grasp_strategy = self.blackboard.get(self.grasp_strategy_key)
        self._publish_fp_grasp_visualization(
            grasp_pair,
            box_axes,
            box_center,
            grasp_strategy,
        )
        result_message = (
            f"[{self.config_label}] {reason}成功: grasp_pair=True, box_axes=True, "
            f"box_center={box_center}（z 为当前 FoundationPose 目标高度）, "
            f"grasp_strategy={grasp_strategy}"
        )
        self.ros_node.get_logger().info(self._color_text(result_message, "highlight"))
        return True

    def _color_text(self, text, color):
        """给关键实机结果增加醒目的 ANSI 颜色，可通过参数关闭。"""
        if not self.enable_colored_log:
            return text

        color_codes = {
            "green": "\033[1;92m",
            "cyan": "\033[1;96m",
            # 粗体亮白字配洋红背景，用于视觉锁定和抓取决策结果。
            "highlight": "\033[1;97;45m",
        }
        color_code = color_codes.get(color, "")
        reset_code = "\033[0m" if color_code else ""
        return f"{color_code}{text}{reset_code}"

    def _update_restart_detection(self):
        """用后台线程执行视觉重置，主 tick 只轮询结果。"""
        if self._reset_thread is None:
            self._reset_result = None
            self._reset_generation += 1
            reset_generation = self._reset_generation
            self._reset_thread = threading.Thread(
                target=self._restart_detection_service_worker,
                args=(reset_generation,),
                name=f"{self.config_label}_reset_detection",
                daemon=True,
            )
            self._reset_thread.start()
            return Status.RUNNING

        if self._reset_thread.is_alive():
            return Status.RUNNING

        with self._reset_lock:
            reset_result = self._reset_result
        if not reset_result or not reset_result.get("ok", False):
            error = "" if reset_result is None else reset_result.get("error", "")
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 视觉重置失败: {error}"
            )
            return Status.FAILURE

        self._reset_thread = None
        self._reset_result = None
        self._clear_detection_cache_once("视觉重置成功后", force=True)
        if self._phase == "RESTART_BEFORE_WAIT":
            self._start_wait("WAIT_INITIAL", "初次等待视觉检测")
        else:
            self._start_wait("WAIT_AFTER_RESTART", "重启后等待视觉检测")
        return Status.RUNNING

    def _restart_detection_service_worker(self, reset_generation):
        """后台调用 HTTP 服务重置 FoundationPose，避免阻塞行为树 tick。"""
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 视觉检测不可用，调用重置 HTTP: {FOUNDATIONPOSE_RESET_URL}"
        )
        # 关键步骤：记录 HTTP 调用耗时，便于判断 reset 卡在网络请求还是服务内部执行慢。
        start_time = time.monotonic()
        try:
            response_payload = post_json(
                FOUNDATIONPOSE_RESET_URL,
                {},
                self.detection_reset_timeout_sec,
                "foundationpose_reset",
            )
        except TimeoutError as exc:
            elapsed_time = time.monotonic() - start_time
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 视觉重置 HTTP 调用超时: "
                f"elapsed={elapsed_time:.3f}s, error={exc}"
            )
            self._store_reset_result(reset_generation, False, str(exc))
            return
        except Exception as exc:
            elapsed_time = time.monotonic() - start_time
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 视觉重置 HTTP 调用失败: "
                f"elapsed={elapsed_time:.3f}s, error={exc}"
            )
            self._store_reset_result(reset_generation, False, str(exc))
            return

        elapsed_time = time.monotonic() - start_time
        response_message = str(response_payload.get("message", ""))
        if response_payload.get("success") is True:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 视觉重置 HTTP 调用成功: "
                f"elapsed={elapsed_time:.3f}s, {response_message}"
            )
            self._store_reset_result(reset_generation, True, "")
            return

        error = (
            "视觉重置 HTTP 返回失败: "
            f"elapsed={elapsed_time:.3f}s, {response_message}"
        )
        self.ros_node.get_logger().error(f"[{self.config_label}] {error}")
        self._store_reset_result(reset_generation, False, error)

    def _store_reset_result(self, reset_generation, ok, error):
        """保存后台 reset 线程的结果，供主 tick 读取。"""
        with self._reset_lock:
            if reset_generation != self._reset_generation:
                return
            self._reset_result = {
                "ok": bool(ok),
                "error": str(error),
            }

    def _clear_detection_cache_once(self, reason, force=False):
        """清空一次检测缓存，确保等待阶段只接受新视觉结果。"""
        if self._detection_cache_cleared and not force:
            return
        if self.services is None or not hasattr(self.services, "box_detector"):
            return
        clear_cache = getattr(self.services.box_detector, "clear_latest_detection_cache", None)
        if clear_cache is None:
            return
        clear_cache()
        self._detection_cache_cleared = True
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已清空 FoundationPose 旧检测缓存: {reason}"
        )

    def _publish_fp_grasp_visualization(
        self,
        grasp_pair,
        box_axes,
        box_center,
        grasp_strategy,
    ):
        """仅发布FP箱体在map下的轮廓与方向轴。

        抓取点/分步目标由 ComputeMoveBox*Targets 节点在策略分支内发布，
        避免 Ensure 阶段提前显示一组可能并不会执行的抓取目标。
        """
        if self.fp_grasp_visualization_publisher is None:
            return
        publish_fp_box_and_targets(
            ros_node=self.ros_node,
            publisher=self.fp_grasp_visualization_publisher,
            topic=self.fp_grasp_visualization_topic,
            config_label=self.config_label,
            odom_transformer=self.odom_transformer,
            services=self.services,
            box_center=box_center,
            box_axes=box_axes,
            strategy=grasp_strategy,
            grasp_pair=grasp_pair,
            target_points=None,
            include_grasp_targets=False,
        )
        return

    def _clear_fp_grasp_visualization(self):
        clear_marker_array(self.fp_grasp_visualization_publisher)
