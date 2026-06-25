"""确保 move_box 视觉检测结果可用，必要时通过 ROS 服务重置 FoundationPose。"""

import time

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class EnsureMoveBoxDetectionReady(TimedMockAction):
    """等待箱体检测数据有效，超时后调用 /foundationpose/reset。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(params.get("box_center_key", "move_box_latest_box_center")).strip()
        self.detect_timeout_sec = float(params.get("detect_timeout_sec", 10.0))
        self.poll_interval_sec = float(params.get("poll_interval_sec", 0.2))
        self.restart_on_timeout = self._to_bool(params.get("restart_on_timeout", True))
        self.restart_before_wait = self._to_bool(params.get("restart_before_wait", False))
        self.detection_reset_service = str(
            params.get("detection_reset_service", "/foundationpose/reset")
        ).strip()
        self.detection_reset_timeout_sec = float(params.get("detection_reset_timeout_sec", 5.0))

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.WRITE)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE

        if self.restart_before_wait:
            if not self._restart_detection_service():
                return Status.FAILURE

        if self._wait_detection_ready(services, reason="初次等待视觉检测"):
            return Status.SUCCESS

        if not self.restart_on_timeout:
            self.ros_node.get_logger().error(f"[{self.config_label}] 视觉检测超时，未配置自动重启")
            return Status.FAILURE

        if not self._restart_detection_service():
            return Status.FAILURE

        if self._wait_detection_ready(services, reason="重启后等待视觉检测"):
            return Status.SUCCESS

        self.ros_node.get_logger().error(f"[{self.config_label}] 重启视觉服务后仍未获得有效箱体检测")
        return Status.FAILURE

    def _wait_detection_ready(self, services, reason):
        """轮询检测器，并在拿到完整抓取数据后写入 blackboard。"""
        deadline = time.monotonic() + self.detect_timeout_sec
        while time.monotonic() <= deadline:
            services.box_detector.update_latest_grasp_pose(
                services.arm_controller.get_initial_left_ypr(),
                services.arm_controller.get_initial_right_ypr(),
            )
            grasp_pair = services.box_detector.get_latest_grasp_pair()
            box_axes = services.box_detector.get_latest_box_axes()
            box_center = services.box_detector.get_latest_box_center()
            if grasp_pair is not None and box_axes is not None and box_center is not None:
                self.blackboard.set(self.grasp_pair_key, grasp_pair, overwrite=True)
                self.blackboard.set(self.box_axes_key, box_axes, overwrite=True)
                self.blackboard.set(self.box_center_key, box_center, overwrite=True)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] {reason}成功: grasp_pair=True, box_axes=True, box_center=True"
                )
                return True

            time.sleep(self.poll_interval_sec)

        self.ros_node.get_logger().warning(
            f"[{self.config_label}] {reason}超时: {self.detect_timeout_sec:.1f}s 内未获得完整检测数据"
        )
        return False

    def _restart_detection_service(self):
        """调用 Trigger 服务重置 FoundationPose，避免行为树直接 SSH 到视觉主机。"""
        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 视觉检测不可用，调用重置服务: {self.detection_reset_service}"
        )
        # 关键步骤：记录服务调用耗时，便于判断 reset 卡在等待服务还是服务内部执行慢。
        start_time = time.monotonic()
        try:
            response = self.ros_node.call_trigger_service(
                self.detection_reset_service,
                timeout_sec=self.detection_reset_timeout_sec,
            )
        except TimeoutError as exc:
            elapsed_time = time.monotonic() - start_time
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 视觉重置服务调用超时: "
                f"elapsed={elapsed_time:.3f}s, error={exc}"
            )
            return False
        except Exception as exc:
            elapsed_time = time.monotonic() - start_time
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 视觉重置服务调用失败: "
                f"elapsed={elapsed_time:.3f}s, error={exc}"
            )
            return False

        elapsed_time = time.monotonic() - start_time
        if getattr(response, "success", False):
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 视觉重置服务调用成功: "
                f"elapsed={elapsed_time:.3f}s, {getattr(response, 'message', '')}"
            )
            return True

        self.ros_node.get_logger().error(
            f"[{self.config_label}] 视觉重置服务返回失败: "
            f"elapsed={elapsed_time:.3f}s, {getattr(response, 'message', '')}"
        )
        return False
