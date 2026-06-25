"""使用 move_box 视觉检测结果进行抓箱前底盘对正。

该节点不调用 HTTP `/detect_box`，而是使用 move_box 的 `box_detector`
刷新并读取 blackboard 中的 `box_center / box_axes`。
这些数据默认来自 `~box_pose_topic=/foundationpose/pose`，由 FPBoxDetector 订阅并缓存。
"""

import math
import time
import uuid

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    DEFAULT_NAVIGATION_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    BoxAlignmentMeasurement,
    build_align_config,
    build_chassis_config,
    extract_navigation_task_id,
    extract_task_status,
    get_chassis_current_pose,
    is_box_aligned,
    make_box_adjust_pose,
    normalize_axis_angle_rad,
    post_chassis_navigation,
    post_navigation_task_status,
    TASK_STATUS_LABELS,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_FAILED_TASK_STATUSES,
)


class MoveBoxAlignChassisToBox(TimedMockAction):
    """根据 move_box 检测器缓存的箱体位姿，发起底盘微调导航。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(
            params.get("box_center_key", "move_box_latest_box_center")
        ).strip()
        self.refresh_detection_each_attempt = self._to_bool(
            params.get("refresh_detection_each_attempt", True)
        )
        # 底盘移动仍然走 m1d HTTP 导航协议；这里的区别只是视觉数据不再来自 /detect_box。
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip()
        )
        # 对正参数沿用 HTTP 对正节点的语义：
        # - target_distance_m: 希望箱体中心最终停在机器人前方多远
        # - tolerance_m / longitudinal_tolerance_m / yaw_tolerance_deg: 判定已对齐的容差
        # - max_attempts: 最多允许几轮“检测 -> 导航 -> 再检测”闭环确认
        self.align_config = build_align_config(
            enabled=self._to_bool(params.get("enabled", True)),
            tolerance_m=float(params.get("tolerance_m", 0.06)),
            target_distance_m=float(params.get("target_distance_m", 0.4)),
            longitudinal_tolerance_m=float(params.get("longitudinal_tolerance_m", 0.06)),
            yaw_tolerance_deg=float(params.get("yaw_tolerance_deg", 5.0)),
            max_attempts=int(params.get("max_attempts", 3)),
        )
        self.navigation_timeout_sec = int(
            params.get("navigation_timeout_sec", DEFAULT_NAVIGATION_TIMEOUT_SEC)
        )
        self.poll_interval_sec = float(params.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="flow_result", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="flow_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="final_pose", access=py_trees.common.Access.WRITE)

        self._phase = "IDLE"
        self._attempt_index = 0
        self._current_pose = None
        self._measurement = None
        self._box_snapshot = None
        self._pending_navigation = None

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def initialise(self):
        super().initialise()
        # 每次节点重新进入时都从读取底盘位姿开始。
        # 这样微调后的下一轮检测会基于最新底盘位置，而不是沿用旧误差。
        self._phase = "GET_POSE" if self.align_config.enabled else "DISABLED_POSE"
        self._attempt_index = 0
        self._current_pose = None
        self._measurement = None
        self._box_snapshot = None
        self._pending_navigation = None
        self._ensure_flow_result()["boxAlignment"] = []

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            if self._phase == "DISABLED_POSE":
                pose = get_chassis_current_pose(self.chassis_config)
                self.blackboard.final_pose = {"x": pose.x, "y": pose.y, "yaw": pose.yaw}
                return Status.SUCCESS

            if self._phase == "GET_POSE":
                # 底盘当前位置来自 HTTP /localization/get_current_pose。
                # 后续计算微调目标时，需要把车体坐标系下的误差转换到全局 x/y。
                self._attempt_index += 1
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "MOVE_BOX_ALIGNMENT",
                    f"第 {self._attempt_index}/{self.align_config.max_attempts} 次对正: 读取底盘位姿",
                )
                self._current_pose = get_chassis_current_pose(self.chassis_config)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 当前底盘位姿: "
                    f"x={self._current_pose.x:.3f}, "
                    f"y={self._current_pose.y:.3f}, "
                    f"yaw={self._current_pose.yaw:.3f}"
                )
                self._phase = "REFRESH_BOX_DETECTION"
                return Status.RUNNING

            if self._phase == "REFRESH_BOX_DETECTION":
                # 每次计算误差前都刷新一次 move_box 视觉缓存。
                # 如果只使用第一次写入 blackboard 的旧数据，连续微调会重复执行同一份误差，
                # 机器人可能一直向箱子靠近而发生碰撞。
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "MOVE_BOX_ALIGNMENT",
                    "读取 move_box 视觉缓存并计算对正误差",
                )
                self._measurement, self._box_snapshot = self._make_measurement_from_blackboard()
                self._phase = "EVALUATE_ALIGNMENT"
                return Status.RUNNING

            if self._phase == "EVALUATE_ALIGNMENT":
                # 如果误差已经落在容差内，就不再发底盘微调导航，直接返回 SUCCESS。
                aligned = is_box_aligned(self._measurement, self.align_config)
                self._append_alignment_result(aligned)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 对正误差: "
                    f"横向={self._measurement.lateral_error_m:.3f}m, "
                    f"前后={self._measurement.longitudinal_error_m:.3f}m, "
                    f"朝向={self._measurement.yaw_error_deg:.3f}deg, "
                    f"aligned={aligned}"
                )
                if aligned:
                    self.blackboard.final_pose = {
                        "x": self._current_pose.x,
                        "y": self._current_pose.y,
                        "yaw": self._current_pose.yaw,
                    }
                    self.ros_node.clear_live_runtime()
                    return Status.SUCCESS

                if self._attempt_index >= self.align_config.max_attempts:
                    raise RuntimeError(
                        "move_box chassis alignment failed. "
                        f"max_attempts={self.align_config.max_attempts}"
                    )

                # 根据视觉误差生成下一次底盘目标点：
                # 横向误差控制左右平移，前后误差控制靠近/远离箱子，yaw 误差控制朝向。
                next_pose = make_box_adjust_pose(
                    self._current_pose,
                    self._measurement,
                    self.align_config,
                )
                self._ensure_flow_result()["boxAlignment"][-1]["adjustedPose"] = {
                    "x": next_pose.x,
                    "y": next_pose.y,
                    "yaw": next_pose.yaw,
                }
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 计划微调目标: "
                    f"x={next_pose.x:.3f}, y={next_pose.y:.3f}, yaw={next_pose.yaw:.3f}; "
                    f"delta=({next_pose.x - self._current_pose.x:.3f}, "
                    f"{next_pose.y - self._current_pose.y:.3f}, "
                    f"{next_pose.yaw - self._current_pose.yaw:.3f})"
                )
                self._pending_navigation = _NavigationSubtask(
                    chassis_config=self.chassis_config,
                    x=next_pose.x,
                    y=next_pose.y,
                    yaw=next_pose.yaw,
                    timeout_sec=self.navigation_timeout_sec,
                    poll_interval_sec=self.poll_interval_sec,
                )
                self._phase = "ADJUST_NAVIGATION_CREATE"
                return Status.RUNNING

            if self._phase in ("ADJUST_NAVIGATION_CREATE", "ADJUST_NAVIGATION_POLL"):
                # 微调导航也拆成 create/poll 两个阶段，避免一个 tick 阻塞到导航完成。
                status = self._tick_adjust_navigation()
                if status is None:
                    return Status.RUNNING

                self._ensure_flow_result()["boxAlignment"][-1]["adjustNavigation"] = status
                self._phase = "GET_POSE"
                return Status.RUNNING

            raise RuntimeError(f"unsupported move_box alignment phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] move_box 对正失败: {exc}")
            return Status.FAILURE

    def _ensure_flow_result(self):
        try:
            result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            result = None
        if result is None:
            result = {}
            self.blackboard.flow_result = result
        return result

    def _make_measurement_from_blackboard(self):
        # box_center / box_axes 的原始输入来自 ROS topic：
        # - FoundationPose: /foundationpose/pose，PoseStamped，表示箱体中心 6D 位姿
        # - ArUco: /aruco_single/poses，PoseArray
        if self.refresh_detection_each_attempt:
            self._refresh_detection_to_blackboard()

        box_center = (
            self.blackboard.get(self.box_center_key)
            if self.blackboard.exists(self.box_center_key)
            else None
        )
        box_axes = (
            self.blackboard.get(self.box_axes_key)
            if self.blackboard.exists(self.box_axes_key)
            else None
        )
        if box_center is None or box_axes is None:
            raise RuntimeError(
                "blackboard 中尚未获得有效 move_box 视觉数据: "
                f"box_center={box_center is not None}, box_axes={box_axes is not None}"
            )

        # detector 输出的是箱体左右方向轴和竖直方向轴。
        # 对底盘对正来说，还需要推导箱体前后方向轴，用于计算朝向误差。
        left_axis = _normalize_vector(box_axes["left"])
        up_axis = _normalize_vector(box_axes["up"])
        front_axis = _normalize_vector(_cross(left_axis, up_axis))
        box_center = [float(value) for value in box_center]

        # front_axis 在 base_link 的 xy 平面上的方向，代表箱体前后方向。
        # normalize_axis_angle_rad 会按 180 度周期归一化，因为箱体长边正反通常等价。
        yaw_error_rad = normalize_axis_angle_rad(
            math.atan2(float(front_axis[1]), float(front_axis[0]))
        )
        # box_center 默认已经在 base_link 下：
        # x 表示箱子在机器人前方多远，y 表示相对机器人中轴左右偏差。
        measurement = BoxAlignmentMeasurement(
            lateral_error_m=float(box_center[1]),
            longitudinal_error_m=float(box_center[0]) - self.align_config.target_distance_m,
            yaw_error_deg=math.degrees(yaw_error_rad),
            box_center_x_m=float(box_center[0]),
            box_center_y_m=float(box_center[1]),
            front_axis_x=float(front_axis[0]),
            front_axis_y=float(front_axis[1]),
        )
        snapshot = {
            "boxCenter": {
                "x": float(box_center[0]),
                "y": float(box_center[1]),
                "z": float(box_center[2]),
            },
            "leftAxis": {
                "x": float(left_axis[0]),
                "y": float(left_axis[1]),
                "z": float(left_axis[2]),
            },
            "upAxis": {
                "x": float(up_axis[0]),
                "y": float(up_axis[1]),
                "z": float(up_axis[2]),
            },
            "frontAxis": {
                "x": float(front_axis[0]),
                "y": float(front_axis[1]),
                "z": float(front_axis[2]),
            },
        }
        return measurement, snapshot

    def _refresh_detection_to_blackboard(self):
        """按 RefreshMoveBoxGraspData 的方式刷新检测结果，确保闭环对正使用最新视觉。"""
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            raise RuntimeError(f"move_box services missing on blackboard: key={self.services_key}")

        updated = services.box_detector.update_latest_grasp_pose(
            services.arm_controller.get_initial_left_ypr(),
            services.arm_controller.get_initial_right_ypr(),
        )
        grasp_pair = services.box_detector.get_latest_grasp_pair()
        box_axes = services.box_detector.get_latest_box_axes()
        box_center = services.box_detector.get_latest_box_center()
        self.blackboard.set(self.grasp_pair_key, grasp_pair, overwrite=True)
        self.blackboard.set(self.box_axes_key, box_axes, overwrite=True)
        self.blackboard.set(self.box_center_key, box_center, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 刷新视觉数据: "
            f"updated={updated}, "
            f"grasp_pair={grasp_pair is not None}, "
            f"box_axes={box_axes is not None}, "
            f"box_center={box_center is not None}"
        )

    def _append_alignment_result(self, aligned):
        # 把每轮检测和误差写到 flow_result，方便 Web/API 或后续节点查看对正过程。
        result = self._ensure_flow_result()
        result.setdefault("boxAlignment", []).append(
            {
                "attempt": self._attempt_index,
                "pose": {
                    "x": self._current_pose.x,
                    "y": self._current_pose.y,
                    "yaw": self._current_pose.yaw,
                },
                "lateralErrorM": self._measurement.lateral_error_m,
                "longitudinalErrorM": self._measurement.longitudinal_error_m,
                "yawErrorDeg": self._measurement.yaw_error_deg,
                "aligned": aligned,
                "detectionSource": "move_box_box_detector",
                "detection": self._box_snapshot,
            }
        )

    def _tick_adjust_navigation(self):
        task = self._pending_navigation
        now = time.monotonic()
        if now > task.deadline:
            raise TimeoutError(
                "wait navigation task success timeout. "
                f"task_instance_id={task.task_instance_id} "
                f"latest_response={task.navigation_status_response}"
            )

        if self._phase == "ADJUST_NAVIGATION_CREATE":
            # 创建一次“微调导航”任务。这里和 HttpNavigateToPose 使用同一套底盘协议。
            self.ros_node.set_live_runtime(
                self.config_label,
                "MOVE_BOX_ALIGNMENT",
                f"创建底盘微调导航: ({task.x:.3f}, {task.y:.3f}, {task.yaw:.3f})",
            )
            task.navigation_response = post_chassis_navigation(
                task.chassis_config,
                task.task_id,
                task.x,
                task.y,
                task.yaw,
                coarse=False,
            )
            task.task_instance_id = extract_navigation_task_id(task.navigation_response)
            task.next_poll_at = now
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 微调导航任务已创建: "
                f"task_instance_id={task.task_instance_id}"
            )
            self._phase = "ADJUST_NAVIGATION_POLL"
            return None

        if self._phase == "ADJUST_NAVIGATION_POLL":
            if task.next_poll_at is not None and now < task.next_poll_at:
                return None

            # 只按 poll_interval_sec 查询状态，避免按 tick 周期高频打底盘接口。
            task.navigation_status_response = post_navigation_task_status(
                task.chassis_config,
                task.task_instance_id,
            )
            task_status = extract_task_status(task.navigation_status_response)
            status_label = TASK_STATUS_LABELS.get(task_status, "未知状态")
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 微调导航状态: "
                f"task_instance_id={task.task_instance_id}, "
                f"task_status={task_status}({status_label})"
            )
            if task_status == TASK_STATUS_SUCCEEDED:
                # 微调到点后不直接判定整个对正成功，而是回到 GET_POSE 重新检测。
                # 这样可以抵消底盘定位误差和视觉噪声。
                result = {
                    "taskId": task.task_id,
                    "taskInstanceId": task.task_instance_id,
                    "navigation": task.navigation_response,
                    "navigationStatus": task.navigation_status_response,
                }
                self._pending_navigation = None
                return result
            if task_status in TERMINAL_FAILED_TASK_STATUSES:
                raise RuntimeError(f"navigation task failed: {task.navigation_status_response}")

            task.next_poll_at = now + task.poll_interval_sec
            return None

        raise RuntimeError(f"unsupported adjustment navigation phase: {self._phase}")

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxAlignChassisToBox start"


class _NavigationSubtask:
    def __init__(
        self,
        chassis_config,
        x: float,
        y: float,
        yaw: float,
        timeout_sec: int,
        poll_interval_sec: float,
    ):
        self.chassis_config = chassis_config
        self.x = x
        self.y = y
        self.yaw = yaw
        self.task_id = str(uuid.uuid4())
        self.task_instance_id = None
        self.navigation_response = None
        self.navigation_status_response = None
        self.deadline = time.monotonic() + timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.next_poll_at = None


def _normalize_vector(vector):
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-8:
        raise ValueError("箱体方向轴长度过小，无法计算底盘对正误差")
    return [value / norm for value in values]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
