"""使用 FoundationPose 箱体中心和前向轴进行近距离精靠近。"""

import time
import uuid

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    DEFAULT_FP_TARGET_DISTANCE_M,
    DEFAULT_NAVIGATION_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    TASK_STATUS_LABELS,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_FAILED_TASK_STATUSES,
    build_chassis_config,
    build_fp_approach_pose,
    extract_navigation_task_id,
    extract_task_status,
    get_chassis_current_pose,
    normalize_angle_deg,
    post_chassis_navigation,
    post_navigation_task_status,
    transform_global_point_to_base,
)
from tree.utils.geometry import transform_base_point_to_map_with_pose2d


class MoveBoxFpApproachToBox(TimedMockAction):
    """近距离使用 FoundationPose 精定位，导航到最终抓取站位。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(params.get("box_center_key", "move_box_latest_box_center")).strip()
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip(),
        )
        self.target_distance_m = float(params.get("target_distance_m", DEFAULT_FP_TARGET_DISTANCE_M))
        self.navigation_timeout_sec = int(
            params.get("navigation_timeout_sec", DEFAULT_NAVIGATION_TIMEOUT_SEC)
        )
        self.poll_interval_sec = float(params.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))
        self.navigation_target_key = str(
            params.get("navigation_target_key", "move_box_fp_navigation_target")
        ).strip()
        self.arrival_box_center_key = str(
            params.get("arrival_box_center_key", "move_box_fp_arrival_box_center")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="flow_result", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="final_pose", access=py_trees.common.Access.WRITE)
        if self.arrival_box_center_key:
            self.blackboard.register_key(
                key=self.arrival_box_center_key,
                access=py_trees.common.Access.WRITE,
            )
        if self.navigation_target_key:
            self.blackboard.register_key(
                key=self.navigation_target_key,
                access=py_trees.common.Access.WRITE,
            )
        self._reset_state()

    def _reset_state(self):
        self._phase = "IDLE"
        self._current_pose = None
        self._box_center = None
        self._front_axis = None
        self._box_global_position = None
        self._arrival_box_center = None
        self._target_pose = None
        self._task_id = None
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = None
        self._next_poll_at = None
        self._latest_errors = None

    def initialise(self):
        super().initialise()
        self._reset_state()
        self._clear_navigation_target_pose()
        self._clear_arrival_box_center()
        self._phase = "GET_POSE"
        self._deadline = time.monotonic() + self.navigation_timeout_sec

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            now = time.monotonic()
            if now > self._deadline:
                raise TimeoutError(
                    f"FP 精靠近导航超时: task_instance_id={self._task_instance_id}, "
                    f"latest_response={self._navigation_status_response}"
                )

            if self._phase == "GET_POSE":
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "FP_APPROACH",
                    "FP 精靠近: 读取底盘当前位姿",
                )
                self._current_pose = get_chassis_current_pose(self.chassis_config)
                self._phase = "READ_FP"
                return Status.RUNNING

            if self._phase == "READ_FP":
                self.ros_node.set_live_runtime(self.config_label, "FP_APPROACH", "读取 FoundationPose 箱体位姿")
                services = self._get_services()
                updated = services.box_detector.update_latest_grasp_pose(
                    services.arm_controller.get_initial_left_ypr(),
                    services.arm_controller.get_initial_right_ypr(),
                )
                grasp_pair = services.box_detector.get_latest_grasp_pair()
                box_axes = services.box_detector.get_latest_box_axes()
                box_center = services.box_detector.get_latest_box_center()
                front_axis = services.box_detector.get_latest_box_front_axis()
                self.blackboard.set(self.grasp_pair_key, grasp_pair, overwrite=True)
                self.blackboard.set(self.box_axes_key, box_axes, overwrite=True)
                self.blackboard.set(self.box_center_key, box_center, overwrite=True)
                if box_center is None or front_axis is None:
                    raise RuntimeError(
                        "尚未获得有效 FoundationPose 箱体中心/前向轴: "
                        f"updated={updated}, box_center={box_center is not None}, "
                        f"front_axis={front_axis is not None}"
                    )

                self._box_center = {
                    "x": float(box_center[0]),
                    "y": float(box_center[1]),
                    "z": float(box_center[2]),
                }
                self._front_axis = {
                    "x": float(front_axis[0]),
                    "y": float(front_axis[1]),
                    "z": float(front_axis[2]),
                }
                self._box_global_position = transform_base_point_to_map_with_pose2d(
                    self._current_pose,
                    self._box_center,
                )
                self._target_pose = build_fp_approach_pose(
                    self._current_pose,
                    self._box_center,
                    self._front_axis,
                    self.target_distance_m,
                )
                self._arrival_box_center = self._calculate_arrival_box_center()
                self._latest_errors = self._calculate_world_errors()
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] FP 精靠近目标与误差: "
                    f"box_center=({self._box_center['x']:.3f}, {self._box_center['y']:.3f}, "
                    f"{self._box_center['z']:.3f}), "
                    f"front_axis=({self._front_axis['x']:.3f}, {self._front_axis['y']:.3f}, "
                    f"{self._front_axis['z']:.3f}), "
                    f"target=({self._target_pose.x:.3f}, {self._target_pose.y:.3f}, {self._target_pose.yaw:.3f}), "
                    f"arrival_box=({self._arrival_box_center['x']:.3f}, "
                    f"{self._arrival_box_center['y']:.3f}, {self._arrival_box_center['z']:.3f}), "
                    f"error=(x={self._latest_errors['x']:.3f}, y={self._latest_errors['y']:.3f}, "
                    f"yaw={self._latest_errors['yaw']:.3f})"
                )
                self._phase = "CREATE_NAVIGATION"
                return Status.RUNNING

            if self._phase == "CREATE_NAVIGATION":
                self._task_id = str(uuid.uuid4())
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "FP_APPROACH",
                    f"创建 FP 精靠近导航: ({self._target_pose.x:.3f}, "
                    f"{self._target_pose.y:.3f}, {self._target_pose.yaw:.3f})",
                )
                self._navigation_response = post_chassis_navigation(
                    self.chassis_config,
                    self._task_id,
                    self._target_pose.x,
                    self._target_pose.y,
                    self._target_pose.yaw,
                    coarse=False,
                )
                self._task_instance_id = extract_navigation_task_id(self._navigation_response)
                self._store_navigation_target_pose()
                self._store_arrival_box_center()
                self._next_poll_at = now
                self._phase = "POLL_NAVIGATION"
                return Status.RUNNING

            if self._phase == "POLL_NAVIGATION":
                if self._next_poll_at is not None and now < self._next_poll_at:
                    return Status.RUNNING
                self._navigation_status_response = post_navigation_task_status(
                    self.chassis_config,
                    self._task_instance_id,
                )
                task_status = extract_task_status(self._navigation_status_response)
                status_label = TASK_STATUS_LABELS.get(task_status, "未知状态")
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] FP 精靠近导航状态: "
                    f"task_instance_id={self._task_instance_id}, task_status={task_status}({status_label})"
                )
                if task_status == TASK_STATUS_SUCCEEDED:
                    # 关键步骤：导航成功后不再重复读取 FP 调整，认为本次导航已经准确到位。
                    self._current_pose = get_chassis_current_pose(self.chassis_config)
                    return self._finish_success()
                if task_status in TERMINAL_FAILED_TASK_STATUSES:
                    raise RuntimeError(f"FP 精靠近导航失败: {self._navigation_status_response}")
                self._next_poll_at = now + self.poll_interval_sec
                return Status.RUNNING

            raise RuntimeError(f"unsupported FP approach phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] FP 精靠近失败: {exc}")
            return Status.FAILURE

    def _finish_success(self):
        self._store_result()
        self.blackboard.final_pose = {
            "x": self._current_pose.x,
            "y": self._current_pose.y,
            "yaw": self._current_pose.yaw,
        }
        self.ros_node.clear_live_runtime()
        return Status.SUCCESS

    def _get_services(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            raise RuntimeError(f"move_box services missing on blackboard: key={self.services_key}")
        if not hasattr(services, "box_detector"):
            raise RuntimeError("move_box services 缺少 box_detector")
        return services

    def _store_result(self):
        try:
            flow_result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            flow_result = None
        if flow_result is None:
            flow_result = {}
        flow_result["fpApproach"] = {
            "currentPose": {
                "x": self._current_pose.x,
                "y": self._current_pose.y,
                "yaw": self._current_pose.yaw,
            },
            "boxCenter": self._box_center,
            "frontAxis": self._front_axis,
            "boxGlobalPosition": self._box_global_position,
            "arrivalBoxCenter": self._arrival_box_center,
            "targetDistanceM": self.target_distance_m,
            "targetPose": {
                "x": self._target_pose.x,
                "y": self._target_pose.y,
                "yaw": self._target_pose.yaw,
            },
            "latestErrors": self._latest_errors,
            "navigation": self._navigation_response,
            "navigationStatus": self._navigation_status_response,
        }
        self.blackboard.flow_result = flow_result

    def _calculate_world_errors(self):
        return {
            "x": self._target_pose.x - self._current_pose.x,
            "y": self._target_pose.y - self._current_pose.y,
            "yaw": normalize_angle_deg(self._target_pose.yaw - self._current_pose.yaw),
        }

    def _calculate_arrival_box_center(self):
        """按 FP 导航目标位姿反算到点后箱体在 base_link 下的位置。"""
        arrival_xy = transform_global_point_to_base(
            self._target_pose,
            self._box_global_position["x"],
            self._box_global_position["y"],
        )
        return {
            "x": float(arrival_xy["x"]),
            "y": float(arrival_xy["y"]),
            # 关键步骤：底盘只做平面导航，箱体高度沿用本次 FP 读取结果。
            "z": float(self._box_center["z"]),
        }

    def _store_arrival_box_center(self):
        """把预计到点后的箱体中心写入黑板，供躯干并行动作读取。"""
        if not self.arrival_box_center_key or self._arrival_box_center is None:
            return
        self.blackboard.set(
            self.arrival_box_center_key,
            [
                self._arrival_box_center["x"],
                self._arrival_box_center["y"],
                self._arrival_box_center["z"],
            ],
            overwrite=True,
        )

    def _clear_arrival_box_center(self):
        """清空上一轮预计到点箱体中心，避免并行节点读取旧值。"""
        if not self.arrival_box_center_key:
            return
        self.blackboard.set(self.arrival_box_center_key, None, overwrite=True)

    def _store_navigation_target_pose(self):
        """把 FP 精靠近目标写入黑板，供接近目标监听节点读取。"""
        if not self.navigation_target_key or self._target_pose is None:
            return

        # 关键步骤：导航任务真正创建后才写入目标，避免监听节点在发车前抢跑。
        self.blackboard.set(
            self.navigation_target_key,
            {
                "x": self._target_pose.x,
                "y": self._target_pose.y,
                "yaw": self._target_pose.yaw,
            },
            overwrite=True,
        )

    def _clear_navigation_target_pose(self):
        """清空上一轮 FP 导航目标，避免监听节点误用旧目标。"""
        if not self.navigation_target_key:
            return
        self.blackboard.set(self.navigation_target_key, None, overwrite=True)

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxFpApproachToBox start"
