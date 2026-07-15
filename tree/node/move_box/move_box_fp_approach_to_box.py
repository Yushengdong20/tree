"""使用 FoundationPose 箱体中心和前向轴进行近距离精靠近。"""

import math
import time
import uuid

import py_trees
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import (
    BASE_LINK_FRAME,
    CHASSIS_FRAME,
    FINAL_POSE_KEY,
    FLOW_RESULT_KEY,
    MAP_FRAME,
    ROBOT_SERVICES_KEY,
)

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
from tree.utils.geometry import get_odom_pose_transformer, transform_base_point_to_map_with_pose2d


class MoveBoxFpApproachToBox(TimedMockAction):
    """近距离使用 FoundationPose 精定位，导航到最终抓取站位。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
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
        self.odom_topic = str(params.get("odom_topic", CHASSIS_FRAME)).strip()
        self.odom_transformer = get_odom_pose_transformer(
            self.ros_node,
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.navigation_target_key = str(
            params.get("navigation_target_key", "move_box_fp_navigation_target")
        ).strip()
        self.arrival_box_center_key = str(
            params.get("arrival_box_center_key", "move_box_fp_arrival_box_center")
        ).strip()
        self.navigation_visualization_enabled = self._to_bool(
            params.get("navigation_visualization_enabled", True)
        )
        self.navigation_visualization_topic = str(
            params.get(
                "navigation_visualization_topic",
                "/move_box/fp_navigation_markers",
            )
        ).strip()
        self.navigation_visualization_pub = None
        if self.navigation_visualization_enabled and self.navigation_visualization_topic:
            self.navigation_visualization_pub = self.ros_node.create_publisher(
                self.navigation_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=FINAL_POSE_KEY, access=py_trees.common.Access.WRITE)
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
        self._clear_navigation_visualization()
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
                if self._next_poll_at is not None and now < self._next_poll_at:
                    return Status.RUNNING

                self.ros_node.set_live_runtime(self.config_label, "FP_APPROACH", "读取 FoundationPose 箱体位姿")
                services = self._get_services()
                updated = services.box_detector.update_latest_grasp_pose(
                    services.arm_controller.get_initial_left_ypr(),
                    services.arm_controller.get_initial_right_ypr(),
                )
                if not updated:
                    # 关键步骤：没有本轮新检测时继续等待，避免使用视觉重启前的旧箱体中心。
                    self._next_poll_at = now + self.poll_interval_sec
                    self.ros_node.get_logger().warning(
                        f"[{self.config_label}] 尚未收到新的 FoundationPose 箱体位姿，继续等待"
                    )
                    return Status.RUNNING

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
                self._publish_navigation_visualization()
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
            raise RuntimeError(f"robot services missing on blackboard: key={self.services_key}")
        if not hasattr(services, "box_detector"):
            raise RuntimeError("robot services 缺少 box_detector")
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

    def _publish_navigation_visualization(self):
        """在map下显示FP箱心、front axis、精导航目标和误差。"""
        if (
            self.navigation_visualization_pub is None
            or self._box_global_position is None
            or self._target_pose is None
        ):
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        box_map_z = float(self._box_center["z"])
        odom_pose = self.odom_transformer.get_current_pose()
        if odom_pose is not None:
            # 底盘近似保持水平，map箱高 = map下base高度 + base_link下箱高。
            box_map_z = float(odom_pose[2]) + float(self._box_center["z"])
        box_point = self._point_message(
            self._box_global_position["x"],
            self._box_global_position["y"],
            box_map_z,
        )
        target_point = self._point_message(
            self._target_pose.x,
            self._target_pose.y,
            0.08,
        )

        box_marker = self._new_visualization_marker(1, "fp_box_center", Marker.SPHERE)
        box_marker.pose.position = box_point
        box_marker.pose.orientation.w = 1.0
        box_marker.scale.x = box_marker.scale.y = box_marker.scale.z = 0.16
        self._set_marker_color(box_marker, 0.0, 1.0, 0.25, 1.0)
        marker_array.markers.append(box_marker)

        relation = self._new_visualization_marker(2, "fp_navigation_relation", Marker.LINE_LIST)
        relation.scale.x = 0.025
        relation.points = [box_point, target_point]
        self._set_marker_color(relation, 0.65, 0.2, 1.0, 0.95)
        marker_array.markers.append(relation)

        goal_arrow = self._new_visualization_marker(3, "fp_navigation_goal", Marker.ARROW)
        goal_arrow.pose.position.x = self._target_pose.x
        goal_arrow.pose.position.y = self._target_pose.y
        goal_arrow.pose.position.z = 0.08
        target_yaw_rad = math.radians(self._target_pose.yaw)
        goal_arrow.pose.orientation.z = math.sin(target_yaw_rad * 0.5)
        goal_arrow.pose.orientation.w = math.cos(target_yaw_rad * 0.5)
        goal_arrow.scale.x = 0.65
        goal_arrow.scale.y = 0.13
        goal_arrow.scale.z = 0.13
        self._set_marker_color(goal_arrow, 0.75, 0.15, 1.0, 1.0)
        marker_array.markers.append(goal_arrow)

        # front_axis来自base_link，仅旋转当前底盘yaw即可映射到map方向。
        current_yaw_rad = math.radians(self._current_pose.yaw)
        map_axis_x = (
            math.cos(current_yaw_rad) * self._front_axis["x"]
            - math.sin(current_yaw_rad) * self._front_axis["y"]
        )
        map_axis_y = (
            math.sin(current_yaw_rad) * self._front_axis["x"]
            + math.cos(current_yaw_rad) * self._front_axis["y"]
        )
        axis_length = 0.55
        axis_end = self._point_message(
            box_point.x + map_axis_x * axis_length,
            box_point.y + map_axis_y * axis_length,
            box_point.z + self._front_axis["z"] * axis_length,
        )
        front_axis_marker = self._new_visualization_marker(4, "fp_front_axis", Marker.ARROW)
        front_axis_marker.points = [box_point, axis_end]
        front_axis_marker.scale.x = 0.035
        front_axis_marker.scale.y = 0.08
        front_axis_marker.scale.z = 0.11
        self._set_marker_color(front_axis_marker, 0.1, 0.55, 1.0, 1.0)
        marker_array.markers.append(front_axis_marker)

        text = self._new_visualization_marker(5, "fp_navigation_text", Marker.TEXT_VIEW_FACING)
        text.pose.position.x = self._target_pose.x
        text.pose.position.y = self._target_pose.y
        text.pose.position.z = 0.65
        text.pose.orientation.w = 1.0
        text.scale.z = 0.11
        self._set_marker_color(text, 1.0, 1.0, 1.0, 1.0)
        text.text = (
            "FP NAV GOAL\n"
            f"box_map=({box_point.x:.2f}, {box_point.y:.2f}, {box_point.z:.2f})\n"
            f"goal=({self._target_pose.x:.2f}, {self._target_pose.y:.2f}) "
            f"yaw={self._target_pose.yaw:.1f}deg\n"
            f"distance={self.target_distance_m:.2f}m "
            f"error=({self._latest_errors['x']:.2f}, {self._latest_errors['y']:.2f}, "
            f"{self._latest_errors['yaw']:.1f}deg)"
        )
        marker_array.markers.append(text)

        self.navigation_visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布FP导航RViz标记: "
            f"topic={self.navigation_visualization_topic}, "
            f"box=({box_point.x:.3f}, {box_point.y:.3f}, {box_point.z:.3f}), "
            f"goal=({self._target_pose.x:.3f}, {self._target_pose.y:.3f}, "
            f"{self._target_pose.yaw:.3f})"
        )

    def _clear_navigation_visualization(self):
        if self.navigation_visualization_pub is None:
            return
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.navigation_visualization_pub.publish(marker_array)

    def _new_visualization_marker(self, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _set_marker_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    @staticmethod
    def _point_message(x, y, z):
        return Point(x=float(x), y=float(y), z=float(z))

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxFpApproachToBox start"
