"""底盘导航节点。

这个节点把“创建导航任务 -> 轮询状态”拆成多次 tick，
从而让 viewer 能看到更细粒度的 RUNNING 过程。

当前会经历的分解动作如下：

1. CREATE_TASK
   创建底盘导航任务，拿到 task_instance_id。

2. POLL_STATUS
   周期性查询导航任务状态，直到成功或失败。
"""

import math
import time
import uuid

from py_trees import common
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import BASE_LINK_FRAME, CHASSIS_FRAME, FLOW_RESULT_KEY, MAP_FRAME
from ..base import TimedMockAction
from tree.utils.chassis_navigation import (
    DEFAULT_CHASSIS_URL,
    DEFAULT_NAVIGATION_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    TASK_STATUS_LABELS,
    TASK_STATUS_SUCCEEDED,
    TERMINAL_FAILED_TASK_STATUSES,
    build_chassis_config,
    extract_navigation_task_id,
    extract_task_status,
    post_chassis_navigation,
    post_navigation_task_status,
)


class HttpNavigateToPose(TimedMockAction):
    """异步导航到目标位姿。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 导航节点负责写入流程结果，供后续节点读取。
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=common.Access.WRITE)
        self.x = float(params.get("x", 0.0))
        self.y = float(params.get("y", 0.0))
        self.yaw = float(params.get("yaw", 0.0))
        self.target_pose_key = str(params.get("target_pose_key", "")).strip()
        self.enabled_key = str(params.get("enabled_key", "")).strip()
        self.navigation_target_key = str(params.get("navigation_target_key", "navigation_target")).strip()
        if self.target_pose_key:
            self.blackboard.register_key(key=self.target_pose_key, access=common.Access.READ)
        if self.enabled_key:
            self.blackboard.register_key(key=self.enabled_key, access=common.Access.READ)
        if self.navigation_target_key:
            self.blackboard.register_key(key=self.navigation_target_key, access=common.Access.WRITE)
        self.navigation_visualization_enabled = self._to_bool(
            params.get("navigation_visualization_enabled", False)
        )
        self.navigation_visualization_topic = str(
            params.get("navigation_visualization_topic", "/move_box/http_navigation_markers")
        ).strip()
        self.odom_topic = str(params.get("odom_topic", CHASSIS_FRAME)).strip()
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.navigation_visualization_pub = None
        if self.navigation_visualization_enabled and self.navigation_visualization_topic:
            self.navigation_visualization_pub = self.ros_node.create_publisher(
                self.navigation_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip(),
        )
        self.navigation_timeout_sec = int(
            params.get("navigation_timeout_sec", DEFAULT_NAVIGATION_TIMEOUT_SEC)
        )
        self.poll_interval_sec = float(params.get("poll_interval_sec", DEFAULT_POLL_INTERVAL_SEC))
        # 关键步骤：普通导航默认使用 coarse 到点；需要精确到点的节点可在配置中显式关闭。
        self.coarse = self._to_bool(params.get("coarse", True))

        self._phase = "IDLE"
        self._task_id = None
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = None
        self._next_poll_at = None

    def initialise(self):
        super().initialise()
        if not self._enabled():
            self._phase = "SKIP"
            return
        # 进入节点时只做“状态机复位”，不直接访问外部 HTTP。
        # 真正的网络调用留到 update() 中按 phase 分步推进。
        self._refresh_target_from_blackboard(allow_missing=True)
        self._store_navigation_target()
        self._publish_navigation_visualization()
        self._phase = "CREATE_TASK"
        self._task_id = str(uuid.uuid4())
        self._task_instance_id = None
        self._navigation_response = None
        self._navigation_status_response = None
        self._deadline = time.monotonic() + self.navigation_timeout_sec
        self._next_poll_at = None

    def update(self):
        if self.should_use_mock_execution():
            status = self.update_mock_result()
            if status == common.Status.SUCCESS:
                self._refresh_target_from_blackboard(allow_missing=True)
                self._store_navigation_result({
                    "simulated": True,
                    "targetPose": {
                        "x": self.x,
                        "y": self.y,
                        "yaw": self.yaw,
                    },
                })
            return status

        try:
            if not self._enabled():
                self.ros_node.clear_live_runtime()
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 跳过导航任务: enabled_key={self.enabled_key}"
                )
                return common.Status.SUCCESS

            now = time.monotonic()
            if now > self._deadline:
                raise TimeoutError(
                    "wait navigation task success timeout. "
                    f"task_instance_id={self._task_instance_id} "
                    f"latest_response={self._navigation_status_response}"
                )

            if self._phase == "CREATE_TASK":
                # 第一个 tick 只负责创建导航任务，拿到 task_instance_id 后立刻返回 RUNNING。
                self._refresh_target_from_blackboard()
                self._store_navigation_target()
                self._publish_navigation_visualization()
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "NAVIGATION",
                    f"Creating navigation task for ({self.x:.3f}, {self.y:.3f}, {self.yaw:.3f})",
                )
                self._navigation_response = post_chassis_navigation(
                    self.chassis_config,
                    self._task_id,
                    self.x,
                    self.y,
                    self.yaw,
                    coarse=self.coarse,
                )
                self._task_instance_id = extract_navigation_task_id(self._navigation_response)
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 导航任务已创建: "
                    f"task_id={self._task_id}, task_instance_id={self._task_instance_id}"
                )
                self._phase = "POLL_STATUS"
                self._next_poll_at = now
                return common.Status.RUNNING

            if self._phase == "POLL_STATUS":
                # 后续 tick 按固定节奏轮询导航状态，这样 viewer 就能看到中间的 RUNNING 过程。
                if self._next_poll_at is not None and now < self._next_poll_at:
                    self.ros_node.set_live_runtime(
                        self.config_label,
                        "NAVIGATION",
                        f"Waiting to poll navigation status for task {self._task_instance_id}",
                    )
                    return common.Status.RUNNING

                self.ros_node.set_live_runtime(
                    self.config_label,
                    "NAVIGATION",
                    f"Polling navigation status for task {self._task_instance_id}",
                )
                self._navigation_status_response = post_navigation_task_status(
                    self.chassis_config,
                    self._task_instance_id,
                )
                task_status = extract_task_status(self._navigation_status_response)
                status_label = TASK_STATUS_LABELS.get(task_status, "未知状态")
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 导航状态: "
                    f"task_instance_id={self._task_instance_id}, "
                    f"task_status={task_status}({status_label})"
                )
                if task_status == TASK_STATUS_SUCCEEDED:
                    self._store_result()
                    self.ros_node.clear_live_runtime()
                    self.ros_node.get_logger().info(f"[{self.config_label}] 导航完成")
                    return common.Status.SUCCESS

                if task_status in TERMINAL_FAILED_TASK_STATUSES:
                    raise RuntimeError(f"navigation task failed: {self._navigation_status_response}")

                self._next_poll_at = now + self.poll_interval_sec
                return common.Status.RUNNING

            raise RuntimeError(f"unsupported navigation phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] navigation failed: {exc}")
            return common.Status.FAILURE

    def _enabled(self):
        """读取可选 enabled_key，缺省时默认执行导航。"""
        if not self.enabled_key:
            return True
        return self.blackboard.exists(self.enabled_key) and self._to_bool(
            self.blackboard.get(self.enabled_key)
        )

    def _store_result(self):
        result = {
            "taskId": self._task_id,
            "taskInstanceId": self._task_instance_id,
            "navigation": self._navigation_response,
            "navigationStatus": self._navigation_status_response,
        }
        self._store_navigation_result(result)

    def _store_navigation_result(self, navigation_result):
        """只更新流程结果中的 navigation 字段，避免覆盖其它节点结果。"""
        try:
            flow_result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            flow_result = None
        if flow_result is None:
            flow_result = {}

        flow_result["navigation"] = navigation_result
        self.blackboard.flow_result = flow_result

    def _store_navigation_target(self):
        """把当前导航目标写入黑板，供并行监控节点读取。"""
        if not self.navigation_target_key:
            return

        self.blackboard.set(
            self.navigation_target_key,
            {
                "x": self.x,
                "y": self.y,
                "yaw": self.yaw,
            },
            overwrite=True,
        )

    def _refresh_target_from_blackboard(self, allow_missing=False):
        """配置 target_pose_key 时，从黑板读取导航目标覆盖静态参数。"""
        if not self.target_pose_key:
            return

        if not self.blackboard.exists(self.target_pose_key):
            if allow_missing:
                return
            raise RuntimeError(f"blackboard key 不存在: {self.target_pose_key}")

        target_pose = self._parse_target_pose(self.blackboard.get(self.target_pose_key))
        if target_pose is None:
            raise RuntimeError(f"无法解析导航目标: key={self.target_pose_key}")

        # 关键步骤：创建 HTTP 任务前刷新 x/y/yaw，让后续日志、黑板目标和请求体保持一致。
        self.x, self.y, self.yaw = target_pose

    def _parse_target_pose(self, raw_target):
        """兼容 dict/list/tuple 或带 x/y/yaw 属性的导航目标。"""
        if raw_target is None:
            return None

        if isinstance(raw_target, dict):
            if "x" in raw_target and "y" in raw_target:
                return (
                    float(raw_target["x"]),
                    float(raw_target["y"]),
                    float(raw_target.get("yaw", raw_target.get("angle", 0.0))),
                )
            if "position" in raw_target:
                return self._parse_target_pose(raw_target["position"])
            if "pose" in raw_target:
                return self._parse_target_pose(raw_target["pose"])

        if isinstance(raw_target, (list, tuple)) and len(raw_target) >= 2:
            target_yaw = float(raw_target[2]) if len(raw_target) >= 3 else 0.0
            return float(raw_target[0]), float(raw_target[1]), target_yaw

        if hasattr(raw_target, "x") and hasattr(raw_target, "y"):
            return (
                float(raw_target.x),
                float(raw_target.y),
                float(getattr(raw_target, "yaw", 0.0)),
            )

        if hasattr(raw_target, "position"):
            return self._parse_target_pose(raw_target.position)

        if hasattr(raw_target, "pose"):
            return self._parse_target_pose(raw_target.pose)

        return None

    def describe_start(self):
        return (
            f"[{self.config_label}] HttpNavigateToPose start: "
            f"x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f}, "
            f"target_pose_key={self.target_pose_key or '<static>'}, "
            f"enabled_key={self.enabled_key or '<none>'}, "
            f"navigation_target_key={self.navigation_target_key or '<disabled>'}, "
            f"coarse={self.coarse}"
        )

    def _publish_navigation_visualization(self):
        """在 map 下显示当前 HttpNavigateToPose 的导航目标和朝向。"""
        if self.navigation_visualization_pub is None:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        current_pose = self.odom_transformer.get_current_pose()
        target_point = self._point_message(self.x, self.y, 0.08)

        goal_arrow = self._new_navigation_marker(1, "http_navigation_goal", Marker.ARROW)
        goal_arrow.pose.position = target_point
        yaw_rad = math.radians(self.yaw)
        goal_arrow.pose.orientation.z = math.sin(yaw_rad * 0.5)
        goal_arrow.pose.orientation.w = math.cos(yaw_rad * 0.5)
        goal_arrow.scale.x = 0.65
        goal_arrow.scale.y = 0.13
        goal_arrow.scale.z = 0.13
        self._set_navigation_marker_color(goal_arrow, 0.15, 0.85, 1.0, 1.0)
        marker_array.markers.append(goal_arrow)

        if current_pose is not None:
            current_point = self._point_message(current_pose[0], current_pose[1], 0.05)
            robot_marker = self._new_navigation_marker(2, "http_navigation_robot", Marker.SPHERE)
            robot_marker.pose.position = current_point
            robot_marker.scale.x = robot_marker.scale.y = robot_marker.scale.z = 0.12
            self._set_navigation_marker_color(robot_marker, 1.0, 0.85, 0.1, 1.0)
            marker_array.markers.append(robot_marker)

            relation = self._new_navigation_marker(3, "http_navigation_relation", Marker.LINE_LIST)
            relation.scale.x = 0.025
            relation.points = [current_point, target_point]
            self._set_navigation_marker_color(relation, 0.25, 0.8, 1.0, 0.95)
            marker_array.markers.append(relation)

        text = self._new_navigation_marker(4, "http_navigation_text", Marker.TEXT_VIEW_FACING)
        text.pose.position.x = self.x
        text.pose.position.y = self.y
        text.pose.position.z = 0.60
        text.pose.orientation.w = 1.0
        text.scale.z = 0.11
        self._set_navigation_marker_color(text, 1.0, 1.0, 1.0, 1.0)
        text.text = (
            "HTTP NAV GOAL\n"
            f"goal=({self.x:.2f}, {self.y:.2f}) yaw={self.yaw:.1f}deg\n"
            f"source={self.target_pose_key or 'static_params'}\n"
            f"key={self.navigation_target_key or '<disabled>'}"
        )
        marker_array.markers.append(text)

        self.navigation_visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布HTTP导航RViz标记: "
            f"topic={self.navigation_visualization_topic}, "
            f"goal=({self.x:.3f}, {self.y:.3f}, {self.yaw:.3f}), "
            f"source={self.target_pose_key or 'static_params'}"
        )

    def _new_navigation_marker(self, marker_id, namespace, marker_type):
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
    def _set_navigation_marker_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    @staticmethod
    def _point_message(x, y, z):
        return Point(x=float(x), y=float(y), z=float(z))
