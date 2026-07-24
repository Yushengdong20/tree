"""持续发布“手中箱子”估计可视化。

这个节点用于实机调试码垛/搬运过程中的箱体落点问题。

为什么需要它：
- ArmsToPose 的 /move_box/claw_point_diagnostics_markers 只在动作节点结束时发布一次；
- ComputeMoveBoxPalletPlaceStrategy 的蓝色 held_box_estimate 也只在策略计算时刷新；
- 但实机排查时更需要“闭爪抓住箱子之后，箱子随当前夹爪实际位置持续刷新”的视角。

运行方式：
1. action=start 时创建一个 ROS timer，节点自身立即返回 SUCCESS，不阻塞行为树。
2. timer 按 publish_interval_sec 周期读取当前左右 EEF 实际位姿。
3. 使用 arm_controller 的 end_effector -> claw 外参，计算左右夹爪点。
4. 通过共享 odom transformer 把 base_link 下夹爪点投到 map。
5. ``action=capture`` 时，把 FP 箱心相对左右夹爪中点的局部偏移锁定到黑板；
   ``action=capture_start`` 则在锁定成功后立即启动同一个持续发布 timer。
6. 以左右夹爪中点作为默认箱心；若存在已锁定偏移则优先使用它，否则使用
   box_center_offset_*_m 做调试补偿。
7. 发布 MarkerArray：估计箱体、左右夹爪点、夹爪连线、箱心、已计算动作点、文字说明。
8. action=stop 时只停止 timer，默认保留最后一帧话题内容，方便 RViz 现场分析。

注意：
- 这个节点不直接下发手臂命令；但 ``capture`` 写出的抓取偏移可被后续放置
  控制节点读取，作为运动学箱体估计的统一来源。
"""

import math
import time

import numpy as np
import py_trees
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.chassis_navigation import Pose2D
from tree.utils.geometry import transform_base_point_to_global
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction


class MoveBoxHeldBoxVisualizationMonitor(TimedMockAction):
    """启动/停止抓箱后的持续箱体估计可视化。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.action = str(params.get("action", "start")).strip().lower()
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.monitor_state_key = str(
            params.get("monitor_state_key", "move_box_held_box_visualization_monitor")
        ).strip()
        self.held_box_transform_key = str(
            params.get("held_box_transform_key", "move_box_held_box_grasp_transform")
        ).strip()
        self.use_captured_transform = self._to_bool(params.get("use_captured_transform", True))
        self.capture_timeout_sec = max(float(params.get("capture_timeout_sec", 2.0)), 0.1)
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/held_box_estimate_markers")
        ).strip()
        self.base_link_visualization_enabled = self._to_bool(
            params.get("base_link_visualization_enabled", True)
        )
        self.publish_interval_sec = max(0.03, float(params.get("publish_interval_sec", 0.10)))
        self.log_interval_sec = max(0.2, float(params.get("log_interval_sec", 2.0)))

        # 箱体尺寸用于 RViz 显示。默认按当前实测箱体：左右长边 0.60m，前后短边 0.40m，高 0.34m。
        self.box_size_x = float(params.get("box_size_x", 0.60))
        self.box_size_y = float(params.get("box_size_y", 0.40))
        self.box_size_z = float(params.get("box_size_z", 0.34))

        # 以左右夹爪中点为基准，在估计箱体局部坐标系下附加偏移。
        # x：沿左右夹爪连线/箱体长边；y：箱体短边方向；z：竖直方向。
        self.box_center_offset_x_m = float(params.get("box_center_offset_x_m", 0.0))
        self.box_center_offset_y_m = float(params.get("box_center_offset_y_m", 0.0))
        self.box_center_offset_z_m = float(params.get("box_center_offset_z_m", 0.0))
        self.fallback_yaw_offset_deg = float(params.get("fallback_yaw_offset_deg", 0.0))
        self.action_points_enabled = self._to_bool(params.get("action_points_enabled", True))
        self.pre_place_left_claw_point_key = str(
            params.get("pre_place_left_claw_point_key", "move_box_pallet_pre_place_left_claw_point")
        ).strip()
        self.pre_place_right_claw_point_key = str(
            params.get("pre_place_right_claw_point_key", "move_box_pallet_pre_place_right_claw_point")
        ).strip()
        self.push_left_claw_point_key = str(
            params.get("push_left_claw_point_key", "move_box_pallet_push_left_claw_point")
        ).strip()
        self.push_right_claw_point_key = str(
            params.get("push_right_claw_point_key", "move_box_pallet_push_right_claw_point")
        ).strip()
        self.lift_left_claw_point_key = str(
            params.get("lift_left_claw_point_key", "move_box_pallet_lift_left_claw_point")
        ).strip()
        self.lift_right_claw_point_key = str(
            params.get("lift_right_claw_point_key", "move_box_pallet_lift_right_claw_point")
        ).strip()

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.monitor_state_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.monitor_state_key, access=py_trees.common.Access.READ)
        if self.held_box_transform_key:
            self.blackboard.register_key(
                key=self.held_box_transform_key, access=py_trees.common.Access.READ
            )
            if self.action in ("capture", "snapshot", "lock", "capture_start", "lock_start"):
                self.blackboard.register_key(
                    key=self.held_box_transform_key, access=py_trees.common.Access.WRITE
                )
        if self.action_points_enabled:
            for key in (
                self.pre_place_left_claw_point_key,
                self.pre_place_right_claw_point_key,
                self.push_left_claw_point_key,
                self.push_right_claw_point_key,
                self.lift_left_claw_point_key,
                self.lift_right_claw_point_key,
            ):
                if key:
                    self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.visualization_pub = None
        if self.visualization_topic:
            self.visualization_pub = self.ros_node.create_publisher(
                self.visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self._timer = None
        self._last_log_at = 0.0
        self._last_publish_error_at = 0.0
        self._capture_deadline = None
        self._capture_then_start_ready = False

    def initialise(self):
        super().initialise()
        # 行为树 Repeat/循环再次进入本节点时必须重新锁定当前抓取的箱体偏移，
        # 不能沿用上轮 action=capture_start 的完成状态。
        self._capture_then_start_ready = False
        self._capture_deadline = None

    def update(self):
        if self.action in ("capture", "snapshot", "lock"):
            return self._capture_held_box_transform()
        if self.action in ("capture_start", "lock_start"):
            if not self._capture_then_start_ready:
                capture_status = self._capture_held_box_transform()
                if capture_status != Status.SUCCESS:
                    return capture_status
                self._capture_then_start_ready = True
            # 这两个动作必须作为同一个行为节点完成：避免 capture 成功后树在
            # 下一个叶子节点才启动 timer，从而出现短暂的可视化断档或配置分叉。
        if self.action in ("stop", "clear", "disable"):
            self._stop_existing_monitor()
            if self.action == "clear" or self._to_bool(self.params.get("clear_on_stop", False)):
                self._publish_clear()
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已停止手中箱体持续可视化: topic={self.visualization_topic}"
            )
            return Status.SUCCESS

        if self.action not in ("start", "enable", "capture_start", "lock_start"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] unsupported action={self.action!r}, expected "
                "capture/capture_start/start/stop"
            )
            return Status.FAILURE

        if self.visualization_pub is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] visualization_topic 为空，跳过手中箱体持续可视化"
            )
            return Status.SUCCESS

        existing_state = self._get_monitor_state()
        existing_timer = existing_state.get("timer") if existing_state else None
        if existing_timer is not None and not existing_timer.is_canceled():
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 手中箱体持续可视化已在运行: "
                f"topic={existing_state.get('topic', self.visualization_topic)}"
            )
            return Status.SUCCESS

        self._timer = self.ros_node.create_timer(self.publish_interval_sec, self._on_timer)
        self.blackboard.set(
            self.monitor_state_key,
            {
                "timer": self._timer,
                "topic": self.visualization_topic,
                "started_at": time.monotonic(),
                "owner": self.config_label,
            },
            overwrite=True,
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已启动手中箱体持续可视化: "
            f"topic={self.visualization_topic}, interval={self.publish_interval_sec:.2f}s, "
            f"box=({self.box_size_x:.2f},{self.box_size_y:.2f},{self.box_size_z:.2f}), "
            f"center_offset=({self.box_center_offset_x_m:.3f},"
            f"{self.box_center_offset_y_m:.3f},{self.box_center_offset_z_m:.3f})"
        )
        return Status.SUCCESS

    def _capture_held_box_transform(self):
        """闭爪后锁定 FP 箱心相对双爪中点的局部偏移。"""
        if not self.held_box_transform_key:
            self.ros_node.get_logger().error(f"[{self.config_label}] held_box_transform_key 不能为空")
            return Status.FAILURE
        if self._capture_deadline is None:
            self._capture_deadline = time.monotonic() + self.capture_timeout_sec
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 开始锁定手中箱体抓取偏移: "
                f"key={self.held_box_transform_key}, timeout={self.capture_timeout_sec:.1f}s"
            )

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        arm_controller = getattr(services, "arm_controller", None) if services else None
        detector = getattr(services, "box_detector", None) if services else None
        claw_pair = self._get_current_claw_pair_base()
        if arm_controller is None or detector is None or claw_pair is None:
            return self._capture_wait_or_fail(
                f"services={'ok' if services is not None else 'missing'}, "
                f"detector={'ok' if detector is not None else 'missing'}, "
                f"claw_pair={'ok' if claw_pair is not None else 'missing'}"
            )

        updater = getattr(detector, "update_latest_grasp_pose", None)
        if callable(updater) and not updater(
            arm_controller.get_initial_left_ypr(), arm_controller.get_initial_right_ypr()
        ):
            return self._capture_wait_or_fail("等待可解析的 FoundationPose 抓取数据")
        center_reader = getattr(detector, "get_latest_box_center", None)
        center = center_reader() if callable(center_reader) else None
        try:
            center = np.array(center, dtype=float)
        except (TypeError, ValueError):
            center = None
        if center is None or center.shape != (3,) or not np.all(np.isfinite(center)):
            return self._capture_wait_or_fail("等待有效 FoundationPose 箱心")

        left, right = claw_pair
        midpoint = (left + right) * 0.5
        axis_x = np.array([right[0] - left[0], right[1] - left[1], 0.0], dtype=float)
        axis_norm = math.hypot(float(axis_x[0]), float(axis_x[1]))
        if axis_norm <= 1e-6:
            return self._capture_wait_or_fail("左右夹爪水平连线退化，无法建立抓取局部坐标")
        axis_x /= axis_norm
        axis_y = np.array([-axis_x[1], axis_x[0], 0.0], dtype=float)
        delta = center - midpoint
        offset_local = np.array(
            [float(np.dot(delta, axis_x)), float(np.dot(delta, axis_y)), float(delta[2])],
            dtype=float,
        )
        transform = {
            "frame": "claw_midpoint",
            "offset_local": {
                "x": float(offset_local[0]),
                "y": float(offset_local[1]),
                "z": float(offset_local[2]),
            },
            "captured_box_center_base": self._np_point_to_dict(center),
            "captured_claw_midpoint_base": self._np_point_to_dict(midpoint),
            "captured_left_claw_base": self._np_point_to_dict(left),
            "captured_right_claw_base": self._np_point_to_dict(right),
            "captured_axis_x_base": self._np_point_to_dict(axis_x),
            "stamp_monotonic": time.monotonic(),
            "source": "foundationpose_box_center",
        }
        self.blackboard.set(self.held_box_transform_key, transform, overwrite=True)
        self._capture_deadline = None
        self.ros_node.get_logger().info(
            f"\033[1;97;46m[{self.config_label}] 已锁定手中箱体抓取偏移: "
            f"FP箱心(base)=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}), "
            f"双爪中点(base)=({midpoint[0]:.3f},{midpoint[1]:.3f},{midpoint[2]:.3f}), "
            f"局部偏移=({offset_local[0]:+.3f},{offset_local[1]:+.3f},{offset_local[2]:+.3f}), "
            f"key={self.held_box_transform_key}\033[0m"
        )
        return Status.SUCCESS

    def _capture_wait_or_fail(self, reason):
        if time.monotonic() <= self._capture_deadline:
            self._log_throttled(f"锁定手中箱体抓取偏移中: {reason}")
            return Status.RUNNING
        self.ros_node.get_logger().error(
            f"[{self.config_label}] 锁定手中箱体抓取偏移超时: {reason}"
        )
        self._capture_deadline = None
        return Status.FAILURE

    def _on_timer(self):
        try:
            self._publish_held_box_visualization()
        except Exception as exc:  # noqa: BLE001 - timer 线程里必须兜住异常，不能打断行为树 tick。
            now = time.monotonic()
            if now - self._last_publish_error_at >= self.log_interval_sec:
                self._last_publish_error_at = now
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 手中箱体持续可视化刷新失败: {exc}"
                )

    def _publish_held_box_visualization(self):
        if self.visualization_pub is None:
            return

        claw_pair_base = self._get_current_claw_pair_base()
        current_pose = self.odom_transformer.get_current_pose()
        if claw_pair_base is None or current_pose is None:
            self._log_throttled(
                "等待手中箱体可视化输入: "
                f"claw_pair={'ok' if claw_pair_base is not None else 'missing'}, "
                f"odom={'ok' if current_pose is not None else 'missing'}"
            )
            return

        left_base, right_base = claw_pair_base
        robot_pose = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        map_z_offset = float(current_pose[2])

        left_map = self._base_point_to_map(left_base, robot_pose, map_z_offset)
        right_map = self._base_point_to_map(right_base, robot_pose, map_z_offset)
        center_map, yaw_deg = self._estimate_box_pose(left_map, right_map, robot_pose)
        left_base_point = self._np_point_to_dict(left_base)
        right_base_point = self._np_point_to_dict(right_base)
        center_base, yaw_base_deg = self._estimate_box_pose(
            left_base_point,
            right_base_point,
            Pose2D(x=0.0, y=0.0, yaw=self.fallback_yaw_offset_deg),
        )

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        marker_id = self._append_box(marker_array, marker_id, center_map, yaw_deg)
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "left_claw_actual",
            left_map,
            color=(1.0, 0.0, 1.0, 1.0),
            scale=0.075,
        )
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "right_claw_actual",
            right_map,
            color=(1.0, 0.55, 0.0, 1.0),
            scale=0.075,
        )
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "held_box_center",
            center_map,
            color=(0.0, 1.0, 0.25, 1.0),
            scale=0.08,
        )
        marker_id = self._append_line(
            marker_array,
            marker_id,
            "claw_connection",
            left_map,
            right_map,
            color=(1.0, 1.0, 0.0, 0.95),
            width=0.025,
        )
        marker_id = self._append_action_points(
            marker_array,
            marker_id,
            robot_pose,
            map_z_offset,
            left_map,
            right_map,
        )
        marker_id = self._append_text(marker_array, marker_id, center_map, yaw_deg, left_map, right_map)

        if self.base_link_visualization_enabled:
            marker_id = self._append_base_link_visualization(
                marker_array,
                marker_id,
                center_base,
                yaw_base_deg,
                left_base_point,
                right_base_point,
            )

        self.visualization_pub.publish(marker_array)

    def _get_current_claw_pair_base(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            return None

        left = self._current_claw_point_base(services.arm_controller, "left")
        right = self._current_claw_point_base(services.arm_controller, "right")
        if left is None or right is None:
            return None
        return left, right

    def _current_claw_point_base(self, arm_controller, side):
        if hasattr(arm_controller, "get_current_end_effector_pose"):
            current_pose = arm_controller.get_current_end_effector_pose(side)
        else:
            current_pose = None

        if current_pose is None:
            current_pose = (
                arm_controller.current_left_target
                if side == "left"
                else arm_controller.current_right_target
            )
        if current_pose is None or len(current_pose) != 6:
            return None

        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            return None

        translation, _ = transform
        rotation = ypr_to_rotation_matrix(current_pose[3:6])
        end_effector_point = np.array(current_pose[:3], dtype=float)
        return end_effector_point + rotation.dot(translation)

    def _base_point_to_map(self, point_base, robot_pose, map_z_offset):
        point_2d = transform_base_point_to_global(
            robot_pose,
            float(point_base[0]),
            float(point_base[1]),
        )
        return {
            "x": point_2d["x"],
            "y": point_2d["y"],
            "z": float(point_base[2]) + map_z_offset,
        }

    @staticmethod
    def _np_point_to_dict(point):
        return {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
        }

    def _estimate_box_pose(self, left_map, right_map, robot_pose):
        midpoint = {
            "x": 0.5 * (left_map["x"] + right_map["x"]),
            "y": 0.5 * (left_map["y"] + right_map["y"]),
            "z": 0.5 * (left_map["z"] + right_map["z"]),
        }

        dx = right_map["x"] - left_map["x"]
        dy = right_map["y"] - left_map["y"]
        norm = math.hypot(dx, dy)
        if math.isfinite(norm) and norm > 1e-6:
            x_axis = (dx / norm, dy / norm)
            yaw_deg = math.degrees(math.atan2(x_axis[1], x_axis[0]))
        else:
            yaw_deg = float(robot_pose.yaw) + self.fallback_yaw_offset_deg
            yaw_rad = math.radians(yaw_deg)
            x_axis = (math.cos(yaw_rad), math.sin(yaw_rad))

        y_axis = (-x_axis[1], x_axis[0])
        offset_x, offset_y, offset_z = self._active_center_offset()
        center = {
            "x": midpoint["x"]
            + offset_x * x_axis[0]
            + offset_y * y_axis[0],
            "y": midpoint["y"]
            + offset_x * x_axis[1]
            + offset_y * y_axis[1],
            "z": midpoint["z"] + offset_z,
        }
        return center, yaw_deg

    def _active_center_offset(self):
        """优先读取闭爪后锁定的局部偏移，缺失时兼容 JSON 固定偏移。"""
        if self.use_captured_transform and self.held_box_transform_key:
            try:
                raw = self.blackboard.get(self.held_box_transform_key)
                offset = raw.get("offset_local", {}) if isinstance(raw, dict) else {}
                values = (float(offset["x"]), float(offset["y"]), float(offset["z"]))
                if all(math.isfinite(value) for value in values):
                    return values
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        return (
            self.box_center_offset_x_m,
            self.box_center_offset_y_m,
            self.box_center_offset_z_m,
        )

    def _append_box(
        self,
        marker_array,
        marker_id,
        center,
        yaw_deg,
        frame_id=MAP_FRAME,
        namespace="held_box_estimate",
        alpha=0.35,
    ):
        marker = self._new_marker(marker_id, namespace, Marker.CUBE, frame_id=frame_id)
        marker.pose.position.x = center["x"]
        marker.pose.position.y = center["y"]
        marker.pose.position.z = center["z"]
        yaw_rad = math.radians(yaw_deg)
        marker.pose.orientation.z = math.sin(yaw_rad * 0.5)
        marker.pose.orientation.w = math.cos(yaw_rad * 0.5)
        marker.scale.x = self.box_size_x
        marker.scale.y = self.box_size_y
        marker.scale.z = self.box_size_z
        self._set_color(marker, 0.1, 0.85, 1.0, alpha)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_sphere(
        self,
        marker_array,
        marker_id,
        namespace,
        point,
        color,
        scale,
        frame_id=MAP_FRAME,
    ):
        marker = self._new_marker(marker_id, namespace, Marker.SPHERE, frame_id=frame_id)
        marker.pose.position.x = point["x"]
        marker.pose.position.y = point["y"]
        marker.pose.position.z = point["z"]
        marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_line(
        self,
        marker_array,
        marker_id,
        namespace,
        start,
        end,
        color,
        width,
        frame_id=MAP_FRAME,
    ):
        marker = self._new_marker(marker_id, namespace, Marker.LINE_LIST, frame_id=frame_id)
        marker.scale.x = float(width)
        marker.points = [
            Point(x=start["x"], y=start["y"], z=start["z"]),
            Point(x=end["x"], y=end["y"], z=end["z"]),
        ]
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_action_points(self, marker_array, marker_id, robot_pose, map_z_offset, current_left, current_right):
        if not self.action_points_enabled:
            return marker_id

        action_points = self._read_action_points_map(robot_pose, map_z_offset)
        if not action_points:
            return marker_id

        color_by_stage = {
            "pre": (1.0, 0.95, 0.05, 1.0),
            "push": (0.0, 1.0, 0.25, 1.0),
            "lift": (0.7, 0.35, 1.0, 1.0),
        }
        scale_by_stage = {
            "pre": 0.065,
            "push": 0.075,
            "lift": 0.065,
        }

        pairs = {}
        for item in action_points:
            stage = item["stage"]
            side = item["side"]
            point = item["point"]
            marker_id = self._append_sphere(
                marker_array,
                marker_id,
                f"action_point_{stage}_{side}",
                point,
                color=color_by_stage.get(stage, (1.0, 1.0, 1.0, 1.0)),
                scale=scale_by_stage.get(stage, 0.06),
            )
            marker_id = self._append_action_point_text(
                marker_array,
                marker_id,
                stage,
                side,
                point,
                color=color_by_stage.get(stage, (1.0, 1.0, 1.0, 1.0)),
            )
            pairs.setdefault(stage, {})[side] = point

            current = current_left if side == "left" else current_right
            marker_id = self._append_line(
                marker_array,
                marker_id,
                f"current_to_action_{stage}_{side}",
                current,
                point,
                color=(0.8, 0.8, 0.8, 0.45),
                width=0.012,
            )

        for stage, points in pairs.items():
            if "left" in points and "right" in points:
                marker_id = self._append_line(
                    marker_array,
                    marker_id,
                    f"action_pair_{stage}",
                    points["left"],
                    points["right"],
                    color=color_by_stage.get(stage, (1.0, 1.0, 1.0, 1.0)),
                    width=0.018,
                )
        return marker_id

    def _read_action_points_map(self, robot_pose, map_z_offset):
        key_specs = [
            ("pre", "left", self.pre_place_left_claw_point_key),
            ("pre", "right", self.pre_place_right_claw_point_key),
            ("push", "left", self.push_left_claw_point_key),
            ("push", "right", self.push_right_claw_point_key),
            ("lift", "left", self.lift_left_claw_point_key),
            ("lift", "right", self.lift_right_claw_point_key),
        ]
        points = []
        for stage, side, key in key_specs:
            point_base = self._read_base_point(key)
            if point_base is None:
                continue
            points.append(
                {
                    "stage": stage,
                    "side": side,
                    "key": key,
                    "point": self._base_point_to_map(point_base, robot_pose, map_z_offset),
                }
            )
        return points

    def _read_action_points_base(self):
        key_specs = [
            ("pre", "left", self.pre_place_left_claw_point_key),
            ("pre", "right", self.pre_place_right_claw_point_key),
            ("push", "left", self.push_left_claw_point_key),
            ("push", "right", self.push_right_claw_point_key),
            ("lift", "left", self.lift_left_claw_point_key),
            ("lift", "right", self.lift_right_claw_point_key),
        ]
        points = []
        for stage, side, key in key_specs:
            point_base = self._read_base_point(key)
            if point_base is None:
                continue
            points.append(
                {
                    "stage": stage,
                    "side": side,
                    "key": key,
                    "point": self._np_point_to_dict(point_base),
                }
            )
        return points

    def _read_base_point(self, key):
        if not key or not self.blackboard.exists(key):
            return None
        raw = self.blackboard.get(key)
        try:
            if isinstance(raw, dict):
                return np.array([float(raw["x"]), float(raw["y"]), float(raw["z"])], dtype=float)
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) >= 3:
                return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=float)
            if hasattr(raw, "x") and hasattr(raw, "y") and hasattr(raw, "z"):
                return np.array([float(raw.x), float(raw.y), float(raw.z)], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        return None

    def _append_action_point_text(
        self,
        marker_array,
        marker_id,
        stage,
        side,
        point,
        color,
        frame_id=MAP_FRAME,
        namespace_prefix="",
    ):
        marker = self._new_marker(
            marker_id,
            f"{namespace_prefix}action_point_text_{stage}_{side}",
            Marker.TEXT_VIEW_FACING,
            frame_id=frame_id,
        )
        marker.pose.position.x = point["x"]
        marker.pose.position.y = point["y"]
        marker.pose.position.z = point["z"] + 0.08
        marker.scale.z = 0.055
        marker.text = f"{stage}_{side}"
        self._set_color(marker, *color)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_text(self, marker_array, marker_id, center, yaw_deg, left, right):
        distance = math.sqrt(
            (right["x"] - left["x"]) ** 2
            + (right["y"] - left["y"]) ** 2
            + (right["z"] - left["z"]) ** 2
        )
        marker = self._new_marker(marker_id, "held_box_text", Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = center["x"]
        marker.pose.position.y = center["y"]
        marker.pose.position.z = center["z"] + 0.45
        marker.scale.z = 0.10
        marker.text = (
            "HELD BOX ESTIMATE\n"
            f"center=({center['x']:.2f},{center['y']:.2f},{center['z']:.2f}) yaw={yaw_deg:.1f}\n"
            f"claw_dist={distance:.3f} box=({self.box_size_x:.2f},{self.box_size_y:.2f},{self.box_size_z:.2f})\n"
            f"topic={self.visualization_topic}"
        )
        self._set_color(marker, 1.0, 1.0, 1.0, 1.0)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _append_base_link_visualization(self, marker_array, marker_id, center_base, yaw_base_deg, left_base, right_base):
        """在同一话题中追加 base_link 下的手中箱体估计和动作点。"""
        marker_id = self._append_box(
            marker_array,
            marker_id,
            center_base,
            yaw_base_deg,
            frame_id=BASE_LINK_FRAME,
            namespace="base_link/held_box_estimate",
            alpha=0.22,
        )
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "base_link/left_claw_actual",
            left_base,
            color=(1.0, 0.0, 1.0, 0.7),
            scale=0.06,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "base_link/right_claw_actual",
            right_base,
            color=(1.0, 0.55, 0.0, 0.7),
            scale=0.06,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_sphere(
            marker_array,
            marker_id,
            "base_link/held_box_center",
            center_base,
            color=(0.0, 1.0, 0.25, 0.75),
            scale=0.065,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_line(
            marker_array,
            marker_id,
            "base_link/claw_connection",
            left_base,
            right_base,
            color=(1.0, 1.0, 0.0, 0.7),
            width=0.018,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id = self._append_action_points_base(
            marker_array,
            marker_id,
            left_base,
            right_base,
        )
        return self._append_base_link_text(
            marker_array,
            marker_id,
            center_base,
            yaw_base_deg,
            left_base,
            right_base,
        )

    def _append_action_points_base(self, marker_array, marker_id, current_left, current_right):
        if not self.action_points_enabled:
            return marker_id

        action_points = self._read_action_points_base()
        if not action_points:
            return marker_id

        color_by_stage = {
            "pre": (1.0, 0.95, 0.05, 0.75),
            "push": (0.0, 1.0, 0.25, 0.75),
            "lift": (0.7, 0.35, 1.0, 0.75),
        }
        scale_by_stage = {
            "pre": 0.052,
            "push": 0.060,
            "lift": 0.052,
        }

        pairs = {}
        for item in action_points:
            stage = item["stage"]
            side = item["side"]
            point = item["point"]
            marker_id = self._append_sphere(
                marker_array,
                marker_id,
                f"base_link/action_point_{stage}_{side}",
                point,
                color=color_by_stage.get(stage, (1.0, 1.0, 1.0, 0.75)),
                scale=scale_by_stage.get(stage, 0.05),
                frame_id=BASE_LINK_FRAME,
            )
            marker_id = self._append_action_point_text(
                marker_array,
                marker_id,
                stage,
                side,
                point,
                color=color_by_stage.get(stage, (1.0, 1.0, 1.0, 0.75)),
                frame_id=BASE_LINK_FRAME,
                namespace_prefix="base_link/",
            )
            pairs.setdefault(stage, {})[side] = point

            current = current_left if side == "left" else current_right
            marker_id = self._append_line(
                marker_array,
                marker_id,
                f"base_link/current_to_action_{stage}_{side}",
                current,
                point,
                color=(0.8, 0.8, 0.8, 0.35),
                width=0.010,
                frame_id=BASE_LINK_FRAME,
            )

        for stage, points in pairs.items():
            if "left" in points and "right" in points:
                marker_id = self._append_line(
                    marker_array,
                    marker_id,
                    f"base_link/action_pair_{stage}",
                    points["left"],
                    points["right"],
                    color=color_by_stage.get(stage, (1.0, 1.0, 1.0, 0.75)),
                    width=0.014,
                    frame_id=BASE_LINK_FRAME,
                )
        return marker_id

    def _append_base_link_text(self, marker_array, marker_id, center, yaw_deg, left, right):
        distance = math.sqrt(
            (right["x"] - left["x"]) ** 2
            + (right["y"] - left["y"]) ** 2
            + (right["z"] - left["z"]) ** 2
        )
        marker = self._new_marker(
            marker_id,
            "base_link/held_box_text",
            Marker.TEXT_VIEW_FACING,
            frame_id=BASE_LINK_FRAME,
        )
        marker.pose.position.x = center["x"]
        marker.pose.position.y = center["y"]
        marker.pose.position.z = center["z"] + 0.35
        marker.scale.z = 0.075
        marker.text = (
            "HELD BOX ESTIMATE base_link\n"
            f"center=({center['x']:.2f},{center['y']:.2f},{center['z']:.2f}) yaw={yaw_deg:.1f}\n"
            f"claw_dist={distance:.3f} box=({self.box_size_x:.2f},{self.box_size_y:.2f},{self.box_size_z:.2f})"
        )
        self._set_color(marker, 0.75, 0.95, 1.0, 0.9)
        marker_array.markers.append(marker)
        return marker_id + 1

    def _publish_clear(self):
        if self.visualization_pub is None:
            return
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        self.visualization_pub.publish(marker_array)

    def _stop_existing_monitor(self):
        state = self._get_monitor_state()
        timer = state.get("timer") if state else None
        if timer is not None and not timer.is_canceled():
            timer.cancel()
        if self.monitor_state_key:
            self.blackboard.set(self.monitor_state_key, {}, overwrite=True)

    def _get_monitor_state(self):
        if not self.monitor_state_key or not self.blackboard.exists(self.monitor_state_key):
            return None
        state = self.blackboard.get(self.monitor_state_key)
        return state if isinstance(state, dict) else None

    def _log_throttled(self, message):
        now = time.monotonic()
        if now - self._last_log_at < self.log_interval_sec:
            return
        self._last_log_at = now
        self.ros_node.get_logger().warning(f"[{self.config_label}] {message}")

    def _new_marker(self, marker_id, namespace, marker_type, frame_id=MAP_FRAME):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.ros_node.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _set_color(marker, red, green, blue, alpha):
        marker.color.r = float(red)
        marker.color.g = float(green)
        marker.color.b = float(blue)
        marker.color.a = float(alpha)

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveBoxHeldBoxVisualizationMonitor start: "
            f"action={self.action}, topic={self.visualization_topic}, "
            f"interval={self.publish_interval_sec:.2f}s, services_key={self.services_key}, "
            f"monitor_key={self.monitor_state_key}"
        )
