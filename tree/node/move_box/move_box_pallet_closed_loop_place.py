"""FoundationPose 驱动的码垛全流程闭环放箱。

该节点接管“从双爪夹持到最终释放”期间所有会改变箱体位置的动作。它只依赖
FoundationPose 的箱心/朝向反馈，箱子尺寸、邻箱关系、预落位和推送方向仍由
``ComputeMoveBoxPalletPlaceStrategy`` 规划。

状态机：

* direct_place：双爪闭环到最终箱心上方 -> 闭环下降 -> 双爪释放；
* right_push_left_place / left_push_right_place：双爪闭环到预落位上方 ->
  闭环下降 -> 释放并抬起避让爪 -> 推送爪按 FP 每帧小步推送 -> 到最终箱心后释放。

任何阶段发生视觉超时、误差越界、推送越过目标或手臂执行失败，节点均返回
FAILURE；不会执行后续释放，从而避免在不确定状态下把箱子放到垛盘上。
"""

import math
import time

import numpy as np
import py_trees
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.base_event import EventStatus

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction
from .fp_grasp_visualization import _map_from_odom_message, _point_base_to_map


class MoveBoxPalletClosedLoopPlace(TimedMockAction):
    """在放置全过程中依据新 FP 帧持续校正箱体实际位置。"""

    allow_manual_result_override = False

    _STAGE_ALIGN_HOVER = "ALIGN_HOVER"
    _STAGE_ALIGN_XY_HIGH = "ALIGN_XY_HIGH"
    _STAGE_ALIGN_DROP = "ALIGN_DROP"
    _STAGE_LIFT_RELEASED = "LIFT_RELEASED"
    _STAGE_PUSH = "PUSH"

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.strategy_key = str(params.get("strategy_key", "move_box_pallet_place_strategy")).strip()
        self.final_box_pose_key = str(params.get("final_box_pose_key", "move_box_pallet_place_final_box_pose")).strip()
        self.pre_box_pose_key = str(params.get("pre_box_pose_key", "move_box_pallet_place_pre_box_pose")).strip()
        self.push_direction_key = str(params.get("push_direction_key", "move_box_pallet_place_push_direction")).strip()
        self.release_first_side_key = str(
            params.get("release_first_side_key", "move_box_pallet_place_release_first_side")
        ).strip()
        self.push_side_key = str(params.get("push_side_key", "move_box_pallet_place_push_side")).strip()
        self.result_key = str(params.get("result_key", "move_box_pallet_closed_loop_result")).strip()

        self.hover_height_m = max(float(params.get("hover_height_m", 0.03)), 0.0)
        self.planar_tolerance_m = max(float(params.get("planar_tolerance_m", 0.015)), 0.0)
        self.z_tolerance_m = max(float(params.get("z_tolerance_m", 0.010)), 0.0)
        self.push_tolerance_m = max(float(params.get("push_tolerance_m", 0.012)), 0.0)
        self.push_cross_track_tolerance_m = max(
            float(params.get("push_cross_track_tolerance_m", 0.020)),
            self.planar_tolerance_m,
        )
        self.max_step_xy_m = max(float(params.get("max_step_xy_m", 0.010)), 0.001)
        self.max_step_z_m = max(float(params.get("max_step_z_m", 0.008)), 0.001)
        self.max_push_step_m = max(float(params.get("max_push_step_m", 0.008)), 0.001)
        self.max_push_travel_m = max(float(params.get("max_push_travel_m", 0.09)), self.max_push_step_m)
        self.drop_planar_guard_m = max(
            float(params.get("drop_planar_guard_m", 0.020)), self.planar_tolerance_m
        )
        self.max_initial_planar_error_m = max(
            float(params.get("max_initial_planar_error_m", 0.10)), self.planar_tolerance_m
        )
        self.max_initial_z_error_m = max(
            float(params.get("max_initial_z_error_m", 0.25)), self.z_tolerance_m
        )
        self.max_align_iterations = max(int(params.get("max_align_iterations", 12)), 1)
        self.max_drop_iterations = max(int(params.get("max_drop_iterations", 12)), 1)
        self.max_push_iterations = max(int(params.get("max_push_iterations", 16)), 1)
        self.release_lift_clearance_m = max(float(params.get("release_lift_clearance_m", 0.18)), 0.0)
        self.frame_timeout_sec = max(float(params.get("frame_timeout_sec", 3.0)), 0.1)
        self.poll_interval_sec = max(float(params.get("poll_interval_sec", 0.15)), 0.03)
        self.post_motion_settle_sec = max(float(params.get("post_motion_settle_sec", 0.25)), 0.0)
        self.open_torque = float(params.get("open_torque", 0.0))
        self.claw_ypr = self._parse_ypr(params.get("claw_ypr", [0.0, -60.0, 0.0]))
        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/pallet_closed_loop_markers")
        ).strip()

        for key in (
            self.services_key,
            self.strategy_key,
            self.final_box_pose_key,
            self.pre_box_pose_key,
            self.push_direction_key,
            self.release_first_side_key,
            self.push_side_key,
        ):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)
        if self.result_key:
            self.blackboard.register_key(key=self.result_key, access=py_trees.common.Access.WRITE)

        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.visualization_pub = None
        if self.visualization_enabled and self.visualization_topic:
            self.visualization_pub = self.ros_node.create_publisher(
                self.visualization_topic, MarkerArray, queue_size=1, latch=True
            )

        self.services = None
        self.arm_controller = None
        self.strategy = ""
        self.final_goal = None
        self.pre_goal = None
        self.push_axis_map = None
        self.release_side = None
        self.push_side = None
        self.stage = None
        self.phase = "IDLE"
        self.stage_iterations = 0
        self.safe_align_z = None
        self.push_travel_m = 0.0
        self._deadline = 0.0
        self._settle_deadline = 0.0
        self._next_poll_at = 0.0
        self._pending_after_arm = None
        self._last_result = None

    def initialise(self):
        super().initialise()
        self.services = None
        self.arm_controller = None
        self.strategy = ""
        self.final_goal = None
        self.pre_goal = None
        self.push_axis_map = None
        self.release_side = None
        self.push_side = None
        self.stage = None
        self.phase = "SETUP"
        self.stage_iterations = 0
        self.safe_align_z = None
        self.push_travel_m = 0.0
        self._deadline = 0.0
        self._settle_deadline = 0.0
        self._next_poll_at = 0.0
        self._pending_after_arm = None
        self._last_result = None

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        if self.phase == "SETUP":
            return self._setup()
        if self.phase == "WAIT_FP":
            return self._wait_fresh_fp()
        if self.phase == "WAIT_SETTLE":
            return self._wait_settle()
        if self.phase == "WAIT_ARM":
            return self._wait_arm()
        return self._fail(f"未知闭环放箱阶段: phase={self.phase}, stage={self.stage}")

    def terminate(self, new_status):
        if self.phase == "WAIT_ARM" and self.arm_controller is not None and new_status != Status.SUCCESS:
            self.arm_controller.stop_arm_event()
        super().terminate(new_status)

    def _setup(self):
        self.services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        self.arm_controller = getattr(self.services, "arm_controller", None) if self.services else None
        detector = getattr(self.services, "box_detector", None) if self.services else None
        if self.arm_controller is None or detector is None:
            return self._fail("robot services 缺少 arm_controller 或 box_detector")
        self.strategy = self._read_text(self.strategy_key)
        self.final_goal = self._read_pose(self.final_box_pose_key)
        self.pre_goal = self._read_pose(self.pre_box_pose_key)
        if self.final_goal is None or self.pre_goal is None:
            return self._fail("缺少 final_box_pose 或 pre_box_pose")
        if self.strategy not in ("direct_place", "right_push_left_place", "left_push_right_place"):
            return self._fail(f"不支持的码垛策略: {self.strategy!r}")
        if self.strategy != "direct_place":
            self.push_axis_map = self._read_push_axis()
            self.release_side = self._read_side(self.release_first_side_key)
            self.push_side = self._read_side(self.push_side_key)
            if self.push_axis_map is None or self.release_side is None or self.push_side is None:
                return self._fail("推箱策略缺少 push_direction / release_side / push_side")
            if self.release_side == self.push_side:
                return self._fail("推送爪与释放避让爪不能相同")
        self.stage = self._STAGE_ALIGN_XY_HIGH
        self.stage_iterations = 0
        self._start_wait_new_frame("闭环放箱开始")
        self.ros_node.get_logger().info(
            f"\033[1;97;46m[{self.config_label}] 码垛全流程闭环开始: strategy={self.strategy}, "
            f"pre=({self.pre_goal[0]:.3f},{self.pre_goal[1]:.3f},{self.pre_goal[2]:.3f}), "
            f"final=({self.final_goal[0]:.3f},{self.final_goal[1]:.3f},{self.final_goal[2]:.3f})\033[0m"
        )
        return Status.RUNNING

    def _start_wait_new_frame(self, reason, settle_after_motion=False):
        if settle_after_motion and self.post_motion_settle_sec > 0.0:
            self._settle_deadline = time.monotonic() + self.post_motion_settle_sec
            self.phase = "WAIT_SETTLE"
            self._settle_reason = str(reason)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 手臂到位后等待相机/TF稳定: "
                f"wait={self.post_motion_settle_sec:.3f}s, stage={self.stage}"
            )
            return
        self._begin_wait_new_frame(reason)

    def _begin_wait_new_frame(self, reason):
        clear_cache = getattr(self.services.box_detector, "clear_latest_detection_cache", None)
        if callable(clear_cache):
            clear_cache()
        self._deadline = time.monotonic() + self.frame_timeout_sec
        self._next_poll_at = 0.0
        self.phase = "WAIT_FP"
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 等待闭环新 FP 帧: stage={self.stage}, reason={reason}, "
            f"iteration={self.stage_iterations}"
        )

    def _wait_settle(self):
        if time.monotonic() < self._settle_deadline:
            return Status.RUNNING
        self._settle_deadline = 0.0
        self._begin_wait_new_frame(f"相机稳定后：{self._settle_reason}")
        return Status.RUNNING

    def _wait_fresh_fp(self):
        now = time.monotonic()
        if now > self._deadline:
            return self._fail(f"等待闭环 FP 新帧超时: stage={self.stage}, timeout={self.frame_timeout_sec:.1f}s")
        if now < self._next_poll_at:
            return Status.RUNNING
        self._next_poll_at = now + self.poll_interval_sec
        detector = self.services.box_detector
        if not detector.update_latest_grasp_pose(
            self.arm_controller.get_initial_left_ypr(), self.arm_controller.get_initial_right_ypr()
        ):
            return Status.RUNNING
        actual = self._read_fp_center_map(detector)
        if actual is None:
            return Status.RUNNING
        if self.stage == self._STAGE_PUSH:
            return self._process_push(actual)
        return self._process_both_claw_stage(actual)

    def _process_both_claw_stage(self, actual):
        if self.stage == self._STAGE_ALIGN_XY_HIGH and self.safe_align_z is None:
            # 先锁住首帧的安全高度，只在当前高度做平面对齐；避免尚未对准垛盘槽位
            # 就开始下降。
            self.safe_align_z = float(actual[2])
        goal = self._stage_goal()
        error = goal - actual
        planar = math.hypot(float(error[0]), float(error[1]))
        z_error = abs(float(error[2]))
        self._record_and_publish(actual, goal, error, planar, z_error)
        if self.stage == self._STAGE_ALIGN_XY_HIGH and self.stage_iterations == 0 and (
            planar > self.max_initial_planar_error_m
        ):
            return self._fail(
                f"初始 FP 平面误差过大: planar={planar:.3f}m，保持夹持不释放"
            )
        if self.stage == self._STAGE_ALIGN_HOVER and self.stage_iterations == 0 and z_error > self.max_initial_z_error_m:
            return self._fail(
                f"安全高度到悬停位的初始高度误差过大: z={z_error:.3f}m，保持夹持不释放"
            )
        if planar <= self.planar_tolerance_m and z_error <= self.z_tolerance_m:
            return self._advance_after_both_stage()
        limit = self.max_align_iterations if self.stage in (
            self._STAGE_ALIGN_XY_HIGH,
            self._STAGE_ALIGN_HOVER,
        ) else self.max_drop_iterations
        if self.stage_iterations >= limit:
            return self._fail(
                f"闭环未收敛: stage={self.stage}, planar={planar:.3f}m, z={z_error:.3f}m, iterations={limit}"
            )
        step = self._limit_both_step(error)
        if self.stage == self._STAGE_ALIGN_DROP and planar > self.drop_planar_guard_m:
            # 已接近垛盘时禁止一边带着明显平面误差一边下降，先恢复对齐再下行。
            step[2] = 0.0
        if not self._start_both_arm_step(step):
            return self._fail("启动双爪闭环小步失败")
        self._pending_after_arm = "REMEASURE_SAME_STAGE"
        self.phase = "WAIT_ARM"
        return Status.RUNNING

    def _advance_after_both_stage(self):
        if self.stage == self._STAGE_ALIGN_XY_HIGH:
            self.stage = self._STAGE_ALIGN_HOVER
            self.stage_iterations = 0
            self._start_wait_new_frame("安全高度平面对齐，开始闭环下降到悬停位")
            return Status.RUNNING
        if self.stage == self._STAGE_ALIGN_HOVER:
            self.stage = self._STAGE_ALIGN_DROP
            self.stage_iterations = 0
            self._start_wait_new_frame("悬停位收敛，开始闭环下降")
            return Status.RUNNING
        if self.strategy == "direct_place":
            if not self._open_claw("both"):
                return self._fail("直接放箱双爪释放失败")
            return self._succeed("直接放箱已闭环对齐并释放")
        if not self._open_claw(self.release_side):
            return self._fail(f"推箱前张开{self.release_side}爪失败")
        if not self._start_lift_released_claw():
            return self._fail(f"推箱前上抬{self.release_side}爪失败")
        self.stage = self._STAGE_LIFT_RELEASED
        self._pending_after_arm = "START_PUSH"
        self.phase = "WAIT_ARM"
        return Status.RUNNING

    def _process_push(self, actual):
        error = self.final_goal - actual
        along = float(np.dot(error[:2], self.push_axis_map[:2]))
        cross = error[:2] - self.push_axis_map[:2] * along
        cross_error = math.hypot(float(cross[0]), float(cross[1]))
        z_error = abs(float(error[2]))
        planar = math.hypot(float(error[0]), float(error[1]))
        self._record_and_publish(actual, self.final_goal, error, planar, z_error, along, cross_error)
        if cross_error > self.push_cross_track_tolerance_m or z_error > self.z_tolerance_m:
            return self._fail(
                f"推送前箱体偏离安全通道: cross={cross_error:.3f}m, z={z_error:.3f}m；拒绝横向硬推"
            )
        if along < -self.push_tolerance_m:
            return self._fail(f"推送已越过最终目标: along_error={along:.3f}m")
        if along <= self.push_tolerance_m:
            if not self._open_claw(self.push_side):
                return self._fail(f"推送到位后张开{self.push_side}爪失败")
            return self._succeed(
                f"推箱闭环收敛并释放: along={along:.3f}m, cross={cross_error:.3f}m, z={z_error:.3f}m"
            )
        if self.stage_iterations >= self.max_push_iterations:
            return self._fail(f"推箱闭环迭代超限: along_error={along:.3f}m")
        push_step = min(along, self.max_push_step_m)
        if self.push_travel_m + push_step > self.max_push_travel_m:
            return self._fail(
                f"推送累计行程超出安全上限: travel={self.push_travel_m:.3f}m, "
                f"next_step={push_step:.3f}m, limit={self.max_push_travel_m:.3f}m"
            )
        step_map = self.push_axis_map * push_step
        if not self._start_push_arm_step(step_map):
            return self._fail(f"启动{self.push_side}爪闭环推送失败")
        self.push_travel_m += push_step
        self._pending_after_arm = "REMEASURE_SAME_STAGE"
        self.phase = "WAIT_ARM"
        return Status.RUNNING

    def _wait_arm(self):
        status = self.arm_controller.get_arm_event_status()
        if status == EventStatus.RUNNING:
            return Status.RUNNING
        if status != EventStatus.SUCCESS:
            return self._fail(f"闭环手臂动作失败: stage={self.stage}, status={status}")
        if self._pending_after_arm == "START_PUSH":
            self.stage = self._STAGE_PUSH
            self.stage_iterations = 0
            self.push_travel_m = 0.0
            self._start_wait_new_frame(
                "避让爪上抬完成，开始单爪闭环推送", settle_after_motion=True
            )
            return Status.RUNNING
        self.stage_iterations += 1
        self._start_wait_new_frame("手臂小步完成后复测", settle_after_motion=True)
        return Status.RUNNING

    def _stage_goal(self):
        goal = self.final_goal if self.strategy == "direct_place" else self.pre_goal
        if self.stage == self._STAGE_ALIGN_XY_HIGH:
            return np.array([goal[0], goal[1], self.safe_align_z], dtype=float)
        if self.stage == self._STAGE_ALIGN_HOVER:
            return goal + np.array([0.0, 0.0, self.hover_height_m], dtype=float)
        return np.array(goal, dtype=float)

    def _limit_both_step(self, error):
        step = np.array(error, dtype=float)
        planar = math.hypot(float(step[0]), float(step[1]))
        if planar > self.max_step_xy_m:
            step[:2] *= self.max_step_xy_m / planar
        step[2] = max(-self.max_step_z_m, min(self.max_step_z_m, step[2]))
        return step

    def _start_both_arm_step(self, step_map):
        claw_pair = self._get_current_claw_pair()
        current_pose = self.odom_transformer.get_current_pose()
        if claw_pair is None or current_pose is None:
            return False
        step_base = self._map_vector_to_base(step_map, current_pose)
        left_eef = self._claw_to_eef(claw_pair[0] + step_base, "left")
        right_eef = self._claw_to_eef(claw_pair[1] + step_base, "right")
        if left_eef is None or right_eef is None:
            return False
        self._log_step("双爪", step_map, step_base)
        return bool(self.arm_controller.start_arm_event(left_eef, right_eef, pose_frame=BASE_LINK_FRAME))

    def _start_lift_released_claw(self):
        released_pose = self.arm_controller.get_current_end_effector_pose(self.release_side)
        held_pose = self.arm_controller.get_current_end_effector_pose(self.push_side)
        if released_pose is None or held_pose is None:
            return False
        released = list(released_pose)
        released[2] += self.release_lift_clearance_m
        released[3:6] = self.claw_ypr
        held = list(held_pose)
        held[3:6] = self.claw_ypr
        if self.release_side == "left":
            left, right, locked = released, held, "right"
        else:
            left, right, locked = held, released, "left"
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 推箱闭环：已释放{self.release_side}爪，"
            f"上抬避让 {self.release_lift_clearance_m:.3f}m"
        )
        return bool(self.arm_controller.start_arm_event(left, right, locked_arm_side=locked, pose_frame=BASE_LINK_FRAME))

    def _start_push_arm_step(self, step_map):
        current_pose = self.odom_transformer.get_current_pose()
        pusher_claw = self._current_claw_point(self.push_side)
        other_side = "left" if self.push_side == "right" else "right"
        other_pose = self.arm_controller.get_current_end_effector_pose(other_side)
        if current_pose is None or pusher_claw is None or other_pose is None:
            return False
        step_base = self._map_vector_to_base(step_map, current_pose)
        pusher_eef = self._claw_to_eef(pusher_claw + step_base, self.push_side)
        if pusher_eef is None:
            return False
        other = list(other_pose)
        other[3:6] = self.claw_ypr
        if self.push_side == "left":
            left, right, locked = pusher_eef, other, "right"
        else:
            left, right, locked = other, pusher_eef, "left"
        self._log_step(f"{self.push_side}爪推送", step_map, step_base)
        return bool(self.arm_controller.start_arm_event(left, right, locked_arm_side=locked, pose_frame=BASE_LINK_FRAME))

    def _claw_to_eef(self, claw, side):
        eef = self.arm_controller.claw_point_to_end_effector_point(claw, side, self.claw_ypr)
        if eef is None:
            return None
        return [float(eef[0]), float(eef[1]), float(eef[2]), *self.claw_ypr]

    def _open_claw(self, side):
        try:
            return bool(self.arm_controller.open_claw(side, effort=self.open_torque))
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] ArmController 不支持显式张爪力矩，按默认力矩执行: {exc}"
            )
            return bool(self.arm_controller.open_claw(side))

    def _read_fp_center_map(self, detector):
        center_base = detector.get_latest_box_center()
        if center_base is None:
            return None
        try:
            center_base = np.array(center_base, dtype=float)
        except (TypeError, ValueError):
            return None
        if center_base.shape != (3,) or not np.all(np.isfinite(center_base)):
            return None
        odom = self.odom_transformer.get_latest_odom()
        if odom is None:
            return None
        return np.array(_point_base_to_map(center_base, _map_from_odom_message(odom)), dtype=float)

    def _get_current_claw_pair(self):
        left = self._current_claw_point("left")
        right = self._current_claw_point("right")
        if left is None or right is None:
            return None
        return left, right

    def _current_claw_point(self, side):
        pose = self.arm_controller.get_current_end_effector_pose(side)
        if pose is None or len(pose) != 6:
            return None
        transform = self.arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            return None
        translation, _ = transform
        return np.array(pose[:3], dtype=float) + ypr_to_rotation_matrix(pose[3:6]).dot(translation)

    def _read_pose(self, key):
        if not self.blackboard.exists(key):
            return None
        raw = self.blackboard.get(key)
        try:
            return np.array([float(raw["x"]), float(raw["y"]), float(raw.get("z", 0.0))], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None

    def _read_push_axis(self):
        if not self.blackboard.exists(self.push_direction_key):
            return None
        raw = self.blackboard.get(self.push_direction_key)
        try:
            axis = np.array([float(raw["x"]), float(raw["y"]), 0.0], dtype=float)
        except (KeyError, TypeError, ValueError):
            return None
        norm = math.hypot(float(axis[0]), float(axis[1]))
        return axis / norm if norm > 1e-6 else None

    def _read_text(self, key):
        return str(self.blackboard.get(key)).strip() if self.blackboard.exists(key) else ""

    def _read_side(self, key):
        side = self._read_text(key).lower()
        return side if side in ("left", "right") else None

    @staticmethod
    def _map_vector_to_base(vector_map, current_pose):
        yaw = math.radians(float(current_pose[3]))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                cos_yaw * vector_map[0] + sin_yaw * vector_map[1],
                -sin_yaw * vector_map[0] + cos_yaw * vector_map[1],
                vector_map[2],
            ],
            dtype=float,
        )

    def _record_and_publish(self, actual, goal, error, planar, z_error, along=None, cross=None):
        self._last_result = {
            "strategy": self.strategy,
            "stage": self.stage,
            "iteration": self.stage_iterations,
            "goal_map": self._as_pose(goal),
            "actual_map": self._as_pose(actual),
            "error_map": self._as_pose(error),
            "planar_error_m": float(planar),
            "z_error_m": float(z_error),
            "push_along_error_m": None if along is None else float(along),
            "push_cross_error_m": None if cross is None else float(cross),
            "push_travel_m": float(self.push_travel_m),
        }
        self._publish_visualization(actual, goal, error)

    def _log_step(self, action, step_map, step_base):
        self.ros_node.get_logger().info(
            f"\033[1;97;44m[{self.config_label}] 闭环{action}小步: stage={self.stage}, "
            f"step_map=({step_map[0]:.3f},{step_map[1]:.3f},{step_map[2]:.3f}), "
            f"step_base=({step_base[0]:.3f},{step_base[1]:.3f},{step_base[2]:.3f})\033[0m"
        )

    def _succeed(self, message):
        result = dict(self._last_result or {})
        result.update({"converged": True, "message": message})
        self._store_result(result)
        self.ros_node.get_logger().info(f"\033[1;97;42m[{self.config_label}] {message}\033[0m")
        return Status.SUCCESS

    def _fail(self, reason):
        result = dict(self._last_result or {})
        result.update({"converged": False, "reason": str(reason), "strategy": self.strategy, "stage": self.stage})
        self._store_result(result)
        self.ros_node.get_logger().error(f"\033[1;97;41m[{self.config_label}] 码垛闭环拒绝释放: {reason}\033[0m")
        return Status.FAILURE

    def _store_result(self, result):
        if self.result_key:
            self.blackboard.set(self.result_key, result, overwrite=True)

    def _publish_visualization(self, actual, goal, error):
        if self.visualization_pub is None:
            return
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        markers.markers.extend(
            [
                self._sphere(1, "closed_loop_goal", goal, 0.075, (0.1, 1.0, 0.1, 1.0)),
                self._sphere(2, "closed_loop_actual", actual, 0.075, (1.0, 0.15, 0.15, 1.0)),
                self._line(3, "closed_loop_error", actual, goal, (1.0, 0.85, 0.0, 1.0)),
                self._text(4, actual),
            ]
        )
        self.visualization_pub.publish(markers)

    def _sphere(self, marker_id, namespace, position, size, color):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns, marker.id, marker.type, marker.action = namespace, marker_id, Marker.SPHERE, Marker.ADD
        marker.pose.position = Point(*[float(value) for value in position])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def _line(self, marker_id, namespace, start, end, color):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns, marker.id, marker.type, marker.action = namespace, marker_id, Marker.LINE_LIST, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.018
        marker.points = [Point(*start.tolist()), Point(*end.tolist())]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def _text(self, marker_id, position):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns, marker.id, marker.type, marker.action = "closed_loop_text", marker_id, Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position = Point(float(position[0]), float(position[1]), float(position[2]) + 0.25)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.09
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.text = (
            "PALLET CLOSED LOOP\n"
            f"strategy={self.strategy} stage={self.stage}\n"
            f"iter={self.stage_iterations}"
        )
        return marker

    @staticmethod
    def _parse_ypr(value):
        if isinstance(value, str):
            import ast
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("claw_ypr 必须是 [yaw, pitch, roll]")
        return [float(item) for item in value]

    @staticmethod
    def _as_pose(value):
        return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveBoxPalletClosedLoopPlace start: hover={self.hover_height_m:.3f}, "
            f"tol=({self.planar_tolerance_m:.3f},{self.z_tolerance_m:.3f}), "
            f"step=({self.max_step_xy_m:.3f},{self.max_step_z_m:.3f}), "
            f"push_step={self.max_push_step_m:.3f}"
        )
