"""使用 common ArmController 控制双臂。

坐标系约定：
- `base_link` 是机器人机体/底盘基准坐标系；其实体原点由 URDF/TF 定义，
  不在行为树节点中定义。实机调试时可用 `rosrun tf tf_echo base_link waist_yaw_link`
  或 `rosrun tf tf_echo base_link left_claw` 确认。
- `waist_yaw_link` 是腰部 yaw 关节相关坐标系，ArmController 的初始化手臂位姿
  默认按该坐标系填写。
- 本节点内部会通过 ArmController 将 `base_link` 目标转换到 `waist_yaw_link`
  后交给底层手臂事件和 IK。
- `target_type=claw_point` 当前仅支持 `pose_frame=base_link`，用于兼容
  夹爪空间点目标的控制语义。

数据来源优先级：
1. JSON `left_pose/right_pose`，直接下发左右 eef 末端完整位姿。
2. blackboard `left_pose_key/right_pose_key`，读取前置计算节点写入的 eef 完整位姿。
3. `target_type=claw_point` 时读取 `left_point_key/right_point_key`，把夹爪目标点反算成 eef 位姿。
4. 默认使用 ArmController 初始化位姿，避免缺参时下发危险目标。

单臂模式：
- `side=left/right` 时只解析对应侧目标，并锁住另一侧手臂。
- 单臂夹爪点可使用通用 `point_key`，也兼容 `left_point_key/right_point_key`。
- 不写 `side` 时默认 `both`，保持原有双臂行为不变。
"""

import ast
import math

import numpy as np

import py_trees
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.base_event import EventStatus

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY, WAIST_YAW_LINK_FRAME
from ..base import TimedMockAction


class ArmsToPose(TimedMockAction):
    """读取左右臂目标，并复用共享 services 中的 common ArmController。

    JSON 参数：
    - left_pose/right_pose: [x, y, z, yaw, pitch, roll]
    - left_pose_key/right_pose_key: blackboard 中的 eef 完整位姿 key
    - side_key: blackboard 中的动态手臂侧别，存在时覆盖 side
    - pose_key: 动态选中手臂共用的 eef 位姿 key
    - target_type: eef_pose / claw_point
    - left_point_key/right_point_key: blackboard 中的夹爪目标点 key
    - claw_ypr: claw_point 模式下夹取姿态 [yaw, pitch, roll]，单位 deg
    - lock_arm_side: 锁住当前关节的手臂，支持 left / right / 空
    - pose_frame: 目标位姿坐标系，支持 base_link / waist_yaw_link。
      注意：claw_point 模式目前只能使用 base_link。

    eef_pose 默认使用 waist_yaw_link，是为了和 ArmController.move_to_initial_pose()
    使用的初始化手臂位姿坐标系保持一致；point_key 会隐式选择
    target_type=claw_point 和 pose_frame=base_link。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.left_pose = self._parse_pose(params.get("left_pose", None), "left_pose")
        self.right_pose = self._parse_pose(params.get("right_pose", None), "right_pose")
        self.left_pose_key = str(params.get("left_pose_key", "")).strip()
        self.right_pose_key = str(params.get("right_pose_key", "")).strip()
        self.side = str(params.get("side", "both")).strip().lower()
        self.side_key = str(params.get("side_key", "")).strip()
        self.pose_key = str(params.get("pose_key", "")).strip()
        self.current_side = self.side
        self.point_key = str(params.get("point_key", "")).strip()
        default_target_type = "claw_point" if self.point_key else "eef_pose"
        self.target_type = str(params.get("target_type", default_target_type)).strip().lower()
        self.left_point_key = str(params.get("left_point_key", "")).strip()
        self.right_point_key = str(params.get("right_point_key", "")).strip()
        self.claw_ypr = self._parse_ypr(params.get("claw_ypr", None), "claw_ypr")
        self.lock_arm_side = str(params.get("lock_arm_side", "")).strip()
        default_pose_frame = (
            BASE_LINK_FRAME if self.target_type == "claw_point" else WAIST_YAW_LINK_FRAME
        )
        self.pose_frame = str(params.get("pose_frame", default_pose_frame)).strip()
        if self.lock_arm_side not in ("", "left", "right"):
            raise ValueError("lock_arm_side 仅支持 left、right 或空")
        self.claw_point_diagnostics_visualization_enabled = self._to_bool(
            params.get("claw_point_diagnostics_visualization_enabled", True)
        )
        self.claw_point_diagnostics_visualization_topic = str(
            params.get(
                "claw_point_diagnostics_visualization_topic",
                "/move_box/claw_point_diagnostics_markers",
            )
        ).strip()
        self.claw_point_diagnostics_base_link_visualization_enabled = self._to_bool(
            params.get("claw_point_diagnostics_base_link_visualization_enabled", True)
        )
        self.claw_point_diagnostics_color_log_enabled = self._to_bool(
            params.get("claw_point_diagnostics_color_log_enabled", True)
        )
        self.claw_point_diagnostics_warn_distance_m = float(
            params.get("claw_point_diagnostics_warn_distance_m", 0.03)
        )
        self.claw_point_diagnostics_error_distance_m = float(
            params.get("claw_point_diagnostics_error_distance_m", 0.08)
        )
        self.claw_wireframe_visualization_enabled = self._to_bool(
            params.get("claw_wireframe_visualization_enabled", True)
        )
        # U 型夹爪线框仅用于 RViz 调参示意：假设 left_claw/right_claw 原点是夹爪接触中心，
        # local +x 为夹指伸出方向，local +/-y 为两根夹指左右方向。
        self.claw_wireframe_finger_length = float(
            params.get("claw_wireframe_finger_length", 0.12)
        )
        self.claw_wireframe_opening_width = float(
            params.get("claw_wireframe_opening_width", 0.10)
        )
        self.claw_wireframe_back_offset = float(
            params.get("claw_wireframe_back_offset", 0.035)
        )
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.claw_point_diagnostics_visualization_pub = None
        if (
            self.claw_point_diagnostics_visualization_enabled
            and self.claw_point_diagnostics_visualization_topic
        ):
            self.claw_point_diagnostics_visualization_pub = self.ros_node.create_publisher(
                self.claw_point_diagnostics_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.arm_controller = None
        self.services = None
        self.started = False
        self.skipped = False
        self.startup_error = None
        self._claw_point_diagnostics = {}
        self._diagnostics_logged = False
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        for key in [
            self.left_pose_key,
            self.right_pose_key,
            self.side_key,
            self.pose_key,
            self.point_key,
            self.left_point_key,
            self.right_point_key,
        ]:
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def initialise(self):
        """解析目标并启动非阻塞手臂事件。"""
        super().initialise()
        self.arm_controller = None
        self.services = None
        self.started = False
        self.skipped = False
        self.startup_error = None
        self._claw_point_diagnostics = {}
        self._diagnostics_logged = False

        if self.should_use_mock_execution():
            return

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            self.startup_error = RuntimeError(
                f"services 或 arm_controller 缺失: key={self.services_key}"
            )
            self.ros_node.get_logger().error(f"[{self.config_label}] {self.startup_error}")
            return
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            self.skipped = True
            return

        try:
            self.services = services
            self.arm_controller = services.arm_controller
            self.current_side = self._resolve_side()
            resolved = self._resolve_targets(self.arm_controller)
            if resolved is None:
                self.startup_error = RuntimeError("解析手臂目标失败")
                return
            left_target, right_target, target_source, automatic_locked_arm_side = resolved
            locked_arm_side = self.lock_arm_side or automatic_locked_arm_side
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 使用 common ArmController 启动手臂目标: "
                f"side={self.current_side}, locked={locked_arm_side}, "
                f"source={target_source}, frame={self.pose_frame}, "
                f"left={left_target}, right={right_target}"
            )
            self._log_execution_targets(left_target, right_target, target_source)

            # 关键步骤：只在 initialise 中启动一次手臂事件，后续 tick 只查询事件状态。
            self.arm_controller.reach_time = 0.0
            if not self.arm_controller.start_arm_event(
                left_target,
                right_target,
                locked_arm_side=locked_arm_side,
                pose_frame=self.pose_frame,
            ):
                self.startup_error = RuntimeError("启动手臂事件失败")
                return
        except Exception as exc:
            self.startup_error = exc
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 启动手臂目标失败: {exc}"
            )
            return
        self.started = True

    @staticmethod
    def _parse_pose(value, name):
        if value is None or (isinstance(value, str) and value == ""):
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError(f"{name} 必须是长度为 6 的列表: [x, y, z, yaw, pitch, roll]")
        return [float(item) for item in value]

    @staticmethod
    def _parse_point(value, name):
        if value is None or (isinstance(value, str) and value == ""):
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} 必须是长度为 3 的列表: [x, y, z]")
        return [float(item) for item in value]

    @staticmethod
    def _parse_ypr(value, name):
        if value is None or (isinstance(value, str) and value == ""):
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} 必须是长度为 3 的列表: [yaw, pitch, roll]")
        return [float(item) for item in value]

    def _get_blackboard_value(self, key, label):
        if not key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少 {label}")
            return None
        if not self.blackboard.exists(key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 缺少 {label}: key={key}"
            )
            return None
        return self.blackboard.get(key)

    @staticmethod
    def _format_pose(pose):
        return (
            f"x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}, "
            f"yaw={pose[3]:.2f}deg, pitch={pose[4]:.2f}deg, roll={pose[5]:.2f}deg"
        )

    def _log_execution_targets(self, left_target, right_target, target_source):
        left_key = self.left_pose_key or "json/default"
        right_key = self.right_pose_key or "json/default"
        # 关键步骤：这里打印的是即将传给 ArmController.start_arm_event 的最终 6 维目标。
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 真正下发双臂目标 pose: "
            f"source={target_source}, frame={self.pose_frame}, "
            f"left_key={left_key}, left=({self._format_pose(left_target)}), "
            f"right_key={right_key}, right=({self._format_pose(right_target)})"
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if self.skipped:
            return Status.SUCCESS
        if self.startup_error is not None:
            return Status.FAILURE
        if not self.started or self.arm_controller is None:
            return Status.FAILURE

        arm_status = self.arm_controller.get_arm_event_status()
        if arm_status == EventStatus.RUNNING:
            return Status.RUNNING
        if arm_status == EventStatus.SUCCESS:
            self._log_claw_point_finish_diagnostics()
            self.started = False
            return Status.SUCCESS

        self.started = False
        return Status.FAILURE

    def terminate(self, new_status):
        """节点被中断时关闭未完成的手臂事件。"""
        if (
            self.started
            and self.arm_controller is not None
            and new_status != Status.SUCCESS
        ):
            # 关键步骤：上层切走 RUNNING 节点时关闭事件，避免手臂后台事件残留。
            self.arm_controller.stop_arm_event()
        self.started = False
        super().terminate(new_status)

    def _resolve_targets(self, arm_controller):
        if self.current_side not in ("both", "left", "right"):
            raise ValueError("side 仅支持 both、left 或 right")
        if self.current_side != "both":
            return self._resolve_single_arm_targets(arm_controller)

        if self.left_pose is not None and self.right_pose is not None:
            return list(self.left_pose), list(self.right_pose), "json:eef_pose", None

        if self.left_pose_key and self.right_pose_key:
            left_value = self._get_blackboard_value(self.left_pose_key, "left_pose_key")
            right_value = self._get_blackboard_value(self.right_pose_key, "right_pose_key")
            if left_value is None or right_value is None:
                return None
            return (
                self._parse_pose(left_value, self.left_pose_key),
                self._parse_pose(right_value, self.right_pose_key),
                "blackboard:eef_pose",
                None,
            )

        if self.target_type == "claw_point":
            return self._resolve_claw_point_targets(arm_controller)

        if self.pose_frame == WAIST_YAW_LINK_FRAME:
            return (
                list(arm_controller.initial_left_pose_in_waist),
                list(arm_controller.initial_right_pose_in_waist),
                "default:initial_pose@waist_yaw_link",
                None,
            )
        if self.pose_frame == BASE_LINK_FRAME:
            if hasattr(arm_controller, "refresh_initial_pose_in_base_link"):
                arm_controller.refresh_initial_pose_in_base_link()
            return (
                list(arm_controller.initial_left_pose_in_baselink),
                list(arm_controller.initial_right_pose_in_baselink),
                "default:initial_pose@base_link",
                None,
            )

        raise ValueError("pose_frame 仅支持 base_link 或 waist_yaw_link")

    def _resolve_claw_point_targets(self, arm_controller):
        if self.pose_frame != BASE_LINK_FRAME:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] claw_point 模式当前仅支持 base_link，当前 pose_frame={self.pose_frame}"
            )
            return None

        left_value = self._get_blackboard_value(self.left_point_key, "left_point_key")
        right_value = self._get_blackboard_value(self.right_point_key, "right_point_key")
        if left_value is None or right_value is None:
            return None

        left_point = self._parse_point(left_value, self.left_point_key)
        right_point = self._parse_point(right_value, self.right_point_key)

        # 关键步骤：夹取姿态优先使用节点配置，避免 claw_point 模式误用回初始位姿的 pitch。
        if self.claw_ypr is not None:
            left_ypr = list(self.claw_ypr)
            right_ypr = list(self.claw_ypr)
            target_source = "blackboard:claw_point+json:claw_ypr"
        else:
            # 抓取姿态沿用控制器启动时缓存的 base_link 初始姿态。
            # 腰部运动后不能在这里重新刷新，否则姿态会随腰部变化，而解析 IK
            # 仍按固定手臂基座模型求解，最终形成明显的夹爪空间偏差。
            left_ypr = arm_controller.get_initial_left_ypr()
            right_ypr = arm_controller.get_initial_right_ypr()
            target_source = "blackboard:claw_point"

        # 关键步骤：用最终下发姿态反算 eef 位置，保证 position 与 yaw/pitch/roll 来自同一姿态。
        left_ee_point = arm_controller.claw_point_to_end_effector_point(left_point, "left", left_ypr)
        right_ee_point = arm_controller.claw_point_to_end_effector_point(right_point, "right", right_ypr)
        if left_ee_point is None or right_ee_point is None:
            return None
        self._store_claw_point_diagnostic_target("left", left_point, left_ee_point, left_ypr)
        self._store_claw_point_diagnostic_target("right", right_point, right_ee_point, right_ypr)

        left_target = [
            left_ee_point[0],
            left_ee_point[1],
            left_ee_point[2],
            left_ypr[0],
            left_ypr[1],
            left_ypr[2],
        ]
        right_target = [
            right_ee_point[0],
            right_ee_point[1],
            right_ee_point[2],
            right_ypr[0],
            right_ypr[1],
            right_ypr[2],
        ]
        return left_target, right_target, target_source, None

    def _resolve_single_arm_targets(self, arm_controller):
        """解析单侧目标，并使用当前缓存目标填充将被锁住的另一侧。"""
        moving_side = self.current_side
        locked_arm_side = "right" if moving_side == "left" else "left"
        moving_target, target_source = self._resolve_single_moving_target(
            arm_controller,
            moving_side,
        )
        if moving_target is None:
            return None

        if moving_side == "left":
            locked_target = arm_controller.convert_target_pose_frame(
                arm_controller.current_right_target,
                arm_controller.current_right_target_frame,
                self.pose_frame,
            )
            return moving_target, locked_target, target_source, locked_arm_side

        locked_target = arm_controller.convert_target_pose_frame(
            arm_controller.current_left_target,
            arm_controller.current_left_target_frame,
            self.pose_frame,
        )
        return locked_target, moving_target, target_source, locked_arm_side

    def _resolve_single_moving_target(self, arm_controller, side):
        """按 claw_point、JSON pose、blackboard pose、默认位姿解析移动侧目标。"""
        if self.target_type == "claw_point":
            if self.pose_frame != "base_link":
                raise ValueError("单臂 claw_point 模式当前仅支持 base_link")
            side_point_key = self.left_point_key if side == "left" else self.right_point_key
            point_key = self.point_key or side_point_key
            point_value = self._get_blackboard_value(point_key, "point_key")
            if point_value is None:
                return None, None
            point = self._parse_point(point_value, point_key)
            if self.claw_ypr is not None:
                ypr = list(self.claw_ypr)
                target_source = f"blackboard:claw_point:{point_key}+json:claw_ypr"
            else:
                ypr = (
                    arm_controller.get_initial_left_ypr()
                    if side == "left"
                    else arm_controller.get_initial_right_ypr()
                )
                target_source = f"blackboard:claw_point:{point_key}"

            # 新 ArmController 接口允许显式传入 claw_ypr。位置反算和最终下发
            # 必须共用这一姿态，否则工具长度会被按另一姿态旋转，形成固定空间偏差。
            ee_point = arm_controller.claw_point_to_end_effector_point(point, side, ypr)
            if ee_point is None:
                return None, None
            self._store_claw_point_diagnostic_target(side, point, ee_point, ypr)
            return [*ee_point, *ypr], target_source

        direct_pose = self.left_pose if side == "left" else self.right_pose
        if direct_pose is not None:
            return list(direct_pose), f"json:{side}_pose"

        # 关键步骤：pose_key 与 side_key 配合，让同一个抓取目标可以动态下发给选中手臂。
        pose_key = self.pose_key or (
            self.left_pose_key if side == "left" else self.right_pose_key
        )
        if pose_key:
            pose_value = self._get_blackboard_value(pose_key, f"{side}_pose_key")
            if pose_value is None:
                return None, None
            return self._parse_pose(pose_value, pose_key), f"blackboard:{pose_key}"

        if self.pose_frame == "waist_yaw_link":
            default_pose = (
                arm_controller.initial_left_pose_in_waist
                if side == "left"
                else arm_controller.initial_right_pose_in_waist
            )
        elif self.pose_frame == "base_link":
            if hasattr(arm_controller, "refresh_initial_pose_in_base_link"):
                arm_controller.refresh_initial_pose_in_base_link()
            default_pose = (
                arm_controller.initial_left_pose_in_baselink
                if side == "left"
                else arm_controller.initial_right_pose_in_baselink
            )
        else:
            raise ValueError("pose_frame 仅支持 base_link 或 waist_yaw_link")
        return list(default_pose), f"default:{side}_initial_pose@{self.pose_frame}"

    def _resolve_side(self):
        """解析本次执行手臂；配置 side_key 时优先使用 blackboard 中的选手结果。"""
        if not self.side_key:
            return self.side
        if not self.blackboard.exists(self.side_key):
            raise RuntimeError(f"手臂侧别不存在: key={self.side_key}")
        side = str(self.blackboard.get(self.side_key)).strip().lower()
        if side not in ("left", "right"):
            raise ValueError(
                f"side_key 仅支持 left 或 right: key={self.side_key}, value={side!r}"
            )
        return side

    def describe_start(self):
        if self.left_pose is not None or self.right_pose is not None:
            left_desc = self.left_pose
            right_desc = self.right_pose
        elif self.left_pose_key or self.right_pose_key:
            left_desc = f"blackboard:{self.left_pose_key}"
            right_desc = f"blackboard:{self.right_pose_key}"
        elif self.target_type == "claw_point":
            if self.current_side == "left":
                left_desc = f"claw_point:{self.point_key or self.left_point_key}"
                right_desc = "locked"
            elif self.current_side == "right":
                left_desc = "locked"
                right_desc = f"claw_point:{self.point_key or self.right_point_key}"
            else:
                left_desc = f"claw_point:{self.left_point_key}"
                right_desc = f"claw_point:{self.right_point_key}"
        else:
            left_desc = f"initial_left_pose@{self.pose_frame}"
            right_desc = f"initial_right_pose@{self.pose_frame}"
        return (
            f"[{self.config_label}] ArmsToPose start: "
            f"target_type={self.target_type}, frame={self.pose_frame}, "
            f"side_key={self.side_key or '<none>'}, pose_key={self.pose_key or '<none>'}, "
            f"lock_arm_side={self.lock_arm_side or 'none'}, "
            f"left={left_desc}, right={right_desc}"
        )

    def _store_claw_point_diagnostic_target(self, side, claw_point, eef_point, ypr):
        """缓存 claw_point 模式的计算目标，动作完成后与实际 TF 做差。"""
        self._claw_point_diagnostics[side] = {
            "target_claw": [float(value) for value in claw_point],
            "target_eef": [float(value) for value in eef_point],
            "target_ypr": [float(value) for value in ypr],
        }

    def _log_claw_point_finish_diagnostics(self):
        """动作完成后输出目标点与实际TF偏差，便于定位FP/外参/IK误差。"""
        if self._diagnostics_logged or not self._claw_point_diagnostics:
            return
        if self.arm_controller is None:
            return
        diagnostic_items = []
        for side, target in self._claw_point_diagnostics.items():
            diagnostic_item = self._log_single_claw_point_finish_diagnostic(side, target)
            if diagnostic_item is not None:
                diagnostic_items.append(diagnostic_item)
        self._publish_claw_point_diagnostics_visualization(diagnostic_items)
        self._diagnostics_logged = True

    def _log_single_claw_point_finish_diagnostic(self, side, target):
        actual_eef = self.arm_controller.get_current_end_effector_pose(
            side,
            target_frame=BASE_LINK_FRAME,
        )
        actual_claw = self._lookup_current_claw_pose(side)
        target_claw = target["target_claw"]
        target_eef = target["target_eef"]
        target_ypr = target["target_ypr"]
        diagnostic_item = {
            "side": side,
            "target_claw": [*target_claw, *target_ypr],
            "target_eef": [*target_eef, *target_ypr],
            "actual_claw": actual_claw,
            "actual_eef": actual_eef,
            "claw_delta": None,
            "eef_delta": None,
        }

        if actual_eef is not None:
            eef_delta = self._pose_delta(actual_eef, [*target_eef, *target_ypr])
            diagnostic_item["eef_delta"] = eef_delta
            self._log_colored_pose_delta(
                side=side,
                target_name="EEF",
                target_pose=[*target_eef, *target_ypr],
                actual_pose=actual_eef,
                delta=eef_delta,
            )
        else:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] {side} EEF目标-实际偏差无法计算: 当前EEF TF不可用"
            )

        if actual_claw is not None:
            claw_delta = self._pose_delta(actual_claw, [*target_claw, *target_ypr])
            diagnostic_item["claw_delta"] = claw_delta
            self._log_colored_pose_delta(
                side=side,
                target_name="claw",
                target_pose=[*target_claw, *target_ypr],
                actual_pose=actual_claw,
                delta=claw_delta,
            )
        else:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] {side} claw目标-实际偏差无法计算: 当前claw TF不可用"
            )
        return diagnostic_item

    def _log_colored_pose_delta(self, side, target_name, target_pose, actual_pose, delta):
        """打印夹爪/末端目标与实际执行偏差；终端支持 ANSI 时按误差距离着色。"""
        severity, color = self._diagnostic_log_style(delta["dist"])
        reset = "\033[0m" if self.claw_point_diagnostics_color_log_enabled else ""
        prefix = color if self.claw_point_diagnostics_color_log_enabled else ""
        message = (
            f"{prefix}★★★ [{self.config_label}] {side} {target_name}目标-实际偏差(base_link) "
            f"[{severity}] ★★★ "
            f"target=({target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}, "
            f"yaw={target_pose[3]:.2f}, pitch={target_pose[4]:.2f}, roll={target_pose[5]:.2f}), "
            f"actual=({actual_pose[0]:.3f}, {actual_pose[1]:.3f}, {actual_pose[2]:.3f}, "
            f"yaw={actual_pose[3]:.2f}, pitch={actual_pose[4]:.2f}, roll={actual_pose[5]:.2f}), "
            f"delta_xyz=({delta['dx']:.3f}, {delta['dy']:.3f}, {delta['dz']:.3f}), "
            f"dist={delta['dist']:.3f}m, "
            f"delta_ypr=({delta['dyaw']:.2f}, {delta['dpitch']:.2f}, {delta['droll']:.2f})deg"
            f"{reset}"
        )
        if delta["dist"] >= self.claw_point_diagnostics_error_distance_m:
            self.ros_node.get_logger().error(message)
        elif delta["dist"] >= self.claw_point_diagnostics_warn_distance_m:
            self.ros_node.get_logger().warning(message)
        else:
            self.ros_node.get_logger().info(message)

    def _diagnostic_log_style(self, distance_m):
        if distance_m >= self.claw_point_diagnostics_error_distance_m:
            return "ERROR", "\033[1;31m"
        if distance_m >= self.claw_point_diagnostics_warn_distance_m:
            return "WARN", "\033[1;33m"
        return "OK", "\033[1;36m"

    def _lookup_current_claw_pose(self, side):
        """查询当前 left_claw/right_claw 在 base_link 下的实际位姿。"""
        if self.arm_controller is None:
            return None
        child_frame = f"{side}_claw"
        try:
            self.arm_controller.tf_listener.waitForTransform(
                BASE_LINK_FRAME,
                child_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
            translation, quaternion = self.arm_controller.tf_listener.lookupTransform(
                BASE_LINK_FRAME,
                child_frame,
                rospy.Time(0),
            )
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 查询 {BASE_LINK_FRAME}->{child_frame} 当前TF失败: {exc}"
            )
            return None
        roll, pitch, yaw = tf_trans.euler_from_quaternion(quaternion)
        return [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        ]

    @staticmethod
    def _pose_delta(actual_pose, target_pose):
        dx = float(actual_pose[0]) - float(target_pose[0])
        dy = float(actual_pose[1]) - float(target_pose[1])
        dz = float(actual_pose[2]) - float(target_pose[2])
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dist": float(np.linalg.norm([dx, dy, dz])),
            "dyaw": ArmsToPose._angle_delta_deg(actual_pose[3], target_pose[3]),
            "dpitch": ArmsToPose._angle_delta_deg(actual_pose[4], target_pose[4]),
            "droll": ArmsToPose._angle_delta_deg(actual_pose[5], target_pose[5]),
        }

    @staticmethod
    def _angle_delta_deg(actual_deg, target_deg):
        return (float(actual_deg) - float(target_deg) + 180.0) % 360.0 - 180.0

    def _publish_claw_point_diagnostics_visualization(self, diagnostic_items):
        """把计算落点和实际落点转到 map 下显示，和 FP 抓取点可视化叠加排查。"""
        if self.claw_point_diagnostics_visualization_pub is None:
            return
        if not diagnostic_items:
            return

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        marker_id = self._append_fp_box_diagnostic_markers(marker_array, marker_id)
        for item in diagnostic_items:
            marker_id = self._append_single_claw_point_diagnostic_markers(
                marker_array,
                marker_id,
                item,
            )

        self.claw_point_diagnostics_visualization_pub.publish(marker_array)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已发布夹爪落点诊断RViz标记: "
            f"topic={self.claw_point_diagnostics_visualization_topic}, "
            f"items={len(diagnostic_items)}"
        )

    def _append_single_claw_point_diagnostic_markers(self, marker_array, marker_id, item):
        side = item["side"]
        target_claw_map = self._transform_pose_from_base_to_map(item["target_claw"])
        actual_claw_map = self._transform_pose_from_base_to_map(item["actual_claw"])
        target_eef_map = self._transform_pose_from_base_to_map(item["target_eef"])
        actual_eef_map = self._transform_pose_from_base_to_map(item["actual_eef"])

        if target_claw_map is not None:
            marker = self._new_diagnostic_marker(marker_id, f"{side}_target_claw", Marker.SPHERE)
            marker_id += 1
            marker.pose.position = self._point_message(*target_claw_map[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.055
            self._set_marker_color(marker, 0.05, 1.0, 0.15, 1.0)
            marker_array.markers.append(marker)

        if actual_claw_map is not None:
            marker = self._new_diagnostic_marker(marker_id, f"{side}_actual_claw", Marker.SPHERE)
            marker_id += 1
            marker.pose.position = self._point_message(*actual_claw_map[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.065
            self._set_marker_color(marker, 1.0, 0.15, 0.05, 1.0)
            marker_array.markers.append(marker)
            marker_id = self._append_claw_wireframe_marker(
                marker_array,
                marker_id,
                side,
                actual_claw_map,
                actual=True,
            )

        if target_eef_map is not None:
            marker = self._new_diagnostic_marker(marker_id, f"{side}_target_eef", Marker.SPHERE)
            marker_id += 1
            marker.pose.position = self._point_message(*target_eef_map[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.04
            self._set_marker_color(marker, 0.15, 0.75, 1.0, 0.9)
            marker_array.markers.append(marker)

        if actual_eef_map is not None:
            marker = self._new_diagnostic_marker(marker_id, f"{side}_actual_eef", Marker.SPHERE)
            marker_id += 1
            marker.pose.position = self._point_message(*actual_eef_map[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.045
            self._set_marker_color(marker, 1.0, 0.55, 0.05, 0.9)
            marker_array.markers.append(marker)

        if target_claw_map is not None and actual_claw_map is not None:
            marker = self._new_diagnostic_marker(marker_id, f"{side}_claw_error_line", Marker.LINE_LIST)
            marker_id += 1
            marker.scale.x = 0.018
            marker.points = [
                self._point_message(*target_claw_map[:3]),
                self._point_message(*actual_claw_map[:3]),
            ]
            self._set_marker_color(marker, 1.0, 0.9, 0.05, 1.0)
            marker_array.markers.append(marker)

        text_pose = actual_claw_map or target_claw_map or actual_eef_map or target_eef_map
        if text_pose is not None:
            claw_delta = item.get("claw_delta")
            eef_delta = item.get("eef_delta")
            text = self._new_diagnostic_marker(marker_id, f"{side}_diagnostic_text", Marker.TEXT_VIEW_FACING)
            marker_id += 1
            text.pose.position.x = text_pose[0]
            text.pose.position.y = text_pose[1]
            text.pose.position.z = text_pose[2] + 0.18
            text.pose.orientation.w = 1.0
            text.scale.z = 0.075
            self._set_marker_color(text, 1.0, 1.0, 1.0, 1.0)
            text.text = self._format_diagnostic_text(side, claw_delta, eef_delta)
            marker_array.markers.append(text)

        if self.claw_point_diagnostics_base_link_visualization_enabled:
            marker_id = self._append_single_claw_point_base_link_markers(
                marker_array,
                marker_id,
                item,
            )

        return marker_id

    def _append_single_claw_point_base_link_markers(self, marker_array, marker_id, item):
        """额外发布 base_link 下的同源夹爪诊断，便于和 TF/IK 输入直接对齐。"""
        side = item["side"]
        target_claw = item.get("target_claw")
        actual_claw = item.get("actual_claw")
        target_eef = item.get("target_eef")
        actual_eef = item.get("actual_eef")

        if target_claw is not None:
            marker = self._new_diagnostic_marker(
                marker_id,
                f"base_link/{side}_target_claw",
                Marker.SPHERE,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            marker.pose.position = self._point_message(*target_claw[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.045
            self._set_marker_color(marker, 0.05, 1.0, 0.15, 0.65)
            marker_array.markers.append(marker)

        if actual_claw is not None:
            marker = self._new_diagnostic_marker(
                marker_id,
                f"base_link/{side}_actual_claw",
                Marker.SPHERE,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            marker.pose.position = self._point_message(*actual_claw[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.055
            self._set_marker_color(marker, 1.0, 0.15, 0.05, 0.75)
            marker_array.markers.append(marker)
            marker_id = self._append_claw_wireframe_marker(
                marker_array,
                marker_id,
                side,
                actual_claw,
                actual=True,
                frame_id=BASE_LINK_FRAME,
                namespace_prefix="base_link/",
            )

        if target_eef is not None:
            marker = self._new_diagnostic_marker(
                marker_id,
                f"base_link/{side}_target_eef",
                Marker.SPHERE,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            marker.pose.position = self._point_message(*target_eef[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.035
            self._set_marker_color(marker, 0.15, 0.75, 1.0, 0.6)
            marker_array.markers.append(marker)

        if actual_eef is not None:
            marker = self._new_diagnostic_marker(
                marker_id,
                f"base_link/{side}_actual_eef",
                Marker.SPHERE,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            marker.pose.position = self._point_message(*actual_eef[:3])
            marker.scale.x = marker.scale.y = marker.scale.z = 0.04
            self._set_marker_color(marker, 1.0, 0.55, 0.05, 0.65)
            marker_array.markers.append(marker)

        if target_claw is not None and actual_claw is not None:
            marker = self._new_diagnostic_marker(
                marker_id,
                f"base_link/{side}_claw_error_line",
                Marker.LINE_LIST,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            marker.scale.x = 0.014
            marker.points = [
                self._point_message(*target_claw[:3]),
                self._point_message(*actual_claw[:3]),
            ]
            self._set_marker_color(marker, 1.0, 0.9, 0.05, 0.75)
            marker_array.markers.append(marker)

        text_pose = actual_claw or target_claw or actual_eef or target_eef
        if text_pose is not None:
            text = self._new_diagnostic_marker(
                marker_id,
                f"base_link/{side}_diagnostic_text",
                Marker.TEXT_VIEW_FACING,
                frame_id=BASE_LINK_FRAME,
            )
            marker_id += 1
            text.pose.position.x = float(text_pose[0])
            text.pose.position.y = float(text_pose[1])
            text.pose.position.z = float(text_pose[2] + 0.14)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.06
            self._set_marker_color(text, 0.75, 0.95, 1.0, 0.9)
            text.text = "BASE_LINK\n" + self._format_diagnostic_text(
                side,
                item.get("claw_delta"),
                item.get("eef_delta"),
            )
            marker_array.markers.append(text)

        return marker_id

    def _append_claw_wireframe_marker(
        self,
        marker_array,
        marker_id,
        side,
        claw_pose,
        actual,
        frame_id=MAP_FRAME,
        namespace_prefix="",
    ):
        """画一个跟随 left_claw/right_claw 位姿的简化 U 型夹爪线框。"""
        if not self.claw_wireframe_visualization_enabled or claw_pose is None:
            return marker_id

        half_width = max(self.claw_wireframe_opening_width * 0.5, 0.005)
        back_x = -abs(self.claw_wireframe_back_offset)
        tip_x = max(self.claw_wireframe_finger_length - abs(self.claw_wireframe_back_offset), 0.02)

        # U 型：后横梁 + 两根夹指。local 原点放在开口中心附近，便于观察 claw 原点与夹持中心关系。
        local_segments = (
            ([back_x, -half_width, 0.0], [back_x, half_width, 0.0]),
            ([back_x, -half_width, 0.0], [tip_x, -half_width, 0.0]),
            ([back_x, half_width, 0.0], [tip_x, half_width, 0.0]),
            ([0.0, -half_width * 0.35, 0.0], [0.0, half_width * 0.35, 0.0]),
        )

        marker = self._new_diagnostic_marker(
            marker_id,
            f"{namespace_prefix}{side}_{'actual' if actual else 'target'}_claw_u_wireframe",
            Marker.LINE_LIST,
            frame_id=frame_id,
        )
        marker_id += 1
        marker.scale.x = 0.012
        if side == "left":
            self._set_marker_color(marker, 1.0, 0.1, 1.0, 1.0)
        else:
            self._set_marker_color(marker, 1.0, 0.45, 0.05, 1.0)
        for start_local, end_local in local_segments:
            marker.points.append(
                    self._point_from_vector(
                    self._transform_local_point_by_pose(claw_pose, start_local)
                )
            )
            marker.points.append(
                self._point_from_vector(
                    self._transform_local_point_by_pose(claw_pose, end_local)
                )
            )
        marker_array.markers.append(marker)

        text = self._new_diagnostic_marker(
            marker_id,
            f"{namespace_prefix}{side}_{'actual' if actual else 'target'}_claw_u_text",
            Marker.TEXT_VIEW_FACING,
            frame_id=frame_id,
        )
        marker_id += 1
        text.pose.position.x = float(claw_pose[0])
        text.pose.position.y = float(claw_pose[1])
        text.pose.position.z = float(claw_pose[2] + 0.08)
        text.scale.z = 0.055
        self._set_marker_color(text, marker.color.r, marker.color.g, marker.color.b, 1.0)
        text.text = f"{side} claw U"
        marker_array.markers.append(text)
        return marker_id

    def _append_fp_box_diagnostic_markers(self, marker_array, marker_id):
        """在夹爪诊断里叠加当前 FP 箱体，方便直接看夹爪相对箱体的位置。"""
        fp_box = self._get_latest_fp_box_for_diagnostics()
        if fp_box is None:
            return marker_id

        center_base, left_axis_base, front_axis_base, up_axis_base, box_size = fp_box
        center_map = self._transform_pose_from_base_to_map([*center_base, 0.0, 0.0, 0.0])
        left_axis_map = self._transform_axis_from_base_to_map(center_base, left_axis_base)
        front_axis_map = self._transform_axis_from_base_to_map(center_base, front_axis_base)
        up_axis_map = self._transform_axis_from_base_to_map(center_base, up_axis_base)
        if (
            center_map is None
            or left_axis_map is None
            or front_axis_map is None
            or up_axis_map is None
        ):
            return marker_id

        center = np.array(center_map[:3], dtype=float)
        corners = self._fp_box_corners(
            center,
            left_axis_map,
            front_axis_map,
            up_axis_map,
            box_size,
        )
        box_marker = self._new_diagnostic_marker(marker_id, "fp_box_outline", Marker.LINE_LIST)
        marker_id += 1
        box_marker.scale.x = 0.025
        self._set_marker_color(box_marker, 0.0, 0.85, 1.0, 0.95)
        for start_index, end_index in self._box_edge_indices():
            box_marker.points.append(self._point_from_vector(corners[start_index]))
            box_marker.points.append(self._point_from_vector(corners[end_index]))
        marker_array.markers.append(box_marker)

        center_marker = self._new_diagnostic_marker(marker_id, "fp_box_center", Marker.SPHERE)
        marker_id += 1
        center_marker.pose.position = self._point_from_vector(center)
        center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.075
        self._set_marker_color(center_marker, 1.0, 1.0, 0.0, 1.0)
        marker_array.markers.append(center_marker)

        for label, axis, color in (
            ("fp_left_axis", left_axis_map, (1.0, 0.1, 1.0)),
            ("fp_front_axis", front_axis_map, (0.1, 1.0, 0.1)),
            ("fp_up_axis", up_axis_map, (1.0, 0.7, 0.1)),
        ):
            arrow = self._new_diagnostic_marker(marker_id, label, Marker.ARROW)
            marker_id += 1
            arrow.scale.x = 0.035
            arrow.scale.y = 0.07
            arrow.scale.z = 0.07
            arrow.points = [
                self._point_from_vector(center),
                self._point_from_vector(center + axis * 0.30),
            ]
            self._set_marker_color(arrow, color[0], color[1], color[2], 0.95)
            marker_array.markers.append(arrow)

        text = self._new_diagnostic_marker(marker_id, "fp_box_text", Marker.TEXT_VIEW_FACING)
        marker_id += 1
        text.pose.position.x = float(center[0])
        text.pose.position.y = float(center[1])
        text.pose.position.z = float(center[2] + box_size[2] * 0.5 + 0.16)
        text.scale.z = 0.075
        self._set_marker_color(text, 0.0, 0.85, 1.0, 1.0)
        text.text = (
            "FP BOX\n"
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})\n"
            f"size=({box_size[0]:.2f},{box_size[1]:.2f},{box_size[2]:.2f})"
        )
        marker_array.markers.append(text)

        if self.claw_point_diagnostics_base_link_visualization_enabled:
            marker_id = self._append_fp_box_base_link_diagnostic_markers(
                marker_array,
                marker_id,
                center_base,
                left_axis_base,
                front_axis_base,
                up_axis_base,
                box_size,
            )
        return marker_id

    def _append_fp_box_base_link_diagnostic_markers(
        self,
        marker_array,
        marker_id,
        center_base,
        left_axis_base,
        front_axis_base,
        up_axis_base,
        box_size,
    ):
        """在同一诊断话题里额外叠加 base_link 下的 FP 箱体。"""
        center = np.array(center_base[:3], dtype=float)
        corners = self._fp_box_corners(
            center,
            left_axis_base,
            front_axis_base,
            up_axis_base,
            box_size,
        )
        box_marker = self._new_diagnostic_marker(
            marker_id,
            "base_link/fp_box_outline",
            Marker.LINE_LIST,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id += 1
        box_marker.scale.x = 0.018
        self._set_marker_color(box_marker, 0.0, 0.85, 1.0, 0.55)
        for start_index, end_index in self._box_edge_indices():
            box_marker.points.append(self._point_from_vector(corners[start_index]))
            box_marker.points.append(self._point_from_vector(corners[end_index]))
        marker_array.markers.append(box_marker)

        center_marker = self._new_diagnostic_marker(
            marker_id,
            "base_link/fp_box_center",
            Marker.SPHERE,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id += 1
        center_marker.pose.position = self._point_from_vector(center)
        center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.06
        self._set_marker_color(center_marker, 1.0, 1.0, 0.0, 0.65)
        marker_array.markers.append(center_marker)

        text = self._new_diagnostic_marker(
            marker_id,
            "base_link/fp_box_text",
            Marker.TEXT_VIEW_FACING,
            frame_id=BASE_LINK_FRAME,
        )
        marker_id += 1
        text.pose.position.x = float(center[0])
        text.pose.position.y = float(center[1])
        text.pose.position.z = float(center[2] + box_size[2] * 0.5 + 0.12)
        text.scale.z = 0.055
        self._set_marker_color(text, 0.75, 0.95, 1.0, 0.9)
        text.text = (
            "FP BOX base_link\n"
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
        )
        marker_array.markers.append(text)
        return marker_id

    def _transform_pose_from_base_to_map(self, pose):
        """将 [x,y,z,yaw,pitch,roll] 从 base_link 转成 map；pose 为空则返回 None。

        这里刻意和 SelectAndPublishHighestYoloBox 保持同一条转换链路：
        - base_link 与 melon_odom 视为重合；
        - melon_odom/odom 消息中的 pose 表示 ``map <- base_link``；
        - 使用完整 odom 四元数构造 4x4 矩阵，而不是只取二维 yaw。
        """
        if pose is None:
            return None

        odom_msg = self.odom_transformer.get_latest_odom()
        if odom_msg is not None:
            map_from_base = self._map_from_odom_message(odom_msg)
            return self._transform_pose_by_matrix(map_from_base, pose)

        if self.arm_controller is None:
            return None
        try:
            translation, quaternion = self.arm_controller.tf_listener.lookupTransform(
                MAP_FRAME,
                BASE_LINK_FRAME,
                rospy.Time(0),
            )
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 夹爪诊断无法转换 {BASE_LINK_FRAME}->{MAP_FRAME}: {exc}"
            )
            return None

        transform = tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(translation),
            tf_trans.quaternion_matrix(quaternion),
        )
        return self._transform_pose_by_matrix(transform, pose)

    @staticmethod
    def _map_from_odom_message(odom_msg):
        """从 odom.pose 构造 ``map <- base_link``，与选箱节点保持一致。"""
        odom_position = odom_msg.pose.pose.position
        odom_orientation = odom_msg.pose.pose.orientation
        return tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(
                [
                    float(odom_position.x),
                    float(odom_position.y),
                    float(odom_position.z),
                ]
            ),
            tf_trans.quaternion_matrix(
                [
                    float(odom_orientation.x),
                    float(odom_orientation.y),
                    float(odom_orientation.z),
                    float(odom_orientation.w),
                ]
            ),
        )

    @staticmethod
    def _transform_pose_by_matrix(transform, pose):
        """使用 ``map <- base`` 矩阵转换位置和姿态。"""
        point = tf_trans.translation_from_matrix(
            np.dot(transform, tf_trans.translation_matrix(pose[:3]))
        )

        base_orientation = tf_trans.quaternion_from_euler(
            math.radians(float(pose[5])),
            math.radians(float(pose[4])),
            math.radians(float(pose[3])),
        )
        transform_orientation = tf_trans.quaternion_from_matrix(transform)
        map_orientation = tf_trans.quaternion_multiply(transform_orientation, base_orientation)
        roll, pitch, yaw = tf_trans.euler_from_quaternion(map_orientation)
        return [
            float(point[0]),
            float(point[1]),
            float(point[2]),
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        ]

    @staticmethod
    def _transform_local_point_by_pose(pose, local_point):
        """将 claw 局部坐标点按 map 下 claw 位姿转换到 map。"""
        translation = tf_trans.translation_matrix(
            [float(pose[0]), float(pose[1]), float(pose[2])]
        )
        rotation = tf_trans.quaternion_matrix(
            tf_trans.quaternion_from_euler(
                math.radians(float(pose[5])),
                math.radians(float(pose[4])),
                math.radians(float(pose[3])),
            )
        )
        transform = tf_trans.concatenate_matrices(translation, rotation)
        point = transform.dot(
            [
                float(local_point[0]),
                float(local_point[1]),
                float(local_point[2]),
                1.0,
            ]
        )
        return np.array([float(point[0]), float(point[1]), float(point[2])], dtype=float)

    def _transform_axis_from_base_to_map(self, center_base, axis_base):
        center_pose = self._transform_pose_from_base_to_map([*center_base, 0.0, 0.0, 0.0])
        end_pose = self._transform_pose_from_base_to_map(
            [*(np.array(center_base, dtype=float) + np.array(axis_base, dtype=float)), 0.0, 0.0, 0.0]
        )
        if center_pose is None or end_pose is None:
            return None
        return self._normalize_vector(np.array(end_pose[:3], dtype=float) - np.array(center_pose[:3], dtype=float))

    def _get_latest_fp_box_for_diagnostics(self):
        """读取当前 FP 箱体中心、方向轴和尺寸；没有有效 FP 数据时返回 None。"""
        if self.services is None or not hasattr(self.services, "box_detector"):
            return None
        detector = self.services.box_detector
        center = self._as_vector(self._call_optional(detector, "get_latest_box_center"))
        axes = self._call_optional(detector, "get_latest_box_axes")
        left_axis = self._normalize_vector(
            self._as_vector(axes.get("left") if isinstance(axes, dict) else None)
        )
        up_axis = self._normalize_vector(
            self._as_vector(axes.get("up") if isinstance(axes, dict) else None)
        )
        front_axis = self._normalize_vector(
            self._as_vector(self._call_optional(detector, "get_latest_box_front_axis"))
        )
        if center is None or left_axis is None or up_axis is None:
            return None
        if front_axis is None:
            front_axis = self._normalize_vector(np.cross(left_axis, up_axis))
        if front_axis is None:
            return None

        # 和 FP 抓取点可视化保持一致：先正交化 left，再按 left/up 求 front 方向。
        left_axis = self._normalize_vector(left_axis - np.dot(left_axis, up_axis) * up_axis)
        if left_axis is None:
            return None
        derived_front_axis = self._normalize_vector(np.cross(left_axis, up_axis))
        if derived_front_axis is None:
            return None
        if np.dot(derived_front_axis, front_axis) < 0.0:
            derived_front_axis = -derived_front_axis

        box_size = self._parse_box_size(self._call_optional(detector, "get_latest_box_size"))
        if box_size is None:
            box_size = self._parse_box_size(
                [
                    getattr(detector, "box_size_x", None),
                    getattr(detector, "box_size_y", None),
                    getattr(detector, "box_size_z", None),
                ]
            )
        if box_size is None:
            return None
        return center, left_axis, derived_front_axis, up_axis, box_size

    @staticmethod
    def _call_optional(obj, method_name):
        method = getattr(obj, method_name, None)
        if not callable(method):
            return None
        return method()

    @staticmethod
    def _as_vector(value):
        if value is None:
            return None
        try:
            vector = np.array(value, dtype=float).reshape(-1)
        except Exception:
            return None
        if len(vector) < 3 or not np.all(np.isfinite(vector[:3])):
            return None
        return vector[:3]

    @staticmethod
    def _normalize_vector(vector):
        if vector is None:
            return None
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6 or not math.isfinite(norm):
            return None
        return np.array(vector, dtype=float) / norm

    @staticmethod
    def _parse_box_size(raw_size):
        if raw_size is None:
            return None
        try:
            if isinstance(raw_size, dict):
                size = np.array(
                    [raw_size.get("x"), raw_size.get("y"), raw_size.get("z")],
                    dtype=float,
                )
            else:
                size = np.array(raw_size[:3], dtype=float)
        except Exception:
            return None
        if len(size) != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            return None
        return size

    @staticmethod
    def _fp_box_corners(center, left_axis, front_axis, up_axis, box_size):
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(
                        center
                        + left_axis * sx * box_size[0] * 0.5
                        + front_axis * sy * box_size[1] * 0.5
                        + up_axis * sz * box_size[2] * 0.5
                    )
        return corners

    @staticmethod
    def _box_edge_indices():
        return (
            (0, 1),
            (0, 2),
            (0, 4),
            (3, 1),
            (3, 2),
            (3, 7),
            (5, 1),
            (5, 4),
            (5, 7),
            (6, 2),
            (6, 4),
            (6, 7),
        )

    def _format_diagnostic_text(self, side, claw_delta, eef_delta):
        lines = [f"{side.upper()} CLAW EXEC DIAG"]
        if claw_delta is not None:
            lines.append(
                "claw Δxyz="
                f"({claw_delta['dx']:.3f},{claw_delta['dy']:.3f},{claw_delta['dz']:.3f}) "
                f"d={claw_delta['dist']:.3f}m"
            )
            lines.append(
                "claw Δypr="
                f"({claw_delta['dyaw']:.1f},{claw_delta['dpitch']:.1f},{claw_delta['droll']:.1f})deg"
            )
        if eef_delta is not None:
            lines.append(
                "eef Δxyz="
                f"({eef_delta['dx']:.3f},{eef_delta['dy']:.3f},{eef_delta['dz']:.3f}) "
                f"d={eef_delta['dist']:.3f}m"
            )
        return "\n".join(lines)

    def _new_diagnostic_marker(self, marker_id, namespace, marker_type, frame_id=MAP_FRAME):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.ros_node.now()
        marker.ns = f"claw_point_diagnostics/{namespace}"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _point_message(x, y, z):
        return Point(x=float(x), y=float(y), z=float(z))

    @staticmethod
    def _point_from_vector(vector):
        return Point(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))

    @staticmethod
    def _set_marker_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha
