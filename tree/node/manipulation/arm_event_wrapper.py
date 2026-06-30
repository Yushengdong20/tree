# -*- coding: utf-8 -*-
"""ArmEventWrapper node."""

import py_trees
from py_trees.common import Status

# SDK 相关导入
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.data_type import Frame, Pose
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate import EventArmMoveKeyPoint
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.base_event import EventStatus
from kuavo_humanoid_sdk.interfaces.data_types import KuavoManipulationMpcCtrlMode

from shared_robot_sdk import get_shared_robot_sdk
from ..base import TimedMockAction
from tree.utils.arm_target import is_pose_object_pair, looks_like_arm_pose_pair, normalize_wrench
from tree.utils.params import parse_param_value


class ArmEventWrapper(TimedMockAction):
    """
    V3.0 标准版机械臂移动节点

    当前版本直接从 JSON params 读取目标位姿/力控参数，不依赖 blackboard。
    """

    def __init__(self, name, config_label, ros_node, params):
        super(ArmEventWrapper, self).__init__(name, config_label, ros_node, params)
        self.control_mode = self.params.get("arm_control_mode", "fixed_base")
        self.arm_timeout = int(self.params.get("arm_timeout", 40))
        self.pos_threshold = float(self.params.get("pos_threshold", 0.21))
        self.angle_threshold = float(self.params.get("angle_threshold", 0.45))
        self.blackboard_target_key = self.params.get("blackboard_target_key", "arm_target")
        self.blackboard_pose_key = self.params.get("blackboard_pose_key", "target_pose")
        self.blackboard_wrench_key = self.params.get("blackboard_wrench_key", "target_wrench")
        self.arm_event = None
        self.robot_sdk = get_shared_robot_sdk()
        self.target_pose = None
        self.target_wrench = None
        self.skipped = False

        self.blackboard.register_key(
            key=self.blackboard_target_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.blackboard_pose_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.blackboard_wrench_key,
            access=py_trees.common.Access.READ,
        )

        self.arm_event = EventArmMoveKeyPoint(
            robot_sdk=self.robot_sdk,
            timeout=self.arm_timeout,
            arm_control_mode=self.control_mode,
            pos_threshold=self.pos_threshold,
            angle_threshold=self.angle_threshold,
        )

        self.ros_node.get_logger().info(
            f"[{self.config_label}] ArmEventWrapper configured: control_mode={self.control_mode}"
        )

    def initialise(self):
        super(ArmEventWrapper, self).initialise()
        self.feedback_message = "准备移动手臂"
        self.target_pose = None
        self.target_wrench = None
        self.skipped = False
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            self.feedback_message = "已跳过手臂动作"
            self.skipped = True
            return
        raw_target_pose = self.params.get("target_pose")
        raw_target_wrench = self.params.get("target_wrench")
        raw_blackboard_target = self._read_blackboard_value(self.blackboard_target_key)
        raw_blackboard_pose = self._read_blackboard_value(self.blackboard_pose_key)
        raw_blackboard_wrench = self._read_blackboard_value(self.blackboard_wrench_key)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] raw target_pose={raw_target_pose!r}"
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] raw target_wrench={raw_target_wrench!r}"
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] blackboard {self.blackboard_target_key}={raw_blackboard_target!r}"
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] blackboard {self.blackboard_pose_key}={raw_blackboard_pose!r}"
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] blackboard {self.blackboard_wrench_key}={raw_blackboard_wrench!r}"
        )

        try:
            self.target_pose, self.target_wrench = self._resolve_target_from_params()
        except Exception as exc:
            self.feedback_message = f"解析 JSON 参数失败: {exc}"
            self.ros_node.get_logger().error(f"[{self.config_label}] {self.feedback_message}")
            return

        self.ros_node.get_logger().info(
            f"[{self.config_label}] parsed target_pose={self.target_pose!r}"
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] parsed target_wrench={self.target_wrench!r}"
        )

        if self.target_pose is None:
            self.feedback_message = "未获取到目标位姿，请在 JSON params 中配置 target_pose"
            self.ros_node.get_logger().error(f"[{self.config_label}] {self.feedback_message}")
            return

        try:

            if self.control_mode == "manipulate_mpc":
                self.robot_sdk.control.set_manipulation_mpc_mode(KuavoManipulationMpcCtrlMode.BaseArm)
            elif self.control_mode == "fixed_base":
                self.robot_sdk.control.set_manipulation_mpc_mode(KuavoManipulationMpcCtrlMode.ArmOnly)

            # 若 terminate 已清空 arm_event，则重新创建（支持同一节点实例被多次执行）
            if self.arm_event is None:
                self.arm_event = EventArmMoveKeyPoint(
                    robot_sdk=self.robot_sdk,
                    timeout=self.arm_timeout,
                    arm_control_mode=self.control_mode,
                    pos_threshold=self.pos_threshold,
                    angle_threshold=self.angle_threshold,
                )

            self.arm_event.open()

            if not self.arm_event.set_target(
                self.target_pose,
                arm_wrench=self.target_wrench,
                tag=None,
            ):
                self.ros_node.get_logger().error(f"[{self.config_label}] SDK set_target 失败")
                self.arm_event.close()
            else:
                self.feedback_message = f"开始移动 (Mode: {self.control_mode})"

        except Exception as e:
            self.ros_node.get_logger().error(f"[{self.config_label}] 初始化异常: {e}")
            import traceback
            traceback.print_exc()

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if self.skipped:
            return Status.SUCCESS

        if not self.arm_event:
            return Status.FAILURE

        arm_status = self.arm_event.step()

        if arm_status == EventStatus.RUNNING:
            return Status.RUNNING
        elif arm_status == EventStatus.SUCCESS:
            self.arm_event.close()
            self.feedback_message = "移动成功"
            return Status.SUCCESS
        else:
            self.arm_event.close()
            self.feedback_message = "移动失败"
            return Status.FAILURE

    def terminate(self, new_status):
        if self.arm_event:
            try:
                self.arm_event.close()
            except Exception:
                pass

    def _resolve_target_from_params(self):
        """优先从 blackboard 解析，其次回退到 JSON params。"""
        blackboard_target = parse_param_value(
            self._read_blackboard_value(self.blackboard_target_key)
        )
        blackboard_pose = parse_param_value(
            self._read_blackboard_value(self.blackboard_pose_key)
        )
        blackboard_wrench = parse_param_value(
            self._read_blackboard_value(self.blackboard_wrench_key)
        )

        if blackboard_target is not None:
            if isinstance(blackboard_target, dict):
                if "pose" in blackboard_target or "wrench" in blackboard_target:
                    target_pose = blackboard_target.get("pose")
                    target_wrench = blackboard_target.get("wrench", blackboard_wrench)
                else:
                    target_pose = blackboard_target
                    target_wrench = blackboard_wrench
                return self._normalize_target(target_pose, target_wrench)

            if isinstance(blackboard_target, (list, tuple)) and len(blackboard_target) == 2:
                first, second = blackboard_target
                if looks_like_arm_pose_pair(first):
                    return self._normalize_target(first, second)

        if blackboard_pose is not None:
            return self._normalize_target(blackboard_pose, blackboard_wrench)

        direct_target = parse_param_value(self.params.get("target_pose"))
        direct_wrench = parse_param_value(self.params.get("target_wrench"))

        if isinstance(direct_target, dict):
            if "pose" in direct_target or "wrench" in direct_target:
                target_pose = direct_target.get("pose")
                target_wrench = direct_target.get("wrench", direct_wrench)
            else:
                target_pose = direct_target
                target_wrench = direct_wrench
            return self._normalize_target(target_pose, target_wrench)

        if isinstance(direct_target, (list, tuple)) and len(direct_target) == 2:
            first, second = direct_target
            if looks_like_arm_pose_pair(first):
                return self._normalize_target(first, second)

        pose_from_nested = parse_param_value(self.params.get("target_pose.pose"))
        wrench_from_nested = parse_param_value(self.params.get("target_pose.wrench"))

        target_pose = pose_from_nested if pose_from_nested is not None else direct_target
        target_wrench = (
            wrench_from_nested if wrench_from_nested is not None else direct_wrench
        )
        return self._normalize_target(target_pose, target_wrench)

    def _read_blackboard_value(self, key):
        if not key:
            return None
        if self.blackboard.exists(key):
            return self.blackboard.get(key)
        return None

    def _normalize_target(self, target_pose, target_wrench):
        if target_pose is None:
            return None, target_wrench

        if is_pose_object_pair(target_pose):
            return target_pose, target_wrench

        if isinstance(target_pose, dict):
            left_specs = target_pose.get("left")
            right_specs = target_pose.get("right")
        elif looks_like_arm_pose_pair(target_pose):
            left_specs, right_specs = target_pose
        else:
            raise ValueError(
                "target_pose 需要是双臂结构："
                "{'left': [...], 'right': [...]} 或 [left_poses, right_poses]"
            )

        left_poses = [self._build_pose(spec) for spec in left_specs]
        right_poses = [self._build_pose(spec) for spec in right_specs]
        normalized_pose = (left_poses, right_poses)
        normalized_wrench = normalize_wrench(target_wrench, len(left_poses), len(right_poses))
        return normalized_pose, normalized_wrench

    def _build_pose(self, spec):
        if hasattr(spec, "frame"):
            return spec
        if not isinstance(spec, dict):
            raise ValueError(f"pose 描述必须是 dict 或 Pose，当前是 {type(spec)}")

        pos = spec.get("pos") or spec.get("position")
        euler = spec.get("euler")
        if pos is None or euler is None:
            raise ValueError("pose 描述缺少 pos/euler 字段")

        frame_name = str(spec.get("frame", "BASE")).upper()
        degrees = bool(spec.get("degrees", True))
        frame = getattr(Frame, frame_name, None)
        if frame is None:
            raise ValueError(f"未知 frame: {frame_name}")

        return Pose.from_euler(
            pos=tuple(pos),
            euler=tuple(euler),
            frame=frame,
            degrees=degrees,
        )
