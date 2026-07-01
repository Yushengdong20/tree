"""双臂关节位置控制节点。

数据来源优先级：
1. JSON 参数 `joint_positions`，直接给出 14 维双臂关节目标。
2. blackboard 参数 `joint_positions_key`，读取前置节点动态写入的 14 维关节目标。

关节顺序与 EventArmMoveJointState / ArmController.start_arm_joint_state_event() 保持一致：
- 前 7 维为左臂关节
- 后 7 维为右臂关节
- 单位为 rad
"""

import ast
import math

import py_trees
from py_trees.common import Status
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.base_event import EventStatus

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class ArmToJointPosition(TimedMockAction):
    """读取 14 维双臂关节目标，并复用 ArmController 的关节事件。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.joint_positions = self._parse_joint_positions(
            params.get("joint_positions", None),
            "joint_positions",
            allow_empty=True,
        )
        self.joint_positions_key = str(params.get("joint_positions_key", "")).strip()
        self.duration = float(params.get("duration", 2.0))
        self.publish_hz = float(params.get("publish_hz", 100.0))

        self.arm_controller = None
        self.started = False
        self.skipped = False
        self.startup_error = None

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.joint_positions_key:
            self.blackboard.register_key(
                key=self.joint_positions_key,
                access=py_trees.common.Access.READ,
            )

    def initialise(self):
        """解析目标关节并启动 ArmController 关节事件。"""
        super().initialise()

        self.arm_controller = None
        self.started = False
        self.skipped = False
        self.startup_error = None

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
            joint_positions, target_source = self._resolve_joint_positions()
            self.arm_controller = services.arm_controller
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 启动双臂关节位置控制: "
                f"source={target_source}, duration={self.duration:.3f}, "
                f"publish_hz={self.publish_hz:.3f}, joint_positions={joint_positions}"
            )

            # 关键步骤：线程和事件状态统一交给 ArmController，节点只负责行为树生命周期。
            if not self.arm_controller.start_arm_joint_state_event(
                joint_positions,
                duration=self.duration,
                publish_hz=self.publish_hz,
            ):
                self.startup_error = RuntimeError("启动双臂关节事件失败")
                return
            self.started = True
        except Exception as exc:
            self.startup_error = exc
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 启动双臂关节位置控制失败: {exc}"
            )

    @staticmethod
    def _parse_joint_positions(value, name, allow_empty=False):
        if value is None or value == "":
            if allow_empty:
                return None
            raise ValueError(f"{name} 不能为空")
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 14:
            raise ValueError(f"{name} 必须是长度为 14 的列表，前 7 维左臂，后 7 维右臂")

        joint_positions = [float(item) for item in value]
        if not all(math.isfinite(item) for item in joint_positions):
            raise ValueError(f"{name} 必须全部为有限数值")
        return joint_positions

    def _resolve_joint_positions(self):
        """按 JSON > blackboard 的优先级解析 14 维关节目标。"""
        if self.joint_positions is not None:
            return list(self.joint_positions), "json:joint_positions"

        if not self.joint_positions_key:
            raise ValueError("缺少 joint_positions 或 joint_positions_key")
        if not self.blackboard.exists(self.joint_positions_key):
            raise ValueError(f"blackboard 缺少双臂关节目标: key={self.joint_positions_key}")

        return (
            self._parse_joint_positions(
                self.blackboard.get(self.joint_positions_key),
                self.joint_positions_key,
            ),
            f"blackboard:{self.joint_positions_key}",
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
            self.started = False
            return Status.SUCCESS

        self.ros_node.get_logger().error(
            f"[{self.config_label}] 双臂关节位置控制失败: {arm_status}"
        )
        self.started = False
        return Status.FAILURE

    def terminate(self, new_status):
        """节点被中断时关闭 ArmController 中正在运行的关节事件。"""
        if (
            self.started
            and self.arm_controller is not None
            and new_status != Status.SUCCESS
        ):
            # 关键步骤：和 ArmsToPose 保持一致，上层切走 RUNNING 节点时停止当前 active arm event。
            self.arm_controller.stop_arm_event()
        self.started = False
        super().terminate(new_status)

    def describe_start(self):
        if self.joint_positions is not None:
            target_desc = self.joint_positions
        else:
            target_desc = f"blackboard:{self.joint_positions_key or '<missing>'}"
        return (
            f"[{self.config_label}] ArmToJointPosition start: "
            f"joint_positions={target_desc}, duration={self.duration:.3f}, "
            f"publish_hz={self.publish_hz:.3f}"
        )
