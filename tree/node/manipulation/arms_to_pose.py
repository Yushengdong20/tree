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

import py_trees
from py_trees.common import Status
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.base_event import EventStatus

from ..base import TimedMockAction


class ArmsToPose(TimedMockAction):
    """读取左右臂目标，并复用共享 services 中的 common ArmController。

    JSON 参数：
    - left_pose/right_pose: [x, y, z, yaw, pitch, roll]
    - left_pose_key/right_pose_key: blackboard 中的 eef 完整位姿 key
    - target_type: eef_pose / claw_point
    - left_point_key/right_point_key: blackboard 中的夹爪目标点 key
    - side: both / left / right，默认 both
    - point_key: 单臂 claw_point 模式下的目标点 key
    - pose_frame: 目标位姿坐标系，支持 base_link / waist_yaw_link。
      注意：claw_point 模式目前只能使用 base_link。

    eef_pose 默认使用 waist_yaw_link，是为了和 ArmController.move_to_initial_pose()
    使用的初始化手臂位姿坐标系保持一致；point_key 会隐式选择
    target_type=claw_point 和 pose_frame=base_link。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.left_pose = self._parse_pose(params.get("left_pose", None), "left_pose")
        self.right_pose = self._parse_pose(params.get("right_pose", None), "right_pose")
        self.left_pose_key = str(params.get("left_pose_key", "")).strip()
        self.right_pose_key = str(params.get("right_pose_key", "")).strip()
        self.side = str(params.get("side", "both")).strip().lower()
        self.point_key = str(params.get("point_key", "")).strip()
        default_target_type = "claw_point" if self.point_key else "eef_pose"
        self.target_type = str(params.get("target_type", default_target_type)).strip().lower()
        self.left_point_key = str(params.get("left_point_key", "")).strip()
        self.right_point_key = str(params.get("right_point_key", "")).strip()
        default_pose_frame = "base_link" if self.target_type == "claw_point" else "waist_yaw_link"
        self.pose_frame = str(params.get("pose_frame", default_pose_frame)).strip()
        self.arm_controller = None
        self.started = False
        self.skipped = False
        self.startup_error = None
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        for key in [
            self.left_pose_key,
            self.right_pose_key,
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
            self.arm_controller = services.arm_controller
            resolved = self._resolve_targets(self.arm_controller)
            if resolved is None:
                self.startup_error = RuntimeError("解析手臂目标失败")
                return
            left_target, right_target, target_source, locked_arm_side = resolved
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 使用 common ArmController 启动手臂目标: "
                f"side={self.side}, locked={locked_arm_side}, "
                f"source={target_source}, frame={self.pose_frame}, "
                f"left={left_target}, right={right_target}"
            )

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
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError(f"{name} 必须是长度为 6 的列表: [x, y, z, yaw, pitch, roll]")
        return [float(item) for item in value]

    @staticmethod
    def _parse_point(value, name):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} 必须是长度为 3 的列表: [x, y, z]")
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
        if self.side not in ("both", "left", "right"):
            raise ValueError("side 仅支持 both、left 或 right")
        if self.side != "both":
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

        if self.pose_frame == "waist_yaw_link":
            return (
                list(arm_controller.initial_left_pose_in_waist),
                list(arm_controller.initial_right_pose_in_waist),
                "default:initial_pose@waist_yaw_link",
                None,
            )
        if self.pose_frame == "base_link":
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
        if self.pose_frame != "base_link":
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
        left_ee_point = arm_controller.claw_point_to_end_effector_point(left_point, "left")
        right_ee_point = arm_controller.claw_point_to_end_effector_point(right_point, "right")
        if left_ee_point is None or right_ee_point is None:
            return None

        left_ypr = arm_controller.get_initial_left_ypr()
        right_ypr = arm_controller.get_initial_right_ypr()
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
        return left_target, right_target, "blackboard:claw_point", None

    def _resolve_single_arm_targets(self, arm_controller):
        """解析单侧目标，并使用当前缓存目标填充将被锁住的另一侧。"""
        moving_side = self.side
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
            ee_point = arm_controller.claw_point_to_end_effector_point(point, side)
            if ee_point is None:
                return None, None
            ypr = (
                arm_controller.get_initial_left_ypr_by_frame(self.pose_frame)
                if side == "left"
                else arm_controller.get_initial_right_ypr_by_frame(self.pose_frame)
            )
            return [*ee_point, *ypr], f"blackboard:claw_point:{point_key}"

        direct_pose = self.left_pose if side == "left" else self.right_pose
        if direct_pose is not None:
            return list(direct_pose), f"json:{side}_pose"

        pose_key = self.left_pose_key if side == "left" else self.right_pose_key
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

    def describe_start(self):
        if self.left_pose is not None or self.right_pose is not None:
            left_desc = self.left_pose
            right_desc = self.right_pose
        elif self.left_pose_key or self.right_pose_key:
            left_desc = f"blackboard:{self.left_pose_key}"
            right_desc = f"blackboard:{self.right_pose_key}"
        elif self.target_type == "claw_point":
            if self.side == "left":
                left_desc = f"claw_point:{self.point_key or self.left_point_key}"
                right_desc = "locked"
            elif self.side == "right":
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
            f"side={self.side}, target_type={self.target_type}, frame={self.pose_frame}, "
            f"left={left_desc}, right={right_desc}"
        )
