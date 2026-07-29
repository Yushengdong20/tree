"""在 blackboard 上创建或复用 move_box 真实共享服务。"""

import ast

import py_trees
from py_trees.common import Status

from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import (
    IK_MODEL_MOVE_BOX,
)
from tree.constants import (
    BASE_LINK_FRAME,
    CHASSIS_FRAME,
    MAP_FRAME,
    MODEL_TYPE_KEY,
    ODOM_POSE_TRANSFORMER_KEY,
    ROBOT_SERVICES_KEY,
)
from tree.utils.geometry import get_odom_pose_transformer

from ..base import TimedMockAction


class EnsureMoveBoxServices(TimedMockAction):
    """确保真实机器人所需的 move_box 控制实例存在，并立即进入准备姿态。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.model_type_key = MODEL_TYPE_KEY
        self.odom_transformer_key = ODOM_POSE_TRANSFORMER_KEY
        self.model_type = str(params.get("model_type", IK_MODEL_MOVE_BOX)).strip() or IK_MODEL_MOVE_BOX
        self.foundationpose_axis_convention = str(
            params.get("foundationpose_axis_convention", "right_x_front_y_up_z")
        ).strip().lower()
        # 仅在需要“/foundationpose/box 原始 OBB 与抓取/托盘计算同帧”时启用。
        # 默认 False，保持历史树继续使用 /foundationpose/pose 的行为。
        self.foundationpose_raw_box_topic = str(
            params.get("foundationpose_raw_box_topic", "/foundationpose/box")
        ).strip()
        self.foundationpose_prefer_raw_box = self._to_bool(
            params.get("foundationpose_prefer_raw_box", False)
        )
        self.odom_topic = str(params.get("odom_topic", CHASSIS_FRAME)).strip()
        self.odom_target_frame = str(params.get("odom_target_frame", MAP_FRAME)).strip()
        self.odom_base_frame = str(params.get("odom_base_frame", BASE_LINK_FRAME)).strip()
        self.odom_history_duration_sec = float(params.get("odom_history_duration_sec", 10.0))
        self.odom_queue_size = int(params.get("odom_queue_size", 10))
        # 初始双臂位姿默认由 ArmController 的 ROS 参数提供。这里允许单棵树用 JSON
        # 覆盖，便于更换夹爪后独立标定抓取预备姿态，不影响未配置该参数的历史流程。
        self.initial_left_pose_in_waist = self._parse_arm_pose(
            params.get("initial_left_pose_in_waist"),
            "initial_left_pose_in_waist",
        )
        self.initial_right_pose_in_waist = self._parse_arm_pose(
            params.get("initial_right_pose_in_waist"),
            "initial_right_pose_in_waist",
        )
        # 可选的新夹爪工具外参，均相对各自 zarm_*_end_effector 坐标系。
        # 未配置时沿用 StaticTfPublisher 中的 ROS 参数/历史默认值。
        self.left_claw_translation = self._parse_vector(
            params.get("left_claw_translation"), "left_claw_translation"
        )
        self.right_claw_translation = self._parse_vector(
            params.get("right_claw_translation"), "right_claw_translation"
        )
        self.left_claw_rotation_rpy_deg = self._parse_vector(
            params.get("left_claw_rotation_rpy_deg"), "left_claw_rotation_rpy_deg"
        )
        self.right_claw_rotation_rpy_deg = self._parse_vector(
            params.get("right_claw_rotation_rpy_deg"), "right_claw_rotation_rpy_deg"
        )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.model_type_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.odom_transformer_key, access=py_trees.common.Access.WRITE)

    @staticmethod
    def _parse_arm_pose(raw_value, parameter_name):
        """解析可选的 waist_yaw_link 初始手臂位姿。"""
        if raw_value is None or raw_value == "":
            return None
        if isinstance(raw_value, str):
            raw_value = ast.literal_eval(raw_value)
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 6:
            raise ValueError(
                f"{parameter_name} 必须是 [x, y, z, yaw, pitch, roll]，长度为 6"
            )
        return [float(value) for value in raw_value]

    @staticmethod
    def _parse_vector(raw_value, parameter_name):
        """解析可选的三维平移或 [roll, pitch, yaw] 参数。"""
        if raw_value is None or raw_value == "":
            return None
        if isinstance(raw_value, str):
            raw_value = ast.literal_eval(raw_value)
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
            raise ValueError(f"{parameter_name} 必须是长度为 3 的列表")
        return [float(value) for value in raw_value]

    def update(self):
        """若服务不存在则创建，并在同一节点内完成头、腰、手臂的准备动作。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if not self._is_move_box_services(services):
            from tree.runtime.move_box.move_box_real import build_robot_services

            services = build_robot_services(
                model_type=self.model_type,
                foundationpose_axis_convention=self.foundationpose_axis_convention,
                foundationpose_raw_box_topic=self.foundationpose_raw_box_topic,
                foundationpose_prefer_raw_box=self.foundationpose_prefer_raw_box,
            )
            self.blackboard.set(self.services_key, services, overwrite=True)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] created robot services: services_id={id(services)}"
            )
        else:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] reused robot services: services_id={id(services)}"
            )
            set_axis_convention = getattr(services.box_detector, "set_axis_convention", None)
            if callable(set_axis_convention):
                set_axis_convention(self.foundationpose_axis_convention)
        self.blackboard.set(self.model_type_key, services.model_type, overwrite=True)
        self._ensure_shared_odom_transformer()
        self._prepare_robot(services)
        return Status.SUCCESS

    def _ensure_shared_odom_transformer(self):
        """统一初始化并缓存 odom 位姿转换器，供后续节点从 blackboard 复用。"""
        transformer = get_odom_pose_transformer(
            self.ros_node,
            odom_topic=self.odom_topic,
            target_frame=self.odom_target_frame,
            base_frame=self.odom_base_frame,
            queue_size=self.odom_queue_size,
            history_duration_sec=self.odom_history_duration_sec,
        )
        self.blackboard.set(self.odom_transformer_key, transformer, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已初始化共享 OdomPoseTransformer: "
            f"key={self.odom_transformer_key}, topic={self.odom_topic}, "
            f"target_frame={self.odom_target_frame}, base_frame={self.odom_base_frame}, "
            f"history={transformer.history_duration_sec:.1f}s, transformer_id={id(transformer)}"
        )

    def _is_move_box_services(self, services):
        return (
            services is not None
            and hasattr(services, "static_tf_publisher")
            and hasattr(services, "arm_controller")
            and getattr(services, "model_type", None) == self.model_type
        )

    def _prepare_robot(self, services):
        """发布静态 TF，并驱动头、腰、手臂进入初始观测和抓取准备状态。"""
        services.static_tf_publisher.publish_claw_tfs(
            left_extrinsics=self._get_claw_extrinsics("left"),
            right_extrinsics=self._get_claw_extrinsics("right"),
        )
        self._apply_initial_arm_pose_override(services.arm_controller)
        if self.should_skip_head_motion():
            self.log_skip_head_motion()
        else:
            services.head_controller.set_head_target(
                float(self.ros_node.get_param("initial_head_yaw", 0.0)),
                float(self.ros_node.get_param("initial_head_pitch", 20.0)),
            )

        self.ros_node.sleep(1.0)
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
        else:
            services.arm_controller.prepare()
        self.ros_node.sleep(1.0)

        torso_ready_pose = list(services.torso_controller.current_pose)
        torso_ready_pose[0] = 0.0
        torso_ready_pose[2] = float(
            self.params.get(
                "initial_torso_height",
                self.ros_node.get_param("initial_torso_height", 1.1),
            )
        )
        torso_ready_pose[4] = 0.0
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
            return
        services.torso_controller.move_to_pose(torso_ready_pose, wait_done=False)

    def _get_claw_extrinsics(self, side):
        """组装单侧显式外参；全部缺省时返回 None，以保留发布器原有默认来源。"""
        translation = getattr(self, f"{side}_claw_translation")
        rotation_rpy_deg = getattr(self, f"{side}_claw_rotation_rpy_deg")
        if translation is None and rotation_rpy_deg is None:
            return None
        result = {}
        if translation is not None:
            result["translation"] = list(translation)
        if rotation_rpy_deg is not None:
            result["rotation_rpy_deg"] = list(rotation_rpy_deg)
        return result

    def _apply_initial_arm_pose_override(self, arm_controller):
        """将树级初始姿态写入控制器，确保 prepare() 与后续默认 YPR 使用同一标定值。"""
        if self.initial_left_pose_in_waist is None and self.initial_right_pose_in_waist is None:
            return

        if self.initial_left_pose_in_waist is not None:
            arm_controller.initial_left_pose_in_waist = list(self.initial_left_pose_in_waist)
        if self.initial_right_pose_in_waist is not None:
            arm_controller.initial_right_pose_in_waist = list(self.initial_right_pose_in_waist)

        # 同步 base_link 缓存；claw_point 未显式配置 claw_ypr 时会使用此处的初始 YPR。
        if hasattr(arm_controller, "refresh_initial_pose_in_base_link"):
            arm_controller.refresh_initial_pose_in_base_link()
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已应用树级初始双臂位姿(waist_yaw_link): "
            f"left={arm_controller.initial_left_pose_in_waist}, "
            f"right={arm_controller.initial_right_pose_in_waist}"
        )

    def describe_start(self):
        """返回节点开始执行时的日志描述。"""
        return (
            f"[{self.config_label}] EnsureMoveBoxServices start: "
            f"key={self.services_key}, odom_key={self.odom_transformer_key}, "
            f"odom_topic={self.odom_topic}, "
            f"foundationpose_axis_convention={self.foundationpose_axis_convention}"
        )
