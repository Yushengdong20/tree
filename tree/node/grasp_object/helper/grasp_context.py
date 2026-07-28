"""grasp_object 运行时上下文和 TF 查询。"""

import numpy as np

from tree.constants import (
    BASE_LINK_FRAME,
    CHASSIS_FRAME,
    FIXED_KNEE_FRAME,
    KNEE_LINK_FRAME,
    MAP_FRAME,
    ODOM_POSE_TRANSFORMER_KEY,
    WAIST_YAW_LINK_FRAME,
)
from tree.utils.geometry import (
    base_from_map_matrix_via_melon_odom,
    get_shared_odom_pose_transformer,
    lookup_transform_matrix,
)


class GraspContext:
    """封装 services、控制器和 TF 上下文。"""

    def prepare_context(self):
        """准备 TF、控制器和当前双臂目标。"""
        self._ensure_runtime_context()
        fixed_knee_from_base = self._lookup_fixed_knee_from_base()
        base_from_waist = self._lookup_transform_matrix(BASE_LINK_FRAME, WAIST_YAW_LINK_FRAME)
        base_from_map = self._base_from_map_matrix_via_melon_odom()
        # 关键步骤：fixed_knee 是计算用虚拟坐标系，原点跟随 knee_link，坐标轴保持与 base_link 对齐。
        current_fixed_knee_from_waist = fixed_knee_from_base @ base_from_waist
        arm_targets = {}
        for arm_side in ("left", "right"):
            target = self._arm_controller.get_current_end_effector_pose(
                arm_side,
                target_frame=WAIST_YAW_LINK_FRAME,
            )
            if target is None or len(target) != 6:
                raise RuntimeError(f"无法获取{arm_side}臂当前末端位姿")
            arm_targets[arm_side] = list(target)
        return fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map

    def _ensure_runtime_context(self):
        if self._arm_controller is not None:
            return

        import tf

        self._tf = tf
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            raise RuntimeError(f"services 或 arm_controller 缺失: key={self.services_key}")
        self._arm_controller = services.arm_controller
        self.model_type = (
            self.blackboard.get(self.model_type_key)
            if self.blackboard.exists(self.model_type_key)
            else getattr(services, "model_type", None)
        )
        if not self.model_type:
            self.model_type = getattr(self._arm_controller, "model_type", None)
        if not self.model_type:
            raise RuntimeError("model_type 缺失")
        self._tf_listener = getattr(services, "tf_listener", None)
        if self._tf_listener is None:
            self._tf_listener = getattr(self._arm_controller, "tf_listener", None)
        if self._tf_listener is None:
            raise RuntimeError("services 中没有可用的 tf_listener")
        self._torso_controller = getattr(services, "torso_controller", None)
        if self._torso_controller is None:
            raise RuntimeError("services 中没有 torso_controller")
        if getattr(self, "_odom_transformer", None) is None:
            self._odom_transformer = get_shared_odom_pose_transformer(
                self.blackboard,
                self.ros_node,
                odom_topic=getattr(self, "odom_topic", CHASSIS_FRAME),
                target_frame=MAP_FRAME,
                base_frame=BASE_LINK_FRAME,
                history_duration_sec=getattr(self, "odom_history_duration_sec", 10.0),
                key=ODOM_POSE_TRANSFORMER_KEY,
            )

    def _get_current_torso_pose(self):
        pose = list(getattr(self._torso_controller, "current_pose", []))
        if len(pose) != 6:
            raise RuntimeError(f"当前腰部位姿长度异常: {pose}")
        return [float(value) for value in pose]

    def _read_current_arm_targets(self):
        """读取当前左右手末端目标，供重选或复核时保留另一侧姿态。"""
        arm_targets = {}
        for arm_side in ("left", "right"):
            target = self._arm_controller.get_current_end_effector_pose(
                arm_side, target_frame=WAIST_YAW_LINK_FRAME
            )
            if target is None or len(target) != 6:
                raise RuntimeError(f"无法获取{arm_side}臂当前末端位姿")
            arm_targets[arm_side] = list(target)
        return arm_targets

    def _lookup_transform_matrix(self, target_frame, source_frame):
        return lookup_transform_matrix(
            self._tf_listener,
            self.ros_node,
            target_frame,
            source_frame,
            timeout=self.tf_timeout_sec,
        )

    def _base_from_map_matrix_via_melon_odom(self):
        """用 melon_odom topic 的最新消息构造 base_link<-map。"""
        odom_msg = self._odom_transformer.get_latest_odom()
        if odom_msg is None:
            raise RuntimeError(f"等待 odom 数据: topic={self.odom_topic}")
        return base_from_map_matrix_via_melon_odom(
            odom_msg,
            map_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )

    def _lookup_fixed_knee_from_base(self):
        """构造 fixed_knee <- base_link 虚拟变换。"""
        knee_from_base = self._lookup_transform_matrix(KNEE_LINK_FRAME, BASE_LINK_FRAME)
        fixed_knee_from_base = np.array(knee_from_base, copy=True)
        # 关键步骤：fixed_knee 只复用 knee_link 的位置，旋转保持与 base_link 对齐。
        fixed_knee_from_base[:3, :3] = np.eye(3)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 使用虚拟 {FIXED_KNEE_FRAME}: "
            f"position_from_{BASE_LINK_FRAME}="
            f"[x={fixed_knee_from_base[0, 3]:.4f}, "
            f"y={fixed_knee_from_base[1, 3]:.4f}, "
            f"z={fixed_knee_from_base[2, 3]:.4f}], "
            f"rotation=identity"
        )
        return fixed_knee_from_base
