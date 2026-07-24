"""根据已选码垛策略，在当前站位刷新放箱动作点。

这个节点的定位很窄：只刷新动作点，不重新选择码垛 slot / 导航点 / 放箱策略。

为什么需要它：
- ComputeMoveBoxPalletPlaceStrategy 在抓箱后、导航到垛盘前执行一次，用 map 系下的
  slot 规划出本轮最终放置箱心、预落位箱心、放箱策略和导航站位。
- 但是 ArmsToPose 执行动作点使用的是 base_link 下的夹爪点。机器人导航到垛盘后，
  base_link 已经变了；如果继续使用导航前算出的 base_link 动作点，就会错位。
- 因此到达垛盘站位后，需要只把“已定好的 map 目标箱心”重新投影到当前 base_link，
  并结合当前左右夹爪实际点，刷新 pre/push/lift 等夹爪动作点。

高度处理原则：
- 优先使用闭爪后 ``MoveBoxHeldBoxVisualizationMonitor`` 锁定的
  ``held_box_transform_key``。其中保存了 FP 箱心、左右夹爪以及它们之间的真实三维
  相对关系。放置时以“目标箱心 + 该真实相对关系”反算左右夹爪目标，避免假设每次
  都夹在箱体固定高度。
- 目标箱心从 map 转到 base_link 时使用 odom 的完整 3D 四元数变换，不能只做二维
  yaw 旋转并用 ``z_map - z_base`` 近似。
- 仅当抓取偏移尚未成功锁定时，才退回旧的固定箱高算法，以兼容旧树和单节点测试。
- 对于平整地面，可锁定首次到达垛盘站位时的 ``map.z`` 作为高度参考；后续箱子仍
  使用实时 x/y/yaw，但不再把 ``melon_odom.z`` 的漂移带进放置高度。

这比第二次运行 ComputeMoveBoxPalletPlaceStrategy 更安全：
- 不会重新选择 direct / right_push_left / left_push_right；
- 不会因为 auto 逻辑覆盖第一次策略；
- 只更新执行动作必须依赖当前机器人姿态的 base_link 点。
"""

import numpy as np
import py_trees
import tf.transformations as tf_trans
from py_trees.common import Status

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.chassis_navigation import Pose2D
from tree.utils.geometry import transform_global_point_to_base
from tree.utils.pallet_place_diagnostics import write_pallet_place_diagnostic
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction


class ComputeMoveBoxPalletPlaceActionPoints(TimedMockAction):
    """按当前站位刷新码垛放箱动作点。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()

        self.strategy_key = str(params.get("strategy_key", "move_box_pallet_place_strategy")).strip()
        self.final_box_pose_key = str(
            params.get("final_box_pose_key", "move_box_pallet_place_final_box_pose")
        ).strip()
        self.pre_box_pose_key = str(
            params.get("pre_box_pose_key", "move_box_pallet_place_pre_box_pose")
        ).strip()
        self.place_plane_height_key = str(
            params.get("place_plane_height_key", "move_box_pallet_stack_place_plane_height")
        ).strip()
        self.held_box_transform_key = str(
            params.get("held_box_transform_key", "move_box_held_box_grasp_transform")
        ).strip()
        self.use_held_box_grasp_transform = self._to_bool(
            params.get("use_held_box_grasp_transform", True)
        )
        self.height_reference_mode = str(
            params.get("height_reference_mode", "current_odom")
        ).strip().lower()
        if self.height_reference_mode not in ("current_odom", "lock_on_first_use", "fixed"):
            self.height_reference_mode = "current_odom"
        self.height_reference_key = str(
            params.get("height_reference_key", "move_box_pallet_map_base_z_reference")
        ).strip()
        self.fixed_height_reference_map_z = float(
            params.get("fixed_height_reference_map_z", 0.0)
        )

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

        self.box_size_z = float(params.get("box_size_z", 0.34))
        self.claw_lift_clearance_m = float(params.get("claw_lift_clearance_m", 0.18))

        for key in (
            self.services_key,
            self.strategy_key,
            self.final_box_pose_key,
            self.pre_box_pose_key,
            self.place_plane_height_key,
            self.held_box_transform_key,
            self.height_reference_key,
        ):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

        for key in (
            self.pre_place_left_claw_point_key,
            self.pre_place_right_claw_point_key,
            self.push_left_claw_point_key,
            self.push_right_claw_point_key,
            self.lift_left_claw_point_key,
            self.lift_right_claw_point_key,
        ):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

        if self.height_reference_key:
            self.blackboard.register_key(
                key=self.height_reference_key, access=py_trees.common.Access.WRITE
            )

        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame="base_link",
        )

    def update(self):
        strategy = self._read_strategy()
        final_box_pose = self._read_pose(self.final_box_pose_key, required=True)
        pre_box_pose = self._read_pose(self.pre_box_pose_key, required=True)
        current_pose = self.odom_transformer.get_current_pose()
        claw_pair = self._get_current_claw_pair()
        if final_box_pose is None or pre_box_pose is None:
            return Status.FAILURE
        if current_pose is None or claw_pair is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 无法刷新码垛动作点: "
                f"odom_ready={current_pose is not None}, claw_pair_ready={claw_pair is not None}"
            )
            return Status.FAILURE

        height_reference_z, height_reference_source = self._resolve_height_reference_map_z(
            current_pose
        )

        left_claw, right_claw = claw_pair
        current_center = (left_claw + right_claw) * 0.5
        action_points = self._build_action_points_from_captured_grasp(
            final_box_pose,
            pre_box_pose,
            right_claw - left_claw,
            height_reference_z,
        )
        if action_points is None:
            # 兼容旧配置：没有成功锁定抓取偏移时，沿用历史固定高度算法。
            target_claw_z = self._resolve_target_claw_z(current_pose, height_reference_z)
            final_center = self._box_center_map_to_base_legacy(
                final_box_pose, current_pose, target_claw_z
            )
            pre_center = self._box_center_map_to_base_legacy(
                pre_box_pose, current_pose, target_claw_z
            )
            pre_delta = pre_center - current_center
            final_delta = final_center - current_center
            pre_left = left_claw + pre_delta
            pre_right = right_claw + pre_delta
            final_left = left_claw + final_delta
            final_right = right_claw + final_delta
            action_point_source = "legacy_fixed_box_height"
        else:
            (
                pre_center,
                final_center,
                pre_left,
                pre_right,
                final_left,
                final_right,
            ) = action_points
            target_claw_z = float(0.5 * (final_left[2] + final_right[2]))
            action_point_source = "captured_held_box_transform"
        lift_left = np.array(final_left, dtype=float)
        lift_right = np.array(final_right, dtype=float)
        lift_left[2] += self.claw_lift_clearance_m
        lift_right[2] += self.claw_lift_clearance_m

        self.blackboard.set(self.pre_place_left_claw_point_key, pre_left, overwrite=True)
        self.blackboard.set(self.pre_place_right_claw_point_key, pre_right, overwrite=True)
        self.blackboard.set(self.push_left_claw_point_key, final_left, overwrite=True)
        self.blackboard.set(self.push_right_claw_point_key, final_right, overwrite=True)
        self.blackboard.set(self.lift_left_claw_point_key, lift_left, overwrite=True)
        self.blackboard.set(self.lift_right_claw_point_key, lift_right, overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已按当前站位刷新码垛动作点: "
            f"strategy={strategy}, "
            f"source={action_point_source}, "
            f"height_ref={height_reference_z:.3f}({height_reference_source}), "
            f"current_center=({current_center[0]:.3f},{current_center[1]:.3f},{current_center[2]:.3f}), "
            f"pre_center=({pre_center[0]:.3f},{pre_center[1]:.3f},{pre_center[2]:.3f}), "
            f"final_center=({final_center[0]:.3f},{final_center[1]:.3f},{final_center[2]:.3f}), "
            f"pre_left=({pre_left[0]:.3f},{pre_left[1]:.3f},{pre_left[2]:.3f}), "
            f"pre_right=({pre_right[0]:.3f},{pre_right[1]:.3f},{pre_right[2]:.3f}), "
            f"final_left=({final_left[0]:.3f},{final_left[1]:.3f},{final_left[2]:.3f}), "
            f"final_right=({final_right[0]:.3f},{final_right[1]:.3f},{final_right[2]:.3f}), "
            f"lift_clearance={self.claw_lift_clearance_m:.3f}"
        )
        write_pallet_place_diagnostic(
            "place_action_points",
            {
                "label": self.config_label,
                "strategy": strategy,
                "source": action_point_source,
                "robot_pose_map": current_pose,
                "height_reference_map_z": height_reference_z,
                "height_reference_source": height_reference_source,
                "target_claw_z_base": float(target_claw_z),
                "current_claw_midpoint_base": current_center,
                "pre_box_center_base": pre_center,
                "final_box_center_base": final_center,
                "pre_left_claw_base": pre_left,
                "pre_right_claw_base": pre_right,
                "final_left_claw_base": final_left,
                "final_right_claw_base": final_right,
            },
        )
        return Status.SUCCESS

    def _read_strategy(self):
        if self.strategy_key and self.blackboard.exists(self.strategy_key):
            return str(self.blackboard.get(self.strategy_key))
        return "<missing>"

    def _read_pose(self, key, required=False):
        if not key or not self.blackboard.exists(key):
            if required:
                self.ros_node.get_logger().error(f"[{self.config_label}] blackboard 缺少 pose: key={key}")
            return None
        value = self.blackboard.get(key)
        if not isinstance(value, dict):
            if required:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] pose 必须是 dict: key={key}, value={value!r}"
                )
            return None
        try:
            pose = dict(value)
            pose["x"] = float(pose["x"])
            pose["y"] = float(pose["y"])
            pose["z"] = float(pose.get("z", 0.0))
            pose["yaw"] = float(pose.get("yaw", 0.0))
            return pose
        except (KeyError, TypeError, ValueError) as exc:
            if required:
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] pose 缺少有效 x/y/z/yaw: "
                    f"key={key}, value={value!r}, error={exc}"
                )
            return None

    def _get_current_claw_pair(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            return None

        left = self._current_claw_point(services.arm_controller, "left")
        right = self._current_claw_point(services.arm_controller, "right")
        if left is None or right is None:
            return None
        return left, right

    def _current_claw_point(self, arm_controller, side):
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

    def _build_action_points_from_captured_grasp(
        self,
        final_box_pose,
        pre_box_pose,
        current_right_minus_left_base,
        height_reference_z,
    ):
        """用闭爪时锁定的真实抓取关系反算左右夹爪目标。

        抓取记录以左右爪中点为局部原点：x 沿左右夹爪连线，y 为其水平法向，z 为
        局部竖直。这里保留左右爪相对箱心的完整三维关系，再按规划箱体 yaw 放到目标
        箱心；因此不同轮次的实际抓取高度不会再被固定 ``box_size_z`` 覆盖。
        """
        if not self.use_held_box_grasp_transform or not self.held_box_transform_key:
            return None
        if not self.blackboard.exists(self.held_box_transform_key):
            return None

        try:
            transform = self.blackboard.get(self.held_box_transform_key)
            offset = transform["offset_local"]
            midpoint = self._dict_to_point(transform["captured_claw_midpoint_base"])
            left = self._dict_to_point(transform["captured_left_claw_base"])
            right = self._dict_to_point(transform["captured_right_claw_base"])
            axis_x = self._dict_to_point(transform["captured_axis_x_base"])
            center_offset_local = np.array(
                [float(offset["x"]), float(offset["y"]), float(offset["z"])], dtype=float
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

        if not all(
            np.all(np.isfinite(value))
            for value in (midpoint, left, right, axis_x, center_offset_local)
        ):
            return None
        axis_x[2] = 0.0
        axis_norm = float(np.linalg.norm(axis_x))
        if axis_norm < 1e-6:
            return None
        axis_x /= axis_norm
        axis_y = np.array([-axis_x[1], axis_x[0], 0.0], dtype=float)
        axis_z = np.array([0.0, 0.0, 1.0], dtype=float)
        left_local = self._project_to_local(left - midpoint, axis_x, axis_y, axis_z)
        right_local = self._project_to_local(right - midpoint, axis_x, axis_y, axis_z)

        base_from_map = self._get_base_from_map_matrix(height_reference_z)
        if base_from_map is None:
            return None
        final_center = self._transform_map_point_to_base(final_box_pose, base_from_map)
        pre_center = self._transform_map_point_to_base(pre_box_pose, base_from_map)
        if final_center is None or pre_center is None:
            return None

        final_left, final_right = self._target_claw_pair_from_box_center(
            final_center,
            final_box_pose["yaw"],
            center_offset_local,
            left_local,
            right_local,
            base_from_map,
            current_right_minus_left_base,
        )
        pre_left, pre_right = self._target_claw_pair_from_box_center(
            pre_center,
            pre_box_pose["yaw"],
            center_offset_local,
            left_local,
            right_local,
            base_from_map,
            current_right_minus_left_base,
        )
        if any(value is None for value in (final_left, final_right, pre_left, pre_right)):
            return None

        self.ros_node.get_logger().info(
            f"\033[1;97;46m[{self.config_label}] 使用已锁定抓取偏移刷新放箱动作点: "
            f"offset_local=({center_offset_local[0]:+.3f},{center_offset_local[1]:+.3f},"
            f"{center_offset_local[2]:+.3f}), "
            f"final_box_base=({final_center[0]:.3f},{final_center[1]:.3f},{final_center[2]:.3f}), "
            f"final_left_z={final_left[2]:.3f}, final_right_z={final_right[2]:.3f}\033[0m"
        )
        return pre_center, final_center, pre_left, pre_right, final_left, final_right

    @staticmethod
    def _dict_to_point(raw):
        return np.array([float(raw["x"]), float(raw["y"]), float(raw["z"])], dtype=float)

    @staticmethod
    def _project_to_local(vector, axis_x, axis_y, axis_z):
        return np.array(
            [
                float(np.dot(vector, axis_x)),
                float(np.dot(vector, axis_y)),
                float(np.dot(vector, axis_z)),
            ],
            dtype=float,
        )

    def _target_claw_pair_from_box_center(
        self,
        box_center_base,
        box_yaw_deg,
        center_offset_local,
        left_local,
        right_local,
        base_from_map,
        current_right_minus_left_base,
    ):
        yaw_rad = np.deg2rad(float(box_yaw_deg))
        map_axis_x = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0], dtype=float)
        map_axis_y = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0], dtype=float)
        map_axis_z = np.array([0.0, 0.0, 1.0], dtype=float)
        rotation_base_from_map = base_from_map[:3, :3]
        rotation_map_from_base = rotation_base_from_map.T

        # 箱体 yaw 只描述无方向的轴，yaw 与 yaw+180° 是同一箱体朝向。选择和当前
        # “右爪 - 左爪”方向一致的一支，避免机器人从垛盘另一侧接近时左右夹爪交叉。
        preferred_axis_map = rotation_map_from_base.dot(current_right_minus_left_base)
        preferred_axis_map[2] = 0.0
        if float(np.dot(map_axis_x[:2], preferred_axis_map[:2])) < 0.0:
            map_axis_x *= -1.0
            map_axis_y *= -1.0
        axis_x = rotation_base_from_map.dot(map_axis_x)
        axis_y = rotation_base_from_map.dot(map_axis_y)
        axis_z = rotation_base_from_map.dot(map_axis_z)
        midpoint = box_center_base - self._local_to_base_vector(
            center_offset_local, axis_x, axis_y, axis_z
        )
        left = midpoint + self._local_to_base_vector(left_local, axis_x, axis_y, axis_z)
        right = midpoint + self._local_to_base_vector(right_local, axis_x, axis_y, axis_z)
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return None, None
        return left, right

    @staticmethod
    def _local_to_base_vector(local, axis_x, axis_y, axis_z):
        return local[0] * axis_x + local[1] * axis_y + local[2] * axis_z

    def _get_base_from_map_matrix(self, height_reference_z):
        """返回 base_link <- map 的完整 3D 矩阵，与 FP 诊断保持同一 odom 链路。"""
        odom_msg = self.odom_transformer.get_latest_odom()
        if odom_msg is None:
            return None
        try:
            position = odom_msg.pose.pose.position
            orientation = odom_msg.pose.pose.orientation
            # 仅替换 map 中的平移 z；平面 x/y 和完整四元数仍使用当前 odom。这样
            # ``lock_on_first_use`` 不会影响导航站位或箱体朝向，只隔离地面上的 z 漂移。
            map_from_base = tf_trans.concatenate_matrices(
                tf_trans.translation_matrix([position.x, position.y, height_reference_z]),
                tf_trans.quaternion_matrix(
                    [orientation.x, orientation.y, orientation.z, orientation.w]
                ),
            )
            return tf_trans.inverse_matrix(map_from_base)
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _transform_map_point_to_base(box_pose, base_from_map):
        try:
            point = base_from_map.dot(
                [float(box_pose["x"]), float(box_pose["y"]), float(box_pose["z"]), 1.0]
            )
        except (KeyError, TypeError, ValueError):
            return None
        result = np.array(point[:3], dtype=float)
        return result if np.all(np.isfinite(result)) else None

    def _box_center_map_to_base_legacy(self, box_pose, current_pose, target_claw_z):
        """旧版 2D + 固定箱高回退；新的主路径不应使用这个近似。"""
        from tree.runtime.http.move_and_grab_flow import Pose2D, transform_global_point_to_base

        current_pose_2d = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        center_xy = transform_global_point_to_base(
            current_pose_2d,
            float(box_pose["x"]),
            float(box_pose["y"]),
        )
        return np.array(
            [
                float(center_xy["x"]),
                float(center_xy["y"]),
                float(target_claw_z),
            ],
            dtype=float,
        )

    def _resolve_target_claw_z(self, current_pose, height_reference_z):
        place_plane_height = None
        if self.place_plane_height_key and self.blackboard.exists(self.place_plane_height_key):
            try:
                place_plane_height = float(self.blackboard.get(self.place_plane_height_key))
            except (TypeError, ValueError):
                place_plane_height = None
        if place_plane_height is None:
            return float(height_reference_z) + self.box_size_z
        # place_plane_height 为 map 系平面高度；动作点需要 base_link 下夹爪 z。
        return float(place_plane_height) - float(height_reference_z) + self.box_size_z

    def _resolve_height_reference_map_z(self, current_pose):
        """取得用于 map→base_link 高度换算的 map z 基准。

        ``lock_on_first_use`` 专为平整码垛区设计：首个箱子在垛盘站位时记录底盘 z，
        第 2～4 箱继续采用该值，避免定位 z 漂移造成夹爪目标逐箱抬高/降低。
        """
        current_z = float(current_pose[2])
        if self.height_reference_mode == "current_odom":
            return current_z, "current_odom"
        if self.height_reference_mode == "fixed":
            return self.fixed_height_reference_map_z, "fixed"

        if self.height_reference_key and self.blackboard.exists(self.height_reference_key):
            try:
                reference_z = float(self.blackboard.get(self.height_reference_key))
                if np.isfinite(reference_z):
                    return reference_z, "locked_blackboard"
            except (AttributeError, TypeError, ValueError):
                pass

        if self.height_reference_key:
            self.blackboard.set(self.height_reference_key, current_z, overwrite=True)
        self.ros_node.get_logger().info(
            f"\033[1;97;46m[{self.config_label}] 已锁定平整地面码垛高度参考: "
            f"map_base_z={current_z:.3f}, key={self.height_reference_key}\033[0m"
        )
        return current_z, "locked_now"

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPalletPlaceActionPoints start: "
            f"strategy_key={self.strategy_key}, final_box_key={self.final_box_pose_key}, "
            f"pre_box_key={self.pre_box_pose_key}, plane_key={self.place_plane_height_key}, "
            f"held_box_transform_key={self.held_box_transform_key or '<disabled>'}, "
            f"use_held_box_grasp_transform={self.use_held_box_grasp_transform}, "
            f"height_reference_mode={self.height_reference_mode}, "
            f"height_reference_key={self.height_reference_key or '<disabled>'}, "
            f"box_size_z={self.box_size_z:.3f}, lift_clearance={self.claw_lift_clearance_m:.3f}"
        )
