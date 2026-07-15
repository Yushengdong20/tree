"""根据已选码垛策略，在当前站位刷新放箱动作点。

这个节点的定位很窄：只刷新动作点，不重新选择码垛 slot / 导航点 / 放箱策略。

为什么需要它：
- ComputeMoveBoxPalletPlaceStrategy 在抓箱后、导航到垛盘前执行一次，用 map 系下的
  slot 规划出本轮最终放置箱心、预落位箱心、放箱策略和导航站位。
- 但是 ArmsToPose 执行动作点使用的是 base_link 下的夹爪点。机器人导航到垛盘后，
  base_link 已经变了；如果继续使用导航前算出的 base_link 动作点，就会错位。
- 因此到达垛盘站位后，需要只把“已定好的 map 目标箱心”重新投影到当前 base_link，
  并结合当前左右夹爪实际点，刷新 pre/push/lift 等夹爪动作点。

这比第二次运行 ComputeMoveBoxPalletPlaceStrategy 更安全：
- 不会重新选择 direct / right_push_left / left_push_right；
- 不会因为 auto 逻辑覆盖第一次策略；
- 只更新执行动作必须依赖当前机器人姿态的 base_link 点。
"""

import numpy as np
import py_trees
from py_trees.common import Status

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from tree.runtime.http.move_and_grab_flow import Pose2D, transform_global_point_to_base
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

        left_claw, right_claw = claw_pair
        current_center = (left_claw + right_claw) * 0.5
        target_claw_z = self._resolve_target_claw_z(current_pose)
        final_center = self._box_center_map_to_base(final_box_pose, current_pose, target_claw_z)
        pre_center = self._box_center_map_to_base(pre_box_pose, current_pose, target_claw_z)

        pre_delta = pre_center - current_center
        final_delta = final_center - current_center
        pre_left = left_claw + pre_delta
        pre_right = right_claw + pre_delta
        final_left = left_claw + final_delta
        final_right = right_claw + final_delta
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
            f"current_center=({current_center[0]:.3f},{current_center[1]:.3f},{current_center[2]:.3f}), "
            f"pre_center=({pre_center[0]:.3f},{pre_center[1]:.3f},{pre_center[2]:.3f}), "
            f"final_center=({final_center[0]:.3f},{final_center[1]:.3f},{final_center[2]:.3f}), "
            f"pre_left=({pre_left[0]:.3f},{pre_left[1]:.3f},{pre_left[2]:.3f}), "
            f"pre_right=({pre_right[0]:.3f},{pre_right[1]:.3f},{pre_right[2]:.3f}), "
            f"final_left=({final_left[0]:.3f},{final_left[1]:.3f},{final_left[2]:.3f}), "
            f"final_right=({final_right[0]:.3f},{final_right[1]:.3f},{final_right[2]:.3f}), "
            f"lift_clearance={self.claw_lift_clearance_m:.3f}"
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

    def _box_center_map_to_base(self, box_pose, current_pose, target_claw_z):
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

    def _resolve_target_claw_z(self, current_pose):
        place_plane_height = None
        if self.place_plane_height_key and self.blackboard.exists(self.place_plane_height_key):
            try:
                place_plane_height = float(self.blackboard.get(self.place_plane_height_key))
            except (TypeError, ValueError):
                place_plane_height = None
        if place_plane_height is None:
            return float(current_pose[2]) + self.box_size_z
        # place_plane_height 为 map 系平面高度；动作点需要 base_link 下夹爪 z。
        return float(place_plane_height) - float(current_pose[2]) + self.box_size_z

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPalletPlaceActionPoints start: "
            f"strategy_key={self.strategy_key}, final_box_key={self.final_box_pose_key}, "
            f"pre_box_key={self.pre_box_pose_key}, plane_key={self.place_plane_height_key}, "
            f"box_size_z={self.box_size_z:.3f}, lift_clearance={self.claw_lift_clearance_m:.3f}"
        )
