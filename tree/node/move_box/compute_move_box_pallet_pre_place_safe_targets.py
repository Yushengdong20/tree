"""在下腰前计算码垛推箱策略的高位安全预落位。

推箱放置不能直接在当前手中箱体位置下腰：即使低位 ``pre_place_box``
（黄色预落位）避开了邻箱，下腰过程也可能让箱体扫过邻箱。

本节点不重新选择放箱策略，只读取已有的黄色预落位，在高于同层邻箱的
安全高度将左右夹爪整体平移过去。可选地，紫色高位点还能向机器人
回收一小段，避免双臂完全伸直时躯干下降扫到邻箱。随后再下腰，并由
``ComputeMoveBoxPalletPlaceActionPoints`` 按最新实际夹爪位置刷新低位动作点。
"""

import math

import numpy as np
import py_trees
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.chassis_navigation import Pose2D
from tree.utils.geometry import transform_global_point_to_base
from tree.utils.geometry import ypr_to_rotation_matrix

from ..base import TimedMockAction


class ComputeMoveBoxPalletPrePlaceSafeTargets(TimedMockAction):
    """根据黄色预落位生成下腰前的高位横移夹爪点。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.strategy_key = str(params.get("strategy_key", "move_box_pallet_place_strategy")).strip()
        self.pre_box_pose_key = str(
            params.get("pre_box_pose_key", "move_box_pallet_place_pre_box_pose")
        ).strip()
        self.place_plane_height_key = str(
            params.get("place_plane_height_key", "move_box_pallet_stack_place_plane_height")
        ).strip()
        self.left_target_key = str(
            params.get("left_target_key", "move_box_pallet_safe_pre_place_left_claw_point")
        ).strip()
        self.right_target_key = str(
            params.get("right_target_key", "move_box_pallet_safe_pre_place_right_claw_point")
        ).strip()
        self.safe_box_pose_key = str(
            params.get("safe_box_pose_key", "move_box_pallet_safe_pre_place_box_pose")
        ).strip()
        self.requires_safe_pre_place_key = str(
            params.get("requires_safe_pre_place_key", "move_box_pallet_requires_safe_pre_place")
        ).strip()

        self.box_size_x = float(params.get("box_size_x", 0.60))
        self.box_size_y = float(params.get("box_size_y", 0.40))
        self.box_size_z = float(params.get("box_size_z", 0.34))
        self.pre_place_safe_clearance_m = max(
            float(params.get("pre_place_safe_clearance_m", 0.10)),
            0.0,
        )
        # 只作用于高位紫色预落位。沿“机器人 -> 预落位”的反方向回收，
        # 不改变低位黄色预落位和最终绿色放置箱心。
        self.pre_place_safe_retract_m = max(
            float(params.get("pre_place_safe_retract_m", 0.0)),
            0.0,
        )
        # 高位避障阶段通常只需把手中箱体横向挪到邻箱外侧。开启后锁住当前
        # base_link.x，防止把黄色低位预落位的前向坐标直接带到高位动作中，
        # 造成双臂前伸到极限或在下腰时扫到已有箱体。
        self.pre_place_safe_lateral_only = self._to_bool(
            params.get("pre_place_safe_lateral_only", False)
        )
        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/pallet_place_dynamic_estimate_markers")
        ).strip()

        for key in (
            self.services_key,
            self.strategy_key,
            self.pre_box_pose_key,
            self.place_plane_height_key,
        ):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)
        for key in (
            self.left_target_key,
            self.right_target_key,
            self.safe_box_pose_key,
            self.requires_safe_pre_place_key,
        ):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame="base_link",
        )
        self.visualization_pub = None
        if self.visualization_enabled and self.visualization_topic:
            self.visualization_pub = self.ros_node.create_publisher(
                self.visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )

    def update(self):
        strategy = self._read_strategy()
        requires_safe_pre_place = strategy in (
            "right_push_left_place",
            "left_push_right_place",
        )
        self.blackboard.set(
            self.requires_safe_pre_place_key,
            requires_safe_pre_place,
            overwrite=True,
        )

        pre_box_pose = self._read_pose(self.pre_box_pose_key)
        current_pose = self.odom_transformer.get_current_pose()
        claw_pair = self._get_current_claw_pair()
        if pre_box_pose is None or current_pose is None or claw_pair is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 无法计算高位安全预落位: "
                f"pre_box_ready={pre_box_pose is not None}, "
                f"odom_ready={current_pose is not None}, claw_pair_ready={claw_pair is not None}"
            )
            return Status.FAILURE

        left_claw, right_claw = claw_pair
        current_center = (left_claw + right_claw) * 0.5
        current_box_center_map_z = float(current_pose[2]) + float(current_center[2])
        place_plane_z = self._read_place_plane_height()
        # 同层邻箱顶面为 plane + box_h；让手中箱底面越过其顶面后再加余量，
        # 因此箱心最低应为 plane + 1.5 * box_h + clearance。
        minimum_safe_center_map_z = (
            place_plane_z + 1.5 * self.box_size_z + self.pre_place_safe_clearance_m
        )
        safe_center_map_z = max(current_box_center_map_z, minimum_safe_center_map_z)
        safe_box_pose = dict(pre_box_pose)
        safe_box_pose["z"] = float(safe_center_map_z)
        safe_box_pose["source"] = "high_pre_place_clearance"

        safe_center_base = self._map_center_to_base(safe_box_pose, current_pose)
        if safe_center_base is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 无法把高位安全预落位转换到 base_link"
            )
            return Status.FAILURE
        if self.pre_place_safe_lateral_only:
            # “左/右横移”以当前机器人坐标系定义：保持双爪中点的前后 x，
            # 只采用规划预落位的 y，并上提到安全 z。此模式下不能再使用
            # 径向回收，否则会重新引入 x 向位移。
            safe_center_base[0] = float(current_center[0])
        else:
            safe_center_base = self._retract_toward_robot(safe_center_base)
        safe_box_pose["x"], safe_box_pose["y"] = self._base_xy_to_map(
            safe_center_base[:2],
            current_pose,
        )
        delta = safe_center_base - current_center
        safe_left = left_claw + delta
        safe_right = right_claw + delta

        self.blackboard.set(self.left_target_key, safe_left, overwrite=True)
        self.blackboard.set(self.right_target_key, safe_right, overwrite=True)
        self.blackboard.set(self.safe_box_pose_key, safe_box_pose, overwrite=True)
        self._publish_visualization(safe_box_pose, requires_safe_pre_place)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 高位安全预落位已计算: strategy={strategy}, "
            f"requires_safe_pre_place={requires_safe_pre_place}, "
            f"pre_xy=({pre_box_pose['x']:.3f},{pre_box_pose['y']:.3f}), "
            f"current_center_map_z={current_box_center_map_z:.3f}, "
            f"safe_center_map_z={safe_center_map_z:.3f}, "
            f"minimum_safe_center_map_z={minimum_safe_center_map_z:.3f}, "
            f"clearance={self.pre_place_safe_clearance_m:.3f}, "
            f"retract={self.pre_place_safe_retract_m:.3f}, "
            f"lateral_only={self.pre_place_safe_lateral_only}, "
            f"left=({safe_left[0]:.3f},{safe_left[1]:.3f},{safe_left[2]:.3f}), "
            f"right=({safe_right[0]:.3f},{safe_right[1]:.3f},{safe_right[2]:.3f})"
        )
        return Status.SUCCESS

    def _read_strategy(self):
        if self.blackboard.exists(self.strategy_key):
            return str(self.blackboard.get(self.strategy_key))
        return "<missing>"

    def _read_pose(self, key):
        if not key or not self.blackboard.exists(key):
            return None
        value = self.blackboard.get(key)
        if not isinstance(value, dict):
            return None
        try:
            pose = dict(value)
            pose["x"] = float(pose["x"])
            pose["y"] = float(pose["y"])
            pose["z"] = float(pose.get("z", 0.0))
            pose["yaw"] = float(pose.get("yaw", 0.0))
            return pose
        except (KeyError, TypeError, ValueError):
            return None

    def _read_place_plane_height(self):
        if self.place_plane_height_key and self.blackboard.exists(self.place_plane_height_key):
            try:
                return float(self.blackboard.get(self.place_plane_height_key))
            except (TypeError, ValueError):
                pass
        return 0.0

    def _get_current_claw_pair(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        arm_controller = getattr(services, "arm_controller", None) if services is not None else None
        if arm_controller is None:
            return None
        left = self._current_claw_point(arm_controller, "left")
        right = self._current_claw_point(arm_controller, "right")
        if left is None or right is None:
            return None
        return left, right

    @staticmethod
    def _current_claw_point(arm_controller, side):
        current_pose = (
            arm_controller.get_current_end_effector_pose(side)
            if hasattr(arm_controller, "get_current_end_effector_pose")
            else None
        )
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
        return np.array(current_pose[:3], dtype=float) + ypr_to_rotation_matrix(
            current_pose[3:6]
        ).dot(translation)

    @staticmethod
    def _map_center_to_base(box_pose, current_pose):
        pose_2d = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        center_xy = transform_global_point_to_base(
            pose_2d,
            float(box_pose["x"]),
            float(box_pose["y"]),
        )
        return np.array(
            [
                float(center_xy["x"]),
                float(center_xy["y"]),
                float(box_pose["z"]) - float(current_pose[2]),
            ],
            dtype=float,
        )

    def _retract_toward_robot(self, center_base):
        """将高位预落位沿 base_link 水平径向向机器人回收。"""
        center = np.array(center_base, dtype=float)
        distance_xy = math.hypot(float(center[0]), float(center[1]))
        if self.pre_place_safe_retract_m <= 0.0 or distance_xy <= 1e-6:
            return center
        retract = min(self.pre_place_safe_retract_m, max(0.0, distance_xy - 0.05))
        center[0] -= center[0] / distance_xy * retract
        center[1] -= center[1] / distance_xy * retract
        return center

    @staticmethod
    def _base_xy_to_map(base_xy, current_pose):
        """把 base_link 平面点按当前 odom 位姿转换回 map，用于紫色标记。"""
        yaw_rad = math.radians(float(current_pose[3]))
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        return (
            float(current_pose[0]) + cos_yaw * float(base_xy[0]) - sin_yaw * float(base_xy[1]),
            float(current_pose[1]) + sin_yaw * float(base_xy[0]) + cos_yaw * float(base_xy[1]),
        )

    def _publish_visualization(self, safe_box_pose, required):
        if self.visualization_pub is None:
            return
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = self.ros_node.now()
        marker.ns = "safe_pre_place_box"
        marker.id = 900
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = safe_box_pose["x"]
        marker.pose.position.y = safe_box_pose["y"]
        marker.pose.position.z = safe_box_pose["z"]
        yaw_rad = math.radians(safe_box_pose["yaw"])
        marker.pose.orientation.z = math.sin(yaw_rad * 0.5)
        marker.pose.orientation.w = math.cos(yaw_rad * 0.5)
        marker.scale.x = self.box_size_x
        marker.scale.y = self.box_size_y
        marker.scale.z = self.box_size_z
        marker.color.r = 0.70
        marker.color.g = 0.35
        marker.color.b = 1.00
        marker.color.a = 0.38 if required else 0.16
        marker_array.markers.append(marker)

        text = Marker()
        text.header.frame_id = MAP_FRAME
        text.header.stamp = self.ros_node.now()
        text.ns = "safe_pre_place_text"
        text.id = 901
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = safe_box_pose["x"]
        text.pose.position.y = safe_box_pose["y"]
        text.pose.position.z = safe_box_pose["z"] + self.box_size_z * 0.5 + 0.10
        text.pose.orientation.w = 1.0
        text.scale.z = 0.09
        text.color.r = 0.80
        text.color.g = 0.45
        text.color.b = 1.00
        text.color.a = 1.0
        text.text = "SAFE PRE PLACE\n(high lateral move)" if required else "SAFE PRE PLACE (skip)"
        marker_array.markers.append(text)
        self.visualization_pub.publish(marker_array)

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPalletPrePlaceSafeTargets start: "
            f"strategy_key={self.strategy_key}, pre_box_key={self.pre_box_pose_key}, "
            f"box=({self.box_size_x:.3f},{self.box_size_y:.3f},{self.box_size_z:.3f}), "
            f"clearance={self.pre_place_safe_clearance_m:.3f}, "
            f"retract={self.pre_place_safe_retract_m:.3f}, "
            f"lateral_only={self.pre_place_safe_lateral_only}"
        )
