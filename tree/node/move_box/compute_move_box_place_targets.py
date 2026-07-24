"""计算 move_box 放置流程的双夹爪下降目标点。

参考业务代码：
/home/ysd/Desktop/leju/leju_wbc/src/kuavo_humanoid_sdk/move_box/place_procedure.py
PlaceProcedure.update_lower_claw_points()
"""

import numpy as np
import py_trees
from py_trees.common import Status

from tree.utils.chassis_navigation import Pose2D
from tree.utils.geometry import transform_global_point_to_base
from tree.utils.geometry import ypr_to_rotation_matrix

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class ComputeMoveBoxPlaceTargets(TimedMockAction):
    """基于当前夹爪位置生成放置下降目标。

    默认兼容旧放箱流程：保持当前双爪 x/y，只把双爪 z 降到放置高度。

    码垛流程可启用 align_xy_to_expected_box：
    - expected_box_pose_key 给出期望箱心 map 坐标；
    - 节点用当前 odom 把期望箱心反算到 base_link；
    - 再整体平移左右夹爪目标，让当前双爪中点对齐期望箱心。
    这样放箱不再只是“原地下降”，而是会向选定码垛槽位对齐。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.place_plane_height = float(params.get("place_plane_height", ros_node.get_param("place_plane_height", 0.0)))
        self.place_plane_height_key = str(params.get("place_plane_height_key", "")).strip()
        self.place_plane_frame = str(params.get("place_plane_frame", "base_link")).strip() or "base_link"
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.expected_box_pose_key = str(params.get("expected_box_pose_key", "")).strip()
        self.align_xy_to_expected_box = self._to_bool(params.get("align_xy_to_expected_box", False))
        self.box_size_z = float(params.get("box_size_z", ros_node.get_param("box_size_z", 0.34)))
        self.left_target_key = str(params.get("left_target_key", "move_box_place_left_lower_claw_point")).strip()
        self.right_target_key = str(params.get("right_target_key", "move_box_place_right_lower_claw_point")).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.place_plane_height_key:
            self.blackboard.register_key(key=self.place_plane_height_key, access=py_trees.common.Access.READ)
        if self.expected_box_pose_key:
            self.blackboard.register_key(key=self.expected_box_pose_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.left_target_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.right_target_key, access=py_trees.common.Access.WRITE)
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame="base_link",
        )

    def _current_claw_point(self, arm_controller, side):
        if side not in ("left", "right"):
            return None

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
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] {side} 当前实际末端 TF 无效，回退到上一次目标缓存"
            )

        if len(current_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] {side} 当前手臂位姿长度必须为 6: {current_pose}"
            )
            return None

        transform = arm_controller.lookup_end_effector_to_claw_transform(side)
        if transform is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] 获取 {side} 末端到夹爪 TF 失败")
            return None

        translation, _ = transform
        rotation = ypr_to_rotation_matrix(current_pose[3:6])
        end_effector_point = np.array(current_pose[:3], dtype=float)
        return end_effector_point + rotation.dot(translation)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 或 arm_controller 缺失: key={self.services_key}"
            )
            return Status.FAILURE

        arm_controller = services.arm_controller
        left_claw_point = self._current_claw_point(arm_controller, "left")
        right_claw_point = self._current_claw_point(arm_controller, "right")
        if left_claw_point is None or right_claw_point is None:
            return Status.FAILURE

        place_plane_height = self._get_place_plane_height()
        if place_plane_height is None:
            return Status.FAILURE
        place_plane_height_base = self._resolve_place_plane_height_in_base(place_plane_height)
        if place_plane_height_base is None:
            return Status.FAILURE

        target_claw_z = place_plane_height_base + self.box_size_z
        lower_left = np.array(left_claw_point, dtype=float)
        lower_right = np.array(right_claw_point, dtype=float)
        current_box_center_base = (lower_left + lower_right) * 0.5
        target_box_center_base = self._get_expected_box_center_base(target_claw_z)
        if self.align_xy_to_expected_box and target_box_center_base is not None:
            # 关键步骤：C 点只决定箱底中心 x/y；最终 z 必须严格由 H + box_size_z 决定。
            delta = np.array(
                [
                    target_box_center_base[0] - current_box_center_base[0],
                    target_box_center_base[1] - current_box_center_base[1],
                    0.0,
                ],
                dtype=float,
            )
            lower_left += delta
            lower_right += delta
            lower_left[2] = target_claw_z
            lower_right[2] = target_claw_z
            align_text = (
                f", align_expected_box=True, current_box_center_base="
                f"({current_box_center_base[0]:.3f}, {current_box_center_base[1]:.3f}, {current_box_center_base[2]:.3f}), "
                f"target_box_center_base=({target_box_center_base[0]:.3f}, {target_box_center_base[1]:.3f}, "
                f"{target_box_center_base[2]:.3f}), "
                f"delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f})"
            )
        else:
            lower_left[2] = target_claw_z
            lower_right[2] = target_claw_z
            align_text = ", align_expected_box=False"

        self.blackboard.set(self.left_target_key, lower_left, overwrite=True)
        self.blackboard.set(self.right_target_key, lower_right, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算放置下降目标: "
            f"plane_z={place_plane_height:.3f}({self.place_plane_frame}), "
            f"plane_z_base={place_plane_height_base:.3f}, "
            f"box_size_z={self.box_size_z:.3f}, "
            f"target_claw_z={target_claw_z:.3f}{align_text}"
        )
        return Status.SUCCESS

    def _get_expected_box_center_base(self, target_claw_z):
        if not self.expected_box_pose_key:
            return None
        if not self.blackboard.exists(self.expected_box_pose_key):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 缺少 expected_box_pose_key，无法对齐目标箱心: "
                f"key={self.expected_box_pose_key}"
            )
            return None
        expected_pose = self.blackboard.get(self.expected_box_pose_key)
        if not isinstance(expected_pose, dict):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] expected_box_pose 必须是 dict: "
                f"key={self.expected_box_pose_key}, value={expected_pose!r}"
            )
            return None
        try:
            expected_map_x = float(expected_pose["x"])
            expected_map_y = float(expected_pose["y"])
        except (KeyError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] expected_box_pose 缺少有效 x/y: "
                f"key={self.expected_box_pose_key}, value={expected_pose!r}, error={exc}"
            )
            return None

        current_pose = self.odom_transformer.get_current_pose()
        if current_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 odom 后才能把 expected_box map x/y 转到 base_link: "
                f"odom_topic={self.odom_topic}"
            )
            return None
        current_pose_2d = Pose2D(
            x=float(current_pose[0]),
            y=float(current_pose[1]),
            yaw=float(current_pose[3]),
        )
        expected_base_xy = transform_global_point_to_base(
            current_pose_2d,
            expected_map_x,
            expected_map_y,
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 码垛目标箱心 map->base_link: "
            f"expected_map=({expected_map_x:.3f}, {expected_map_y:.3f}), "
            f"base_pose_map=({current_pose_2d.x:.3f}, {current_pose_2d.y:.3f}, "
            f"{current_pose_2d.yaw:.2f}deg), "
            f"expected_base=({float(expected_base_xy['x']):.3f}, "
            f"{float(expected_base_xy['y']):.3f}, {float(target_claw_z):.3f})"
        )
        return np.array(
            [
                float(expected_base_xy["x"]),
                float(expected_base_xy["y"]),
                float(target_claw_z),
            ],
            dtype=float,
        )

    def _get_place_plane_height(self):
        if not self.place_plane_height_key:
            return self.place_plane_height

        if not self.blackboard.exists(self.place_plane_height_key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] blackboard 缺少动态放置平面高度: "
                f"key={self.place_plane_height_key}"
            )
            return None

        try:
            return float(self.blackboard.get(self.place_plane_height_key))
        except (TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 动态放置平面高度无效: "
                f"key={self.place_plane_height_key}, value={self.blackboard.get(self.place_plane_height_key)!r}, "
                f"error={exc}"
            )
            return None

    def _resolve_place_plane_height_in_base(self, place_plane_height):
        """把配置/黑板中的放置平面高度转换成 base_link 下的 z。

        旧放箱流程里的 place_plane_height 本来就是 base_link 高度，所以默认不变。
        码垛流程给的是 map 系垛盘高度，需要减去当前 base_link 在 map 下的 z。
        """
        frame = self.place_plane_frame.lower()
        if frame in ("base", "base_link", ""):
            return float(place_plane_height)

        if frame in ("map", MAP_FRAME.lower()):
            current_pose = self.odom_transformer.get_current_pose()
            if current_pose is None:
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 等待 odom 后才能把 map 放置高度转换到 base_link: "
                    f"odom_topic={self.odom_topic}"
                )
                return None
            base_map_z = float(current_pose[2])
            return float(place_plane_height) - base_map_z

        self.ros_node.get_logger().error(
            f"[{self.config_label}] 不支持的 place_plane_frame={self.place_plane_frame!r}，"
            "仅支持 base_link/map"
        )
        return None

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPlaceTargets start: "
            f"place_plane_height={self.place_plane_height:.3f}, "
            f"place_plane_height_key={self.place_plane_height_key or '<static>'}, "
            f"place_plane_frame={self.place_plane_frame}, "
            f"expected_box_pose_key={self.expected_box_pose_key or '<disabled>'}, "
            f"align_xy_to_expected_box={self.align_xy_to_expected_box}, "
            f"box_size_z={self.box_size_z:.3f}"
        )
