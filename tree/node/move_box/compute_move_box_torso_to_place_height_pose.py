"""根据放置平面高度计算躯干目标位姿，并写入 blackboard。

本节点只负责计算，不直接控制腰部；后续由 MoveTorsoPose 读取 pose_key 执行。
计算逻辑保留原放箱流程的躯干高度约束。
"""

import py_trees
from py_trees.common import Status

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from ..base import TimedMockAction


class ComputeMoveBoxTorsoToPlaceHeightPose(TimedMockAction):
    """根据放置平面高度生成放箱前躯干目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.place_plane_height = float(params.get("place_plane_height", ros_node.get_param("place_plane_height", 0.0)))
        self.place_plane_height_key = str(params.get("place_plane_height_key", "")).strip()
        self.place_plane_frame = str(params.get("place_plane_frame", "base_link")).strip() or "base_link"
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.height_offset = float(
            params.get("place_torso_height_offset", ros_node.get_param("place_torso_height_offset", 0.4))
        )
        self.min_height = float(
            params.get("place_torso_min_height", ros_node.get_param("place_torso_min_height", 0.6))
        )
        self.max_height = float(
            params.get("place_torso_max_height", ros_node.get_param("place_torso_max_height", 1.4))
        )
        self.torso_x = float(params.get("place_torso_x", ros_node.get_param("place_torso_x", 0.5)))
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_place_torso_pose")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        if self.place_plane_height_key:
            self.blackboard.register_key(key=self.place_plane_height_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.target_pose_key, access=py_trees.common.Access.WRITE)
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame="base_link",
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "torso_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 或 torso_controller 缺失: key={self.services_key}"
            )
            return Status.FAILURE

        target_pose = list(services.torso_controller.current_pose)
        if len(target_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前躯干位姿长度异常: {target_pose}"
            )
            return Status.FAILURE

        place_plane_height = self._get_place_plane_height()
        if place_plane_height is None:
            return Status.FAILURE
        place_plane_height_base = self._resolve_place_plane_height_in_base(place_plane_height)
        if place_plane_height_base is None:
            return Status.FAILURE

        target_z = min(
            max(place_plane_height_base + self.height_offset, self.min_height),
            self.max_height,
        )
        target_pose[0] = self.torso_x
        target_pose[2] = target_z
        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算放箱前躯干目标: "
            f"plane_z={place_plane_height:.3f}({self.place_plane_frame}), "
            f"plane_z_base={place_plane_height_base:.3f}, pose={target_pose}, "
            f"key={self.target_pose_key}"
        )
        return Status.SUCCESS

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
            f"[{self.config_label}] ComputeMoveBoxTorsoToPlaceHeightPose start: "
            f"place_plane_height={self.place_plane_height:.3f}, "
            f"place_plane_height_key={self.place_plane_height_key or '<static>'}, "
            f"place_plane_frame={self.place_plane_frame}, "
            f"target_pose_key={self.target_pose_key}"
        )
