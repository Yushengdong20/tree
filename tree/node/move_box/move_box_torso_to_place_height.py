"""按放置平面高度调整 move_box 躯干姿态。

参考业务代码：
/home/ysd/Desktop/leju/leju_wbc/src/kuavo_humanoid_sdk/move_box/place_procedure.py
PlaceProcedure.adjust_torso_to_place_height()
"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxTorsoToPlaceHeight(TimedMockAction):
    """根据放置平面高度动态计算并发布躯干目标姿态。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.place_plane_height = float(params.get("place_plane_height", ros_node.get_param("place_plane_height", 0.0)))
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
        self.wait_done = self._to_bool(params.get("wait_done", True))
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "torso_controller"):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] services 或 torso_controller 缺失: key={self.services_key}"
            )
            return Status.FAILURE
        if self.should_skip_torso_motion():
            self.log_skip_torso_motion()
            return Status.SUCCESS

        target_pose = list(services.torso_controller.current_pose)
        if len(target_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前躯干位姿长度异常: {target_pose}"
            )
            return Status.FAILURE

        target_z = min(
            max(self.place_plane_height + self.height_offset, self.min_height),
            self.max_height,
        )
        target_pose[0] = self.torso_x
        target_pose[2] = target_z

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 根据放置平面调整躯干: "
            f"plane_z={self.place_plane_height:.3f}, torso_x={target_pose[0]:.3f}, "
            f"torso_z={target_pose[2]:.3f}, wait_done={self.wait_done}"
        )
        ok = services.torso_controller.move_to_pose(target_pose, wait_done=self.wait_done)
        return Status.SUCCESS if ok else Status.FAILURE

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveBoxTorsoToPlaceHeight start: "
            f"place_plane_height={self.place_plane_height:.3f}"
        )
