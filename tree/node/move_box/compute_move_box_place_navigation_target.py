"""根据箱子底部中心 C 反算放置导航目标。"""

import math

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeMoveBoxPlaceNavigationTarget(TimedMockAction):
    """用 C 点和站位距离计算机器人最终放置导航位姿。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        """
        初始化节点参数。
        :param name: 节点类型名。
        :param config_label: JSON 中的节点展示名。
        :param ros_node: MercuryTree runner。
        :param params: JSON 参数包装器。
        """
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.place_bottom_center_key = str(
            params.get("place_bottom_center_key", "move_box_box_pose_place_center")
        ).strip()
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_place_navigation_target")
        ).strip()
        self.place_standoff_m = float(params.get("place_standoff_m", 0.5))
        self.blackboard.register_key(key=self.place_bottom_center_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.target_pose_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if not self.blackboard.exists(self.place_bottom_center_key):
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 缺少放置箱底中心 C: key={self.place_bottom_center_key}"
            )
            return Status.FAILURE

        try:
            place_pose = self._parse_pose(self.blackboard.get(self.place_bottom_center_key))
            yaw_rad = math.radians(place_pose["yaw"])
            # 关键步骤：机器人沿 C 的朝向反方向退 place_standoff_m，最终 yaw 与 C.yaw 对齐。
            target_pose = {
                "x": place_pose["x"] - math.cos(yaw_rad) * self.place_standoff_m,
                "y": place_pose["y"] - math.sin(yaw_rad) * self.place_standoff_m,
                "yaw": place_pose["yaw"],
            }
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.get_logger().error(f"[{self.config_label}] 计算放置导航目标失败: {exc}")
            return Status.FAILURE

        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算放置导航目标: "
            f"C=({place_pose['x']:.3f}, {place_pose['y']:.3f}, {place_pose['yaw']:.3f}), "
            f"standoff={self.place_standoff_m:.3f}, "
            f"target=({target_pose['x']:.3f}, {target_pose['y']:.3f}, {target_pose['yaw']:.3f})"
        )
        return Status.SUCCESS

    @staticmethod
    def _parse_pose(value):
        """解析 dict/list/tuple 形式的二维位姿。"""
        if isinstance(value, dict):
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "yaw": float(value.get("yaw", value.get("angle", 0.0))),
            }
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return {
                "x": float(value[0]),
                "y": float(value[1]),
                "yaw": float(value[2]) if len(value) >= 3 else 0.0,
            }
        raise ValueError(f"无法解析 C 点位姿: {value!r}")

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxPlaceNavigationTarget start: "
            f"place_bottom_center_key={self.place_bottom_center_key}, "
            f"target_pose_key={self.target_pose_key}, "
            f"place_standoff_m={self.place_standoff_m:.3f}"
        )
