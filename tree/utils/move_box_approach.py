"""move_box 箱体靠近目标计算工具函数。

这里不访问任何 HTTP/ROS 服务，只根据视觉结果计算底盘靠近目标位姿。
"""

import math
from typing import Dict

from tree.utils.chassis_navigation import Pose2D
from tree.utils.geometry import (
    normalize_angle_deg,
    normalize_axis_angle_rad,
    transform_base_point_to_global,
)


# YOLO 远距离检测后，底盘目标点距离箱子中心的默认距离，单位 m。
DEFAULT_YOLO_TARGET_DISTANCE_M = 1.5
# FoundationPose 精定位后，底盘最终抓取站位距离箱子中心的默认距离，单位 m。
DEFAULT_FP_TARGET_DISTANCE_M = 1.0


def build_yolo_approach_pose(
    current_pose: Pose2D,
    box_global_x: float,
    box_global_y: float,
    target_distance_m: float,
) -> Pose2D:
    """根据 YOLO 箱体 global 位置计算远距离粗靠近目标点。"""
    dx = box_global_x - current_pose.x
    dy = box_global_y - current_pose.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        raise RuntimeError("箱体中心与底盘当前位置重合，无法计算 YOLO 靠近目标")
    if distance <= target_distance_m:
        raise RuntimeError(
            "箱体距离 %.3fm 小于等于目标距离 %.3fm，无需继续 YOLO 靠近"
            % (distance, target_distance_m)
        )

    unit_x = dx / distance
    unit_y = dy / distance
    # 关键步骤：目标点位于机器人到箱子的连线上，并保持到箱子 target_distance_m 的距离。
    return Pose2D(
        x=box_global_x - unit_x * target_distance_m,
        y=box_global_y - unit_y * target_distance_m,
        yaw=normalize_angle_deg(math.degrees(math.atan2(unit_y, unit_x))),
    )


def build_fp_approach_pose(
    current_pose: Pose2D,
    box_center: Dict[str, float],
    front_axis: Dict[str, float],
    target_distance_m: float,
) -> Pose2D:
    """根据 FoundationPose 箱体中心和前向轴计算最终抓取站位。"""
    box_global_position = transform_base_point_to_global(
        current_pose,
        box_center["x"],
        box_center["y"],
    )
    axis_xy_norm = math.hypot(front_axis["x"], front_axis["y"])
    if axis_xy_norm < 1e-6:
        raise RuntimeError("FoundationPose frontAxis xy too small: %r" % front_axis)

    # 关键步骤：frontAxis 是 base_link 下的箱体前后轴，叠加当前底盘 global yaw 得到最终朝向。
    front_axis_angle_rad = normalize_axis_angle_rad(
        math.atan2(front_axis["y"], front_axis["x"])
    )
    target_yaw = normalize_angle_deg(
        current_pose.yaw + math.degrees(front_axis_angle_rad)
    )
    target_yaw_rad = math.radians(target_yaw)
    return Pose2D(
        x=box_global_position["x"] - math.cos(target_yaw_rad) * target_distance_m,
        y=box_global_position["y"] - math.sin(target_yaw_rad) * target_distance_m,
        yaw=target_yaw,
    )
