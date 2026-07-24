"""右臂抓取目标位姿计算。"""

import numpy as np

from tree.utils.geometry import matrix_to_xyz_ypr


# 关键步骤：抓取算法坐标系到 SDK 末端坐标系的姿态修正，等价于绕 x 轴旋转 180 度。
GRASP_POSE_TO_SDK_EEF_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def build_grasp_target(grasp_pose, target_from_source, offset_m, rotate_z_180=False):
    """构造通用抓取目标，返回 [x, y, z, yaw, pitch, roll] 和 4x4 位姿矩阵。"""
    offset = np.eye(4)
    offset[2, 3] = -offset_m
    source_target = grasp_pose @ offset @ GRASP_POSE_TO_SDK_EEF_TRANSFORM

    target_pose = target_from_source @ source_target
    if rotate_z_180:
        rotate_z_180_transform = np.eye(4)
        rotate_z_180_transform[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        target_pose = target_pose @ rotate_z_180_transform

    x_axis_offset = 0.01 * target_pose[:3, 0]
    positive_translation = target_pose[:3, 3] + x_axis_offset
    negative_translation = target_pose[:3, 3] - x_axis_offset
    if np.linalg.norm(positive_translation) < np.linalg.norm(negative_translation):
        target_pose[:3, 3] = positive_translation
    else:
        target_pose[:3, 3] = negative_translation

    return matrix_to_xyz_ypr(target_pose), target_pose
