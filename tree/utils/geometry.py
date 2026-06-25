#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""move_box 运行时通用工具函数。"""

import math

import numpy as np
import rospy
import tf
import tf.transformations as tf_trans
import tf2_ros
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


def transform_pose(tf_listener, pose_stamped, target_frame, timeout=0.5):
    """等待 TF 并将 PoseStamped 转换到目标坐标系。"""
    source_frame = pose_stamped.header.frame_id
    if source_frame == target_frame:
        return pose_stamped

    tf_listener.waitForTransform(
        target_frame,
        source_frame,
        rospy.Time(0),
        rospy.Duration(timeout),
    )
    pose_stamped.header.stamp = rospy.Time(0)
    return tf_listener.transformPose(target_frame, pose_stamped)


def normalize_vector(vector):
    """归一化三维向量。"""
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("marker 轴向长度过小，无法计算箱体边缘")
    return vector / norm


def normalize_quaternion(quaternion):
    """归一化四元数。"""
    quat = np.array(quaternion, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quat / norm


def xyz_ypr_to_pose_stamped(xyz_ypr, frame_id):
    """将 [x, y, z, yaw, pitch, roll] 转为 ROS PoseStamped。"""
    if len(xyz_ypr) != 6:
        raise ValueError("末端目标位姿长度必须为 6")

    pose_stamped = PoseStamped()
    pose_stamped.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    pose_stamped.pose.position.x = float(xyz_ypr[0])
    pose_stamped.pose.position.y = float(xyz_ypr[1])
    pose_stamped.pose.position.z = float(xyz_ypr[2])

    quat = tf_trans.quaternion_from_euler(
        math.radians(float(xyz_ypr[5])),
        math.radians(float(xyz_ypr[4])),
        math.radians(float(xyz_ypr[3])),
    )
    quat = normalize_quaternion(quat)
    pose_stamped.pose.orientation.x = float(quat[0])
    pose_stamped.pose.orientation.y = float(quat[1])
    pose_stamped.pose.orientation.z = float(quat[2])
    pose_stamped.pose.orientation.w = float(quat[3])
    return pose_stamped


def transform_xyz_ypr(tf_listener, xyz_ypr, source_frame, target_frame, timeout=0.5):
    """将 [x, y, z, yaw, pitch, roll] 从源坐标系转换到目标坐标系。"""
    pose_stamped = xyz_ypr_to_pose_stamped(xyz_ypr, source_frame)
    pose_stamped.header.stamp = rospy.Time(0)

    try:
        transformed_pose = transform_pose(
            tf_listener,
            pose_stamped,
            target_frame,
            timeout=timeout,
        )
    except (
        tf.LookupException,
        tf.ConnectivityException,
        tf.ExtrapolationException,
        tf2_ros.TransformException,
    ) as err:
        raise RuntimeError(
            "末端目标从 {} 转到 {} 失败: {}".format(source_frame, target_frame, err)
        )

    quat = [
        transformed_pose.pose.orientation.x,
        transformed_pose.pose.orientation.y,
        transformed_pose.pose.orientation.z,
        transformed_pose.pose.orientation.w,
    ]
    quat = normalize_quaternion(quat)
    roll, pitch, yaw = tf_trans.euler_from_quaternion(quat)
    return [
        float(transformed_pose.pose.position.x),
        float(transformed_pose.pose.position.y),
        float(transformed_pose.pose.position.z),
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    ]


def ypr_to_rotation_matrix(ypr):
    """根据 yaw/pitch/roll 生成目标姿态旋转矩阵。"""
    quat = tf_trans.quaternion_from_euler(
        math.radians(float(ypr[2])),
        math.radians(float(ypr[1])),
        math.radians(float(ypr[0])),
    )
    return tf_trans.quaternion_matrix(quat)[:3, :3]
