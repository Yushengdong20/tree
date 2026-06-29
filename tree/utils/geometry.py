#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""move_box 运行时通用工具函数。"""

import math
import time

import numpy as np
import rospy
import tf
import tf.transformations as tf_trans
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


_ODOM_POSE_TRANSFORMERS = {}


class OdomPoseTransformer:
    """缓存 odom 位姿，并提供 base/map 坐标转换能力。"""

    def __init__(
        self,
        ros_node,
        odom_topic="melon_odom",
        target_frame="map",
        base_frame="base_link",
        queue_size=10,
    ):
        self.ros_node = ros_node
        self.odom_topic = str(odom_topic).strip()
        self.target_frame = str(target_frame).strip()
        self.base_frame = str(base_frame).strip()
        self._latest_odom = None
        self._latest_odom_time = None

        # 关键步骤：odom 订阅集中在工具类里，避免每个 node 重复维护订阅和缓存逻辑。
        self._odom_subscriber = self.ros_node.create_message_subscription(
            self.odom_topic,
            Odometry,
            self._on_odom,
            queue_size=queue_size,
        )

    def _on_odom(self, message):
        """缓存最新 odom，供行为树节点在 update 中读取。"""
        self._latest_odom = message
        self._latest_odom_time = time.monotonic()

    def get_latest_odom(self):
        """返回最近一次收到的 odom 消息。"""
        return self._latest_odom

    def get_current_pose(self):
        """返回当前底盘 map/odom 位姿: (x, y, z, yaw_deg)。"""
        if self._latest_odom is None:
            return None

        position = self._latest_odom.pose.pose.position
        orientation = self._latest_odom.pose.pose.orientation
        return (
            float(position.x),
            float(position.y),
            float(position.z),
            self._quaternion_to_yaw_deg(orientation),
        )

    def get_latest_odom_age_sec(self):
        """返回最新 odom 距当前的时间，尚未收到时返回 None。"""
        if self._latest_odom_time is None:
            return None
        return time.monotonic() - self._latest_odom_time

    def transform_point_to_map(self, tf_listener, point_xyz, source_frame, timeout=0.2):
        """先把 source_frame 点转到 base_frame，再用 odom 位姿转到 map/odom。"""
        return transform_point_to_map_via_base_and_odom(
            tf_listener,
            self.ros_node,
            self._latest_odom,
            point_xyz,
            source_frame=source_frame,
            base_frame=self.base_frame,
            target_frame=self.target_frame,
            timeout=timeout,
        )

    def transform_point_to_map_with_pose2d_fallback(
        self,
        tf_listener,
        point_xyz,
        source_frame,
        fallback_pose,
        timeout=0.2,
    ):
        """优先使用 3D odom 转换，失败时按底盘 2D yaw 回退。"""
        try:
            return self.transform_point_to_map(
                tf_listener,
                point_xyz,
                source_frame=source_frame,
                timeout=timeout,
            ), None
        except Exception as exc:
            if fallback_pose is None:
                raise
            # 关键步骤：回退路径只对 x/y 做 2D map 转换，z 沿用 base_link 高度近似值。
            return transform_base_point_to_map_with_pose2d(fallback_pose, point_xyz), exc

    def transform_base_point_to_map_with_pose2d(self, base_position, fallback_pose):
        """按底盘 2D yaw 将 base_link 下的点近似转换到 map。"""
        return transform_base_point_to_map_with_pose2d(fallback_pose, base_position)

    @staticmethod
    def _quaternion_to_yaw_deg(orientation):
        """把 odom 四元数转换为底盘 yaw，单位 deg。"""
        quaternion = [
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        ]
        _, _, yaw = tf_trans.euler_from_quaternion(quaternion)
        return math.degrees(yaw)


def get_odom_pose_transformer(
    ros_node,
    odom_topic="melon_odom",
    target_frame="map",
    base_frame="base_link",
    queue_size=10,
):
    """返回共享 odom 位姿转换器，同一 ros_node/topic/frame 组合只订阅一次。"""
    key = (
        id(ros_node),
        str(odom_topic).strip(),
        str(target_frame).strip(),
        str(base_frame).strip(),
    )
    transformer = _ODOM_POSE_TRANSFORMERS.get(key)
    if transformer is None:
        transformer = OdomPoseTransformer(
            ros_node,
            odom_topic=odom_topic,
            target_frame=target_frame,
            base_frame=base_frame,
            queue_size=queue_size,
        )
        _ODOM_POSE_TRANSFORMERS[key] = transformer
    return transformer


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


def transform_point(tf_listener, ros_node, point_xyz, source_frame, target_frame, timeout=0.2):
    """等待 TF 并将三维点转换到目标坐标系。"""
    if source_frame == target_frame:
        return {
            "x": float(point_xyz["x"]),
            "y": float(point_xyz["y"]),
            "z": float(point_xyz.get("z", 0.0)),
        }
    if tf_listener is None:
        raise RuntimeError("缺少 tf_listener，无法执行 source -> base_link 点转换")

    point_msg = PointStamped()
    point_msg.header.stamp = ros_node.zero_time()
    point_msg.header.frame_id = source_frame
    point_msg.point.x = float(point_xyz["x"])
    point_msg.point.y = float(point_xyz["y"])
    point_msg.point.z = float(point_xyz.get("z", 0.0))

    tf_listener.waitForTransform(
        target_frame,
        source_frame,
        point_msg.header.stamp,
        ros_node.duration(timeout),
    )
    transformed = tf_listener.transformPoint(target_frame, point_msg)
    return {
        "x": float(transformed.point.x),
        "y": float(transformed.point.y),
        "z": float(transformed.point.z),
    }


def transform_base_point_to_map_with_odom(
    odom_msg,
    base_position,
    target_frame="map",
    base_frame="base_link",
):
    """使用 odom 中的 base_link 位姿，将 base_link 三维点转换到 map/odom 坐标系。"""
    odom_frame = getattr(getattr(odom_msg, "header", None), "frame_id", "")
    if target_frame and odom_frame and odom_frame != target_frame:
        raise RuntimeError(
            "odom frame_id=%s 与目标 frame=%s 不一致" % (odom_frame, target_frame)
        )
    odom_position = odom_msg.pose.pose.position
    odom_orientation = odom_msg.pose.pose.orientation
    transform_matrix = tf_trans.concatenate_matrices(
        tf_trans.translation_matrix(
            [
                float(odom_position.x),
                float(odom_position.y),
                float(odom_position.z),
            ]
        ),
        tf_trans.quaternion_matrix(
            [
                float(odom_orientation.x),
                float(odom_orientation.y),
                float(odom_orientation.z),
                float(odom_orientation.w),
            ]
        ),
    )
    # 关键步骤：odom.pose 表示 base_link 在 map/odom 下的位姿，这里用完整 3D 位姿做 base -> map。
    transformed = transform_matrix.dot(
        [
            float(base_position["x"]),
            float(base_position["y"]),
            float(base_position.get("z", 0.0)),
            1.0,
        ]
    )
    return {
        "x": float(transformed[0]),
        "y": float(transformed[1]),
        "z": float(transformed[2]),
    }


def transform_base_point_to_map_with_pose2d(current_pose, base_position):
    """使用底盘二维位姿，将 base_link 点近似转换到 map 坐标。"""
    yaw_rad = math.radians(float(current_pose.yaw))
    base_x = float(base_position["x"])
    base_y = float(base_position["y"])
    return {
        "x": float(current_pose.x) + math.cos(yaw_rad) * base_x - math.sin(yaw_rad) * base_y,
        "y": float(current_pose.y) + math.sin(yaw_rad) * base_x + math.cos(yaw_rad) * base_y,
        "z": float(base_position.get("z", 0.0)),
    }


def transform_point_to_map_via_base_and_odom(
    tf_listener,
    ros_node,
    odom_msg,
    point_xyz,
    source_frame,
    base_frame="base_link",
    target_frame="map",
    timeout=0.2,
):
    """先把 source_frame 点转到 base_frame，再用 odom 位姿转到 map/odom 坐标系。"""
    if odom_msg is None:
        raise RuntimeError("缺少 odom 数据，无法执行 source -> base -> map 三维转换")

    # 关键步骤：只查询 source -> base_link 这一段 TF，避免直接查询 map <- source 完整 TF 链。
    base_position = transform_point(
        tf_listener,
        ros_node,
        point_xyz,
        source_frame,
        base_frame,
        timeout=timeout,
    )
    return transform_base_point_to_map_with_odom(
        odom_msg,
        base_position,
        target_frame=target_frame,
        base_frame=base_frame,
    )


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
