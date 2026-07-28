#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""行为树运行时通用几何工具函数。"""

import math
import time
import threading
from collections import deque

import numpy as np
import rospy
import tf
import tf.transformations as tf_trans
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from tree.constants import BASE_LINK_FRAME, CHASSIS_FRAME, MAP_FRAME, ODOM_POSE_TRANSFORMER_KEY


_ODOM_POSE_TRANSFORMERS = {}


def normalize_angle_deg(angle_deg):
    """把角度归一化到 [-180, 180) 区间。"""
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def normalize_axis_angle_rad(angle_rad):
    """把“轴向方向角”归一化到 [-pi/2, pi/2)。"""
    # 关键步骤：箱体长边等轴向目标正反向等价，按 180 度周期归一化。
    while angle_rad >= math.pi / 2.0:
        angle_rad -= math.pi
    while angle_rad < -math.pi / 2.0:
        angle_rad += math.pi
    return angle_rad


def transform_base_point_to_global(current_pose, base_x, base_y):
    """将 base_link 下二维点转换到 global/map 坐标系。"""
    yaw_rad = math.radians(current_pose.yaw)
    # 关键步骤：base_link 的 x/y 先按当前底盘 yaw 旋到 global，再平移到底盘 global 位置。
    return {
        "x": current_pose.x + math.cos(yaw_rad) * base_x - math.sin(yaw_rad) * base_y,
        "y": current_pose.y + math.sin(yaw_rad) * base_x + math.cos(yaw_rad) * base_y,
    }


def transform_global_point_to_base(current_pose, global_x, global_y):
    """将 global/map 坐标系下二维点转换到 base_link 坐标系。"""
    yaw_rad = math.radians(current_pose.yaw)
    dx = global_x - current_pose.x
    dy = global_y - current_pose.y
    # 关键步骤：先减去底盘 global 平移，再乘当前 yaw 的逆旋转，得到 base_link 坐标。
    return {
        "x": math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy,
        "y": -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy,
    }


class OdomPoseTransformer:
    """缓存 odom 位姿，并提供 base/map 坐标转换能力。"""

    def __init__(
        self,
        ros_node,
        odom_topic=CHASSIS_FRAME,
        target_frame=MAP_FRAME,
        base_frame=BASE_LINK_FRAME,
        queue_size=10,
        history_duration_sec=10.0,
    ):
        self.ros_node = ros_node
        self.odom_topic = str(odom_topic).strip()
        self.target_frame = str(target_frame).strip()
        self.base_frame = str(base_frame).strip()
        self._latest_odom = None
        self._latest_odom_time = None
        self.history_duration_sec = max(float(history_duration_sec), 1.0)
        self._odom_history = deque()
        self._odom_lock = threading.Lock()

        # 关键步骤：odom 订阅集中在工具类里，避免每个 node 重复维护订阅和缓存逻辑。
        self._odom_subscriber = self.ros_node.create_message_subscription(
            self.odom_topic,
            Odometry,
            self._on_odom,
            queue_size=queue_size,
        )

    def _on_odom(self, message):
        """缓存最新 odom，供行为树节点在 update 中读取。"""
        with self._odom_lock:
            self._latest_odom = message
            self._latest_odom_time = time.monotonic()
            stamp_sec = self._odom_stamp_to_seconds(message)
            self._odom_history.append((stamp_sec, message))
            min_stamp_sec = stamp_sec - self.history_duration_sec
            while len(self._odom_history) > 1 and self._odom_history[0][0] < min_stamp_sec:
                self._odom_history.popleft()

    def get_latest_odom(self):
        """返回最近一次收到的 odom 消息。"""
        with self._odom_lock:
            return self._latest_odom

    def get_nearest_odom_by_stamp_sec(self, stamp_sec):
        """返回时间上最接近指定秒数时间戳的 odom。"""
        if stamp_sec is None:
            return None

        target_sec = float(stamp_sec)
        nearest_msg = None
        nearest_delta = None
        # rospy 的 odom 回调和行为树 tick 在不同线程中运行。这里必须先复制快照，
        # 否则遍历 deque 时 _on_odom() 追加/裁剪历史会触发
        # RuntimeError: deque mutated during iteration。
        with self._odom_lock:
            history_snapshot = list(self._odom_history)
        if not history_snapshot:
            return None

        for history_stamp_sec, history_msg in history_snapshot:
            delta = abs(history_stamp_sec - target_sec)
            if nearest_delta is None or delta < nearest_delta:
                nearest_delta = delta
                nearest_msg = history_msg
        return nearest_msg

    def get_current_pose(self):
        """返回当前底盘 map/odom 位姿: (x, y, z, yaw_deg)。"""
        with self._odom_lock:
            latest_odom = self._latest_odom
        if latest_odom is None:
            return None

        position = latest_odom.pose.pose.position
        orientation = latest_odom.pose.pose.orientation
        return (
            float(position.x),
            float(position.y),
            float(position.z),
            self._quaternion_to_yaw_deg(orientation),
        )

    def get_latest_odom_age_sec(self):
        """返回最新 odom 距当前的时间，尚未收到时返回 None。"""
        with self._odom_lock:
            latest_odom_time = self._latest_odom_time
        if latest_odom_time is None:
            return None
        return time.monotonic() - latest_odom_time

    def transform_point_to_map(
        self,
        tf_listener,
        point_xyz,
        source_frame,
        timeout=0.2,
        odom_msg=None,
    ):
        """先把 source_frame 点转到 base_frame，再用 odom 位姿转到 map/odom。"""
        return transform_point_to_map_via_base_and_odom(
            tf_listener,
            self.ros_node,
            odom_msg if odom_msg is not None else self._latest_odom,
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
        odom_msg=None,
    ):
        """优先使用 3D odom 转换，失败时按底盘 2D yaw 回退。"""
        try:
            return self.transform_point_to_map(
                tf_listener,
                point_xyz,
                source_frame=source_frame,
                timeout=timeout,
                odom_msg=odom_msg,
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

    @staticmethod
    def _odom_stamp_to_seconds(odom_msg):
        header = getattr(odom_msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return 0.0
        if hasattr(stamp, "secs"):
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def get_odom_pose_transformer(
    ros_node,
    odom_topic=CHASSIS_FRAME,
    target_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
    queue_size=10,
    history_duration_sec=10.0,
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
            history_duration_sec=history_duration_sec,
        )
        _ODOM_POSE_TRANSFORMERS[key] = transformer
    return transformer


def get_shared_odom_pose_transformer(
    blackboard,
    ros_node,
    odom_topic=CHASSIS_FRAME,
    target_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
    queue_size=10,
    history_duration_sec=10.0,
    key=ODOM_POSE_TRANSFORMER_KEY,
):
    """优先从 blackboard 读取统一初始化的 odom 转换器，缺失时回退创建。

    新流程中，EnsureMoveBoxServices 会在 blackboard 写入一个共享
    OdomPoseTransformer。其它节点通过本函数读取它，避免每个节点看起来
    都在各自初始化 odom 订阅。

    为了兼容单测/老树，如果 blackboard 中没有共享对象，或请求的
    odom_topic/target_frame/base_frame 与共享对象不一致，则回退到原来的
    get_odom_pose_transformer()。原函数内部仍有全局缓存，同一组合只会订阅一次。
    """
    requested_topic = str(odom_topic).strip()
    requested_target = str(target_frame).strip()
    requested_base = str(base_frame).strip()
    requested_history = max(float(history_duration_sec), 1.0)

    transformer = None
    if blackboard is not None and key and blackboard.exists(key):
        candidate = blackboard.get(key)
        if (
            isinstance(candidate, OdomPoseTransformer)
            and candidate.odom_topic == requested_topic
            and candidate.target_frame == requested_target
            and candidate.base_frame == requested_base
        ):
            transformer = candidate

    if transformer is None:
        transformer = get_odom_pose_transformer(
            ros_node,
            odom_topic=requested_topic,
            target_frame=requested_target,
            base_frame=requested_base,
            queue_size=queue_size,
            history_duration_sec=requested_history,
        )

    # 某些节点需要更长历史窗口；共享实例可安全放大窗口，后续回调会按新窗口裁剪。
    if transformer.history_duration_sec < requested_history:
        transformer.history_duration_sec = requested_history
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


def lookup_transform_matrix(tf_listener, ros_node, target_frame, source_frame, timeout=0.5):
    """等待 TF 并返回 target_frame <- source_frame 的 4x4 变换矩阵。"""
    if target_frame == source_frame:
        return tf_trans.identity_matrix()
    stamp = ros_node.zero_time()
    tf_listener.waitForTransform(
        target_frame,
        source_frame,
        stamp,
        ros_node.duration(timeout),
    )
    translation, quaternion = tf_listener.lookupTransform(
        target_frame,
        source_frame,
        stamp,
    )
    return tf_trans.concatenate_matrices(
        tf_trans.translation_matrix(translation),
        tf_trans.quaternion_matrix(quaternion),
    )


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
    target_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
):
    """使用 odom 中的 base_link 位姿，将 base_link 三维点转换到 map/odom 坐标系。"""
    map_from_base = base_link_pose_matrix_from_melon_odom_msg(
        odom_msg,
        target_frame=target_frame,
        base_frame=base_frame,
    )
    # 关键步骤：odom.pose 表示 base_link 在 map/odom 下的位姿，这里用完整 3D 位姿做 base -> map。
    transformed = map_from_base.dot(
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


def base_link_pose_matrix_from_melon_odom_msg(
    odom_msg,
    target_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
):
    """使用 melon_odom topic 消息里的 base_link pose 构造 target_frame <- base_frame 矩阵。"""
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
    return transform_matrix


def map_from_source_matrix_via_melon_odom(
    tf_listener,
    ros_node,
    odom_msg,
    source_frame,
    map_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
    timeout=0.5,
):
    """用 odom topic 做 map<-base，用 TF 只查 base<-source，返回 map<-source。"""
    if source_frame == map_frame:
        return tf_trans.identity_matrix()
    map_from_base = base_link_pose_matrix_from_melon_odom_msg(
        odom_msg,
        target_frame=map_frame,
        base_frame=base_frame,
    )
    if source_frame == base_frame:
        return map_from_base
    # 关键步骤：TF 只用于机器人本体内部链路，底盘世界位姿只来自 odom topic。
    base_from_source = lookup_transform_matrix(
        tf_listener,
        ros_node,
        base_frame,
        source_frame,
        timeout=timeout,
    )
    return tf_trans.concatenate_matrices(map_from_base, base_from_source)


def base_from_map_matrix_via_melon_odom(
    odom_msg,
    map_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
):
    """用 odom topic 消息构造 base_frame <- map_frame 的 4x4 矩阵。"""
    map_from_base = base_link_pose_matrix_from_melon_odom_msg(
        odom_msg,
        target_frame=map_frame,
        base_frame=base_frame,
    )
    # 关键步骤：base<-map 必须由 odom topic 的 map<-base 取逆得到，避免走 TF 中的 melon_odom frame。
    return tf_trans.inverse_matrix(map_from_base)


def lookup_transform_matrix_via_melon_odom(
    tf_listener,
    ros_node,
    odom_msg,
    target_frame,
    source_frame,
    map_frame=MAP_FRAME,
    base_frame=BASE_LINK_FRAME,
    timeout=0.5,
):
    """用 odom topic 处理 map/base 世界位姿，用 TF 处理机器人本体内部链路。"""
    if target_frame == source_frame:
        return tf_trans.identity_matrix()
    if target_frame == map_frame:
        return map_from_source_matrix_via_melon_odom(
            tf_listener,
            ros_node,
            odom_msg,
            source_frame,
            map_frame=map_frame,
            base_frame=base_frame,
            timeout=timeout,
        )
    if target_frame == base_frame and source_frame == map_frame:
        return base_from_map_matrix_via_melon_odom(
            odom_msg,
            map_frame=map_frame,
            base_frame=base_frame,
        )
    if target_frame == base_frame:
        return lookup_transform_matrix(
            tf_listener,
            ros_node,
            base_frame,
            source_frame,
            timeout=timeout,
        )
    raise RuntimeError(
        "不支持的 odom 分段转换: %s <- %s，仅支持 target 为 %s/%s"
        % (target_frame, source_frame, map_frame, base_frame)
    )


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
    base_frame=BASE_LINK_FRAME,
    target_frame=MAP_FRAME,
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


def matrix_to_xyz_ypr(pose_matrix):
    """将 4x4 位姿矩阵转为 [x, y, z, yaw, pitch, roll]，角度单位为 deg。"""
    roll, pitch, yaw = tf_trans.euler_from_matrix(pose_matrix)
    return [
        float(pose_matrix[0, 3]),
        float(pose_matrix[1, 3]),
        float(pose_matrix[2, 3]),
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    ]


def format_xyz_ypr(xyz_ypr):
    """格式化 [x, y, z, yaw, pitch, roll]，便于日志输出。"""
    return (
        f"x={xyz_ypr[0]:.4f}, y={xyz_ypr[1]:.4f}, z={xyz_ypr[2]:.4f}, "
        f"yaw={xyz_ypr[3]:.2f}deg, pitch={xyz_ypr[4]:.2f}deg, roll={xyz_ypr[5]:.2f}deg"
    )


def make_xz_pitch_transform(x, z, pitch):
    """构造仅包含 x/z 平移和 pitch 旋转的 4x4 变换矩阵。"""
    transform = tf_trans.euler_matrix(0.0, float(pitch), 0.0)
    transform[:3, 3] = [float(x), 0.0, float(z)]
    return transform


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
