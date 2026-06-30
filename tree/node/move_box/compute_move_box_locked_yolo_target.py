"""使用单帧 YOLO 计算箱体在 map/control_frame 下的固定头部跟踪点。"""

import math
import threading
import time

import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray
from py_trees.common import Status

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class ComputeMoveBoxLockedYoloTarget(TimedMockAction):
    """只消费一帧非空 YOLO，并把选中的箱体点写入 blackboard。

    本节点只负责计算，不直接控制头部；后续由 MoveBoxTrackHeadToMapPoint
    读取 target_point_key 持续执行头部跟踪。
    """

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.yolo_topic = str(params.get("yolo_topic", "/yolo/target_poses")).strip()
        self.target_select_frame = str(params.get("target_select_frame", BASE_LINK_FRAME)).strip()
        self.control_frame = str(params.get("control_frame", MAP_FRAME)).strip()
        self.chassis_frame = str(params.get("chassis_frame", "melon_odom")).strip()
        self.target_point_key = str(
            params.get("target_point_key", "move_box_locked_yolo_head_target_point")
        ).strip()
        self.no_target_log_interval_sec = float(params.get("no_target_log_interval_sec", 1.0))

        self.latest_msg = None
        self.lock = threading.Lock()
        self.subscriber = self.ros_node.create_message_subscription(
            self.yolo_topic,
            PoseArray,
            self._on_yolo_pose_array,
            queue_size=1,
        )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.target_point_key, access=py_trees.common.Access.WRITE)
        self._last_no_target_log_time = 0.0

    def initialise(self):
        super().initialise()
        self._last_no_target_log_time = 0.0

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "head_controller"):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] services 或 head_controller 缺失: key={self.services_key}"
            )
            return Status.RUNNING

        pose_array = self._get_latest_pose_array()
        if pose_array is None or len(pose_array.poses) == 0:
            self._log_no_target(f"[{self.config_label}] 等待第一帧非空 YOLO 目标...")
            return Status.RUNNING

        head_controller = services.head_controller
        source_frame = pose_array.header.frame_id or head_controller.head_frame
        nearest_pose = self._select_nearest_pose(head_controller, pose_array, source_frame)
        if nearest_pose is None:
            self._log_no_target(f"[{self.config_label}] 暂无可用 YOLO 目标...")
            return Status.RUNNING

        locked_point = self._transform_yolo_point_to_control_frame(
            head_controller,
            nearest_pose.position,
            source_frame,
        )
        if locked_point is None:
            return Status.RUNNING

        self.blackboard.set(self.target_point_key, locked_point, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算单帧 YOLO 头部跟踪点: "
            f"source_frame={source_frame}, control_frame={self.control_frame}, "
            f"target=({locked_point[0]:.3f}, {locked_point[1]:.3f}, {locked_point[2]:.3f}), "
            f"key={self.target_point_key}"
        )
        return Status.SUCCESS

    def _on_yolo_pose_array(self, msg):
        """只缓存最新 YOLO 消息，计算节点成功后会退出，不再消费后续 YOLO。"""
        with self.lock:
            self.latest_msg = msg

    def _get_latest_pose_array(self):
        with self.lock:
            return self.latest_msg

    def _select_nearest_pose(self, head_controller, pose_array, source_frame):
        """把候选点转到 target_select_frame 后，选择离该坐标系原点最近的目标。"""
        select_to_source = None
        if self.target_select_frame and self.target_select_frame != source_frame:
            select_to_source = self._lookup_transform_matrix(
                head_controller,
                target_frame=self.target_select_frame,
                source_frame=source_frame,
            )

        nearest_pose = None
        nearest_distance = None
        for pose in pose_array.poses:
            if select_to_source is None:
                xyz = [pose.position.x, pose.position.y, pose.position.z]
            else:
                xyz = self._matrix_dot_point(select_to_source, pose.position)
            distance = math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_pose = pose
        return nearest_pose

    def _transform_yolo_point_to_control_frame(self, head_controller, point, source_frame):
        """通过两段 TF 把 YOLO 点从 source_frame 锁定到 control_frame。

        T_control_source = T_control_chassis * T_base_source
        其中默认认为 chassis_frame 与 base_frame 重合。
        """
        control_to_chassis = self._lookup_transform_matrix(
            head_controller,
            target_frame=self.control_frame,
            source_frame=self.chassis_frame,
        )
        base_to_source = self._lookup_transform_matrix(
            head_controller,
            target_frame=head_controller.base_frame,
            source_frame=source_frame,
        )
        if control_to_chassis is None or base_to_source is None:
            return None

        control_to_source = tf_trans.concatenate_matrices(
            control_to_chassis,
            base_to_source,
        )
        return self._matrix_dot_point(control_to_source, point)

    def _lookup_transform_matrix(self, head_controller, target_frame, source_frame):
        latest_tf_time = self.ros_node.zero_time()
        try:
            head_controller.tf_listener.waitForTransform(
                target_frame,
                source_frame,
                latest_tf_time,
                self.ros_node.duration(head_controller.tf_timeout),
            )
            translation, rotation = head_controller.tf_listener.lookupTransform(
                target_frame,
                source_frame,
                latest_tf_time,
            )
        except Exception as err:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 无法查询 {source_frame} -> {target_frame}: {err}"
            )
            return None

        return tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(translation),
            tf_trans.quaternion_matrix(rotation),
        )

    @staticmethod
    def _matrix_dot_point(matrix, point):
        transformed = matrix.dot([float(point.x), float(point.y), float(point.z), 1.0])
        return [float(transformed[0]), float(transformed[1]), float(transformed[2])]

    def _log_no_target(self, message):
        now = time.monotonic()
        if now - self._last_no_target_log_time >= self.no_target_log_interval_sec:
            self.ros_node.get_logger().warning(message)
            self._last_no_target_log_time = now

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxLockedYoloTarget start: "
            f"topic={self.yolo_topic}, select_frame={self.target_select_frame}, "
            f"control_frame={self.control_frame}, target_point_key={self.target_point_key}"
        )
