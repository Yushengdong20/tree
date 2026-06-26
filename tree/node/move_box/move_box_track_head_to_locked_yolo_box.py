"""使用单帧 YOLO 锁定箱子 map 点，然后按固定点持续盯住。"""

import math
import threading
import time

import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray
from py_trees.common import Status

from .move_box_track_head_to_map_point import MoveBoxTrackHeadToMapPoint


class MoveBoxTrackHeadToLockedYoloBox(MoveBoxTrackHeadToMapPoint):
    """只消费一帧 YOLO，并把箱体位置锁定为 map/control_frame 下的固定点。

    这个节点适合“抓箱前箱子静止”的场景：
    1. 等待第一帧非空 YOLO PoseArray。
    2. 按 target_select_frame 选择最近目标，默认是 base_link 下最近的箱子。
    3. 通过 T_control_chassis * T_base_source 把 YOLO 点转换到 control_frame。
    4. 后续不再使用新的 YOLO 帧，完全复用固定点盯点逻辑。

    注意：它解决的是 YOLO 后续跳变/多目标切换问题；如果行为树被阻塞，
    该节点仍然只有在被 tick 到时才会发布新的头部控制。
    """

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.yolo_topic = str(params.get("yolo_topic", "/yolo/target_poses")).strip()
        self.target_select_frame = str(params.get("target_select_frame", "base_link")).strip()
        self.control_frame = str(params.get("control_frame", self.target_frame)).strip()
        self.no_target_log_interval_sec = float(params.get("no_target_log_interval_sec", 1.0))

        # 锁点最终固定在 control_frame 下；通常 control_frame=map。
        self.target_frame = self.control_frame
        self.tracking_mode = "split_tf"

        self.latest_msg = None
        self.lock = threading.Lock()
        self.subscriber = self.ros_node.create_message_subscription(
            self.yolo_topic,
            PoseArray,
            self._on_yolo_pose_array,
            queue_size=1,
        )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self._last_no_target_log_time = 0.0
        self._locked_target = False

    def initialise(self):
        super().initialise()
        self._last_no_target_log_time = 0.0
        self._locked_target = False
        self.target_point = None

    def update(self):
        if self.should_skip_head_motion():
            if not self._skip_logged:
                self.log_skip_head_motion()
                self._skip_logged = True
            return Status.RUNNING

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "head_controller"):
            self._log_failure(
                f"[{self.config_label}] services 或 head_controller 缺失: key={self.services_key}"
            )
            return Status.RUNNING

        if not self._locked_target:
            self._try_lock_yolo_target(services.head_controller)
            if not self._locked_target:
                self._log_no_target(f"[{self.config_label}] 等待第一帧非空 YOLO 目标...")
                return Status.RUNNING

        return super().update()

    def _on_yolo_pose_array(self, msg):
        """只缓存最新 YOLO 消息；锁定成功后不会再读取新 YOLO 更新目标点。"""
        with self.lock:
            self.latest_msg = msg

    def _get_latest_pose_array(self):
        with self.lock:
            return self.latest_msg

    def _try_lock_yolo_target(self, head_controller):
        pose_array = self._get_latest_pose_array()
        if pose_array is None or len(pose_array.poses) == 0:
            return

        source_frame = pose_array.header.frame_id or head_controller.head_frame
        nearest_pose = self._select_nearest_pose(head_controller, pose_array, source_frame)
        if nearest_pose is None:
            return

        locked_point = self._transform_yolo_point_to_control_frame(
            head_controller,
            nearest_pose.position,
            source_frame,
        )
        if locked_point is None:
            return

        self.target_point = locked_point
        self._locked_target = True
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已锁定单帧 YOLO 箱体目标: "
            f"source_frame={source_frame}, control_frame={self.control_frame}, "
            f"target=({locked_point[0]:.3f}, {locked_point[1]:.3f}, {locked_point[2]:.3f})"
        )

    def _select_nearest_pose(self, head_controller, pose_array, source_frame):
        """把候选点转到 target_select_frame 后，选择离该坐标系原点最近的目标。"""
        if not pose_array.poses:
            return None

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
                point = pose.position
                xyz = [point.x, point.y, point.z]
            else:
                transformed = self._matrix_dot_point(select_to_source, pose.position)
                xyz = transformed
            distance = math.sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_pose = pose
        return nearest_pose

    def _transform_yolo_point_to_control_frame(self, head_controller, point, source_frame):
        """通过两段 TF 把 YOLO 点从 source_frame 锁定到 control_frame。

        T_control_source = T_control_chassis * T_base_source

        这里与固定点节点同样假设 chassis_frame 与 base_frame 重合，
        避免直接查询 control_frame -> source_frame 整条链路。
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
            f"[{self.config_label}] 开始单帧 YOLO 锁定并持续盯箱: "
            f"topic={self.yolo_topic}, select_frame={self.target_select_frame}, "
            f"control_frame={self.control_frame}, chassis_frame={self.chassis_frame}, "
            f"debug_marker_topic={self.debug_marker_topic if self.debug_enabled else 'disabled'}"
        )
