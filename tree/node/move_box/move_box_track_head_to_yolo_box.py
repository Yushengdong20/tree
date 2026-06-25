"""YOLO 粗靠近阶段，让头部持续朝向最近箱体目标。

整体链路：
1. 订阅 `/yolo/target_poses`，拿到 camera/head_frame 下的箱体候选点。
2. 选取最近目标，把 YOLO 新帧转换成 map/control_frame 下的“锁点”。
3. 控制周期内把锁点按当前 TF 重投影回 camera/head_frame，计算 yaw/pitch 误差。
4. 按 gain 与单步限幅发布头部目标，同时发布 RViz Marker 辅助观察。
"""

import math
import threading
import time

import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import Point, PointStamped, PoseArray
from py_trees.common import Status
from visualization_msgs.msg import Marker

from ..base import TimedMockAction


class MoveBoxTrackHeadToYoloBox(TimedMockAction):
    """持续订阅 YOLO PoseArray，并驱动头部盯住最近目标。

    RViz 调试 Marker 与控制链路使用同一个 YOLO 最近目标：
    - 橙色球：当前被选中的最近 YOLO 目标点。
    - 蓝色球：camera/head_frame 原点。
    - 绿色线：camera/head_frame 原点到 YOLO 目标点。
    - 红色线：camera/head_frame 当前 x 轴方向。

    当红色线和绿色线方向逐渐重合时，就表示 camera/head_frame 的 x 轴正在对准箱体。
    debug_frame 可配置为 map/base_link/camera 等坐标系，只影响 RViz 显示，不影响控制。
    """

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 共享服务实例由 EnsureMoveBoxServices 创建，里面包含 head_controller/tf_listener。
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        # YOLO 输出 PoseArray：通常 header.frame_id=camera，pose.position 为目标点。
        self.yolo_topic = str(params.get("yolo_topic", "/yolo/target_poses")).strip()
        # 多目标选择坐标系：默认先把 YOLO 候选点转到 base_link，再选离底盘最近的箱子。
        # 如果现场需要恢复旧行为，可在 JSON 中填 target_select_frame=camera。
        self.target_select_frame = str(params.get("target_select_frame", "base_link")).strip()
        # 行为树 tick 很快，这里用 track_interval_sec 控制头部控制发布频率。
        self.track_interval_sec = float(params.get("track_interval_sec", 0.2))
        # latched_map_target：把 YOLO 点锁到 map，再高频重投影追踪。
        # legacy/其它值：只在新 YOLO 帧到来时直接按当前 YOLO 点控制一次。
        self.control_mode = str(params.get("control_mode", "latched_map_target")).strip()
        self.control_frame = str(params.get("control_frame", "map")).strip()
        self.no_target_log_interval_sec = float(params.get("no_target_log_interval_sec", 1.0))
        self.failure_log_interval_sec = float(params.get("failure_log_interval_sec", 1.0))
        # 单次控制增量限幅。它限制的是“每次发布目标角度变化”，不是头部硬限位。
        self.max_delta_yaw_deg = float(params.get("max_delta_yaw_deg", 3.0))
        self.max_delta_pitch_deg = float(params.get("max_delta_pitch_deg", 3.0))
        # 头部增量控制模式：
        # - p：保持原始 P 控制，raw_step = gain * error。
        # - pd：在 P 基础上加入误差变化率 D 项，目标是接近目标时提前减速，降低超调。
        self.head_control_mode = str(params.get("head_control_mode", "p")).strip().lower()
        self.head_kp_yaw = self._optional_float(params.get("head_kp_yaw", None))
        self.head_kp_pitch = self._optional_float(params.get("head_kp_pitch", None))
        self.head_kd_yaw = float(params.get("head_kd_yaw", 0.0))
        self.head_kd_pitch = float(params.get("head_kd_pitch", 0.0))
        self.head_derivative_filter_alpha = self._clamp(
            float(params.get("head_derivative_filter_alpha", 0.5)),
            0.0,
            1.0,
        )
        # 锁点更新阈值说明：
        # residual = 新 YOLO camera 观测点 与 旧 map 锁点重投影到 camera 后预测点 的距离。
        # residual 小于 fast：认为观测稳定，快速吸收新目标。
        # residual 小于 slow：认为可能有轻微噪声/箱体慢动，低速吸收。
        # residual 大于 slow：认为发生大变化，进入 pending，连续确认后才重锁。
        self.lock_residual_fast_m = float(params.get("lock_residual_fast_m", 0.12))
        self.lock_residual_slow_m = float(params.get("lock_residual_slow_m", 0.35))
        self.lock_fast_alpha = self._clamp(float(params.get("lock_fast_alpha", 0.5)), 0.0, 1.0)
        self.lock_slow_alpha = self._clamp(float(params.get("lock_slow_alpha", 0.2)), 0.0, 1.0)
        # 大变化不立刻相信，避免单帧误检/TF 瞬时错位把锁点拉飞。
        # 只有连续 confirm_count 帧都落在同一个新位置附近，才确认箱子真的移动并重锁。
        self.lock_pending_confirm_count = max(
            1,
            int(params.get("lock_pending_confirm_count", 2)),
        )
        self.lock_pending_match_m = float(params.get("lock_pending_match_m", 0.20))
        # 底盘快速转向时 TF/YOLO 时间不一致更明显；这里不再冻结锁点，
        # 只在“中等 residual 慢速更新”分支里进一步降低 alpha。
        self.chassis_yaw_rate_slow_update_deg_s = float(
            params.get("chassis_yaw_rate_slow_update_deg_s", 12.0)
        )
        self.axis_length_m = float(params.get("axis_length_m", 1.0))
        # debug_point_topic 只发布目标点；debug_marker_topic 同时包含目标、头部原点、视线和 x 轴。
        self.debug_enabled = self._to_bool(params.get("debug_enabled", True))
        self.debug_point_topic = str(
            params.get("debug_point_topic", "/head_track_yolo_target_point")
        ).strip()
        self.debug_marker_topic = str(
            params.get("debug_marker_topic", "/head_track_yolo_marker")
        ).strip()
        self.debug_frame = str(params.get("debug_frame", "")).strip()
        # melon_odom 是底盘在 map 下的实时位姿；默认认为 melon_odom 与 base_link 重合。
        self.chassis_frame = str(params.get("chassis_frame", "melon_odom")).strip()

        # ROS 订阅只缓存最新消息，不在回调线程里做 TF 查询或真机控制。
        self.latest_msg = None
        self.lock = threading.Lock()
        self.subscriber = self.ros_node.create_message_subscription(
            self.yolo_topic,
            PoseArray,
            self._on_yolo_pose_array,
            queue_size=1,
        )
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.debug_point_pub = None
        self.debug_marker_pub = None
        if self.debug_enabled:
            self.debug_point_pub = self.ros_node.create_publisher(
                self.debug_point_topic,
                PointStamped,
                queue_size=1,
                latch=True,
            )
            self.debug_marker_pub = self.ros_node.create_publisher(
                self.debug_marker_topic,
                Marker,
                queue_size=10,
                latch=True,
            )
        # 运行态状态：
        # - _latched_target_point：当前被信任的 map/control_frame 锁点。
        # - _last_latch_candidate_stamp：避免同一 YOLO 帧重复刷新锁点。
        # - _pending_relock_*：大 residual 时的连续确认重锁候选。
        self._last_track_time = 0.0
        self._last_no_target_log_time = 0.0
        self._last_failure_log_time = 0.0
        self._last_control_stamp = None
        self._latched_target_point = None
        self._latched_target_stamp = None
        self._last_latch_candidate_stamp = None
        self._last_chassis_yaw_deg = None
        self._last_chassis_yaw_time = None
        self._pending_relock_point = None
        self._pending_relock_count = 0
        self._last_head_control_time = None
        self._last_error_yaw = None
        self._last_error_pitch = None
        self._filtered_derivative_yaw = 0.0
        self._filtered_derivative_pitch = 0.0
        self._skip_logged = False

    def initialise(self):
        super().initialise()
        # 每次节点重新进入时清空追踪状态，避免上一轮测试残留锁点影响新一轮。
        self._last_track_time = 0.0
        self._last_control_stamp = None
        self._latched_target_point = None
        self._latched_target_stamp = None
        self._last_latch_candidate_stamp = None
        self._last_chassis_yaw_deg = None
        self._last_chassis_yaw_time = None
        self._pending_relock_point = None
        self._pending_relock_count = 0
        self._reset_head_control_state()
        self._skip_logged = False

    def update(self):
        # 测试保护：全局 skip_head_motion=True 时不下发头部命令，但树保持 RUNNING。
        if self.should_skip_head_motion():
            if not self._skip_logged:
                self.log_skip_head_motion()
                self._skip_logged = True
            return Status.RUNNING

        now = time.monotonic()
        # 控制频率限流：避免树 tick 频率过高时连续刷头部目标。
        if now - self._last_track_time < self.track_interval_sec:
            return Status.RUNNING
        self._last_track_time = now

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "head_controller"):
            self._log_throttled(
                "failure",
                f"[{self.config_label}] services 或 head_controller 缺失: key={self.services_key}",
            )
            return Status.RUNNING

        # 每个周期先尝试消费最新 YOLO：有新帧就更新锁点，没有新帧也继续追旧锁点。
        pose_array = self._get_latest_pose_array()
        nearest_pose = self._get_nearest_pose(services.head_controller, pose_array)
        if nearest_pose is not None:
            self._update_latched_target_if_new(services.head_controller, pose_array, nearest_pose)

        if self.control_mode == "latched_map_target":
            # 推荐模式：YOLO 负责低频刷新 map 锁点，头部高频重投影追踪。
            ok = self._track_latched_target(services.head_controller)
        else:
            # 兼容旧模式：同一 YOLO 帧只控制一次，避免重复追同一 camera 观测导致超调。
            if nearest_pose is None:
                self._log_throttled("no_target", f"[{self.config_label}] 等待 YOLO 最近目标...")
                return Status.RUNNING
            current_stamp = self._stamp_key(pose_array.header.stamp)
            if current_stamp == self._last_control_stamp:
                return Status.RUNNING
            self._last_control_stamp = current_stamp
            source_frame = pose_array.header.frame_id or services.head_controller.base_frame
            target_point_msg = self._build_target_point_msg(nearest_pose, source_frame)
            self._publish_debug_markers(services.head_controller, target_point_msg)
            ok = self._turn_to_target_with_limited_step(
                services.head_controller,
                nearest_pose,
                source_frame,
            )
        if not ok:
            self._log_throttled("failure", f"[{self.config_label}] 头部朝向 YOLO 最近目标失败")
        return Status.RUNNING

    def _on_yolo_pose_array(self, msg):
        """只缓存最新 YOLO 消息，避免在 ROS 回调里等待 TF 或控制真机。"""
        with self.lock:
            self.latest_msg = msg

    def _get_latest_pose_array(self):
        """取出订阅回调缓存的最新 YOLO 消息。"""
        with self.lock:
            return self.latest_msg

    @staticmethod
    def _stamp_key(stamp):
        """用 YOLO header.stamp 去重，保证同一帧视觉最多触发一次头部控制。"""
        return (int(stamp.secs), int(stamp.nsecs))

    def _get_nearest_pose(self, head_controller, pose_array):
        """从 YOLO PoseArray 中选择离 target_select_frame 原点最近的目标。

        YOLO 通常发布在 camera 下；如果直接按 camera 原点选最近，头部转动会改变
        多个箱子的相对距离。这里默认转到 base_link 后再算距离，更符合“离机器人最近”。
        TF 查询失败时降级为原始 frame 下距离，保证现场不会因为一次 TF 异常卡死。
        """
        if pose_array is None or len(pose_array.poses) == 0:
            return None

        source_frame = pose_array.header.frame_id or head_controller.head_frame
        nearest_pose = None
        nearest_distance = None
        for pose in pose_array.poses:
            point = self._pose_position_in_select_frame(
                head_controller,
                pose,
                source_frame,
            )
            distance = math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_pose = pose
        return nearest_pose

    def _pose_position_in_select_frame(self, head_controller, pose, source_frame):
        """返回候选目标在 target_select_frame 下的位置，用于多目标最近距离筛选。"""
        if not self.target_select_frame or self.target_select_frame == source_frame:
            return pose.position

        point_msg = PointStamped()
        point_msg.header.stamp = self.ros_node.zero_time()
        point_msg.header.frame_id = source_frame
        point_msg.point = pose.position
        point = self._transform_point_between_frames(
            head_controller,
            point_msg,
            self.target_select_frame,
        )
        if point is None:
            self._log_throttled(
                "failure",
                f"[{self.config_label}] YOLO目标从 {source_frame} "
                f"转到 {self.target_select_frame} 失败，降级按 {source_frame} 距离筛选",
            )
            return pose.position
        return point

    @staticmethod
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        return float(text)

    def _reset_head_control_state(self):
        """重置 PD 历史项，避免重新进入节点或进入误差死区后残留 D 项。"""
        self._last_head_control_time = None
        self._last_error_yaw = None
        self._last_error_pitch = None
        self._filtered_derivative_yaw = 0.0
        self._filtered_derivative_pitch = 0.0

    def _build_target_point_msg(self, pose, source_frame):
        """把当前选中的 YOLO 目标转成 PointStamped，方便 RViz 和 rostopic 检查。"""
        point_msg = PointStamped()
        point_msg.header.stamp = self.ros_node.now()
        point_msg.header.frame_id = source_frame
        point_msg.point = pose.position
        return point_msg

    def _update_latched_target_if_new(self, head_controller, pose_array, nearest_pose):
        """YOLO 新帧到来时，把 camera 下目标点锁定到 control_frame/map 下。

        机器人靠近箱子时 base_link 会移动，所以不能把目标锁在 base_link。
        这里默认锁到 map：箱子静止时 map 下目标应保持稳定；之后头部高频控制
        都重新把这个 map 目标投影到当前 camera 下计算误差。
        """
        current_stamp = self._stamp_key(pose_array.header.stamp)
        # YOLO 频率低于控制频率；同一视觉帧只允许作为锁点候选处理一次。
        if current_stamp == self._last_latch_candidate_stamp:
            return
        self._last_latch_candidate_stamp = current_stamp

        target_to_head = None
        source_frame = pose_array.header.frame_id or head_controller.base_frame
        source_point_msg = self._build_target_point_msg(nearest_pose, source_frame)
        if self.control_frame == "map" and source_frame == head_controller.head_frame:
            # 关键路径：camera 点 -> map 点。
            # 不直接查 map->camera 整链 TF，而是分段组合 map->melon_odom 与 base_link->camera。
            target_to_head = self._build_split_tf_target_to_head(
                head_controller,
                self.control_frame,
            )
            if target_to_head is None:
                return
            target_point = self._transform_point_with_matrix(
                target_to_head,
                [
                    source_point_msg.point.x,
                    source_point_msg.point.y,
                    source_point_msg.point.z,
                ],
            )
        else:
            # 非默认路径：如果 YOLO 已经不是 camera/frame，退回普通 TF 转换。
            target_point = self._transform_point_between_frames(
                head_controller,
                source_point_msg,
                self.control_frame,
            )
            if target_point is None:
                return

        # 候选点不会直接覆盖锁点；先经过残差门控、平滑或连续确认重锁。
        accepted_point = self._filter_latched_target_update(
            head_controller,
            candidate_point=target_point,
            candidate_source_point=source_point_msg.point,
            target_to_head=target_to_head,
        )
        if accepted_point is None:
            return

        self._latched_target_point = accepted_point
        self._latched_target_stamp = current_stamp
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 更新YOLO锁定目标: "
            f"source_frame={source_frame}, control_frame={self.control_frame}, "
            f"target=({accepted_point.x:.3f}, {accepted_point.y:.3f}, {accepted_point.z:.3f})"
        )

    def _filter_latched_target_update(
        self,
        head_controller,
        candidate_point,
        candidate_source_point,
        target_to_head,
    ):
        """基于预测残差自适应更新 YOLO map 锁点。

        新 YOLO 点先转换成 control_frame/map 下候选点。随后把旧锁点重投影到
        当前 camera/head_frame，和新 YOLO camera 点比较 residual：
        - residual 小：新观测和预测一致，快速更新；
        - residual 中：可能是箱子慢动或轻微噪声，慢速更新；
        - residual 大：不立刻丢弃，连续多帧支持同一新位置后才重锁。
        """
        if self._latched_target_point is None:
            self._sample_chassis_yaw_rate(head_controller)
            self._reset_pending_relock()
            return candidate_point

        yaw_rate = self._sample_chassis_yaw_rate(head_controller)
        residual = self._compute_observation_residual(
            candidate_source_point=candidate_source_point,
            target_to_head=target_to_head,
        )
        if residual is None:
            residual = self._point_distance(candidate_point, self._latched_target_point)

        if residual < self.lock_residual_fast_m:
            # 小 residual：新观测和旧锁点预测基本一致，说明 YOLO/TF 当前可信。
            # 用较大的 alpha 让锁点快速贴近观测，可提升跟随响应。
            self._reset_pending_relock()
            return self._blend_points(
                self._latched_target_point,
                candidate_point,
                self.lock_fast_alpha,
            )

        if residual < self.lock_residual_slow_m:
            # 中等 residual：可能是箱子慢慢移动，也可能是观测噪声。
            # 不丢弃，但用较小 alpha 慢慢吸收；如果底盘正在快速转向，再降一半 alpha。
            self._reset_pending_relock()
            alpha = self.lock_slow_alpha
            if (
                yaw_rate is not None
                and self.chassis_yaw_rate_slow_update_deg_s > 0.0
                and abs(yaw_rate) > self.chassis_yaw_rate_slow_update_deg_s
            ):
                alpha *= 0.5
            yaw_rate_text = "unknown" if yaw_rate is None else f"{yaw_rate:.1f}deg/s"
            self.ros_node.get_logger().info(
                f"[{self.config_label}] YOLO锁点慢速更新: "
                f"residual={residual:.3f}m, alpha={alpha:.2f}, "
                f"yaw_rate={yaw_rate_text}"
            )
            return self._blend_points(self._latched_target_point, candidate_point, alpha)

        # 大 residual：单帧不能判断是箱子真的移动，还是 YOLO/TF 瞬时污染。
        # 进入 pending，只有连续多帧都支持同一新位置时才重锁。
        return self._update_pending_relock(candidate_point, residual)

    def _compute_observation_residual(self, candidate_source_point, target_to_head):
        """比较新 YOLO camera 点和旧锁点按当前 TF 重投影到 camera 的预测点。

        candidate_source_point 是本帧 YOLO 在 camera/head_frame 下的实际观测。
        predicted 是“旧 map 锁点”按当前 map->camera 矩阵反投影回 camera 后的位置。
        两者距离越小，说明“旧锁点 + 当前 TF”能解释新观测，锁点更新越可信。
        """
        if target_to_head is None or self._latched_target_point is None:
            return None

        head_to_target = tf_trans.inverse_matrix(target_to_head)
        predicted = head_to_target.dot(
            [
                self._latched_target_point.x,
                self._latched_target_point.y,
                self._latched_target_point.z,
                1.0,
            ]
        )
        dx = float(candidate_source_point.x) - float(predicted[0])
        dy = float(candidate_source_point.y) - float(predicted[1])
        dz = float(candidate_source_point.z) - float(predicted[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _update_pending_relock(self, candidate_point, residual):
        """大残差连续多帧接近同一新位置时，确认箱子真的移动并重锁。

        第一帧大 residual 只记录 pending，不更新正式锁点。
        后续大 residual 候选点如果离 pending 点小于 lock_pending_match_m，
        就认为多帧在支持同一个新位置；计数达到 lock_pending_confirm_count 后重锁。
        """
        if (
            self._pending_relock_point is None
            or self._point_distance(candidate_point, self._pending_relock_point)
            > self.lock_pending_match_m
        ):
            self._pending_relock_point = candidate_point
            self._pending_relock_count = 1
        else:
            self._pending_relock_count += 1
            self._pending_relock_point = self._blend_points(
                self._pending_relock_point,
                candidate_point,
                0.5,
            )

        if self._pending_relock_count >= self.lock_pending_confirm_count:
            accepted = self._pending_relock_point
            self._reset_pending_relock()
            self.ros_node.get_logger().info(
                f"[{self.config_label}] YOLO锁点连续确认后重锁: "
                f"residual={residual:.3f}m, "
                f"confirm_count={self.lock_pending_confirm_count}"
            )
            return accepted

        self._log_throttled(
            "failure",
            f"[{self.config_label}] YOLO锁点大残差待确认: "
            f"residual={residual:.3f}m, "
            f"pending={self._pending_relock_count}/{self.lock_pending_confirm_count}",
        )
        return None

    def _reset_pending_relock(self):
        self._pending_relock_point = None
        self._pending_relock_count = 0

    @staticmethod
    def _blend_points(old_point, new_point, alpha):
        return Point(
            x=(1.0 - alpha) * old_point.x + alpha * new_point.x,
            y=(1.0 - alpha) * old_point.y + alpha * new_point.y,
            z=(1.0 - alpha) * old_point.z + alpha * new_point.z,
        )

    def _track_latched_target(self, head_controller):
        """把锁定的 map 目标按当前头部姿态重投影到 camera，并控制头部追踪。"""
        if self._latched_target_point is None:
            self._log_throttled("no_target", f"[{self.config_label}] 等待 YOLO 锁定目标...")
            return True

        target_to_head = None
        if self.control_frame == "map":
            # 当前 TF 下的 T_map_camera。取逆矩阵后可把 map 锁点转回 camera/head_frame。
            target_to_head = self._build_split_tf_target_to_head(head_controller, self.control_frame)
            if target_to_head is None:
                return False
            head_to_target = tf_trans.inverse_matrix(target_to_head)
            target_xyz_in_head = head_to_target.dot(
                [
                    self._latched_target_point.x,
                    self._latched_target_point.y,
                    self._latched_target_point.z,
                    1.0,
                ]
            )
            target_in_head = PointStamped()
            target_in_head.header.stamp = self.ros_node.zero_time()
            target_in_head.header.frame_id = head_controller.head_frame
            target_in_head.point.x = float(target_xyz_in_head[0])
            target_in_head.point.y = float(target_xyz_in_head[1])
            target_in_head.point.z = float(target_xyz_in_head[2])
        else:
            # 其它 control_frame 走标准 TF 转换，适合后续扩展 base_link/odom 等模式。
            target_in_control = PointStamped()
            target_in_control.header.stamp = self.ros_node.zero_time()
            target_in_control.header.frame_id = self.control_frame
            target_in_control.point = self._latched_target_point
            try:
                head_controller.tf_listener.waitForTransform(
                    head_controller.head_frame,
                    self.control_frame,
                    self.ros_node.zero_time(),
                    self.ros_node.duration(head_controller.tf_timeout),
                )
                target_in_head = head_controller.tf_listener.transformPoint(
                    head_controller.head_frame,
                    target_in_control,
                )
            except Exception as err:
                self._log_throttled(
                    "failure",
                    f"[{self.config_label}] 锁定目标从 {self.control_frame} "
                    f"转到 {head_controller.head_frame} 失败: {err}",
                )
                return False

        # 可视化必须和控制使用同一个锁点和同一套 TF 矩阵，避免 RViz 看起来和实机动作不一致。
        debug_point = PointStamped()
        debug_point.header.stamp = self.ros_node.now()
        debug_point.header.frame_id = self.control_frame
        debug_point.point = self._latched_target_point
        self._publish_debug_markers(
            head_controller,
            debug_point,
            target_to_head_matrix=target_to_head,
        )
        return self._turn_to_head_point_with_limited_step(head_controller, target_in_head)

    def _turn_to_target_with_limited_step(self, head_controller, target_pose, source_frame):
        """只对当前 YOLO 新帧发布一次头部控制，并限制单次 yaw/pitch 修正量。

        HeadController 原始逻辑是增量控制：
            send = current + gain * delta

        这里保留同样的几何解算，但额外把本次增量限制在
        max_delta_yaw_deg / max_delta_pitch_deg 内，避免 YOLO 点跳变时一次追太猛。
        """
        target_point = head_controller.build_target_point(target_pose, source_frame)
        try:
            # legacy 路径：先把目标转到 head_frame，再复用统一的限幅控制函数。
            head_controller.tf_listener.waitForTransform(
                head_controller.head_frame,
                target_point.header.frame_id,
                self.ros_node.zero_time(),
                self.ros_node.duration(head_controller.tf_timeout),
            )
            target_in_head = head_controller.tf_listener.transformPoint(
                head_controller.head_frame,
                target_point,
            )
        except Exception as err:
            self._log_throttled(
                "failure",
                f"[{self.config_label}] YOLO 目标点从 {target_point.header.frame_id} "
                f"转到 {head_controller.head_frame} 失败: {err}",
            )
            return False
        return self._turn_to_head_point_with_limited_step(head_controller, target_in_head)

    def _turn_to_head_point_with_limited_step(self, head_controller, target_in_head):
        """根据当前 camera/head_frame 下目标点计算受限的单次头部控制量。

        target_in_head 已经是最新 camera/head_frame 下的坐标：
        - legacy 模式：直接由当前 YOLO 点转换得到。
        - latched_map_target 模式：由锁定的 map 目标按当前头部 TF 重投影得到。
        因此这里不再关心原始 YOLO frame，只根据 head_frame 下 x/y/z 误差做增量控制。
        """
        x = target_in_head.point.x
        y = target_in_head.point.y
        z = target_in_head.point.z
        horizontal_distance = math.hypot(x, y)
        target_distance = math.hypot(horizontal_distance, z)
        if target_distance < 1e-6:
            self._log_throttled("failure", f"[{self.config_label}] YOLO 目标距离头部过近")
            return False

        if (
            abs(y) < head_controller.head_target_y_tolerance
            and abs(z) < head_controller.head_target_z_tolerance
        ):
            # 目标已经落在 camera x 轴附近，不再发布新目标，避免在小误差内来回抖。
            self._reset_head_control_state()
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 头部目标已在误差阈值内: "
                f"y={y:.3f}, z={z:.3f}"
            )
            return True

        # camera/head_frame 下：x 为前方，y 为左右误差，z 为上下误差。
        # yaw 修正左右偏差；pitch 修正上下偏差。
        delta_yaw = math.degrees(math.atan2(y, x))
        delta_pitch = -math.degrees(math.atan2(z, horizontal_distance))
        raw_step_yaw, raw_step_pitch, control_debug = self._compute_head_control_step(
            head_controller,
            delta_yaw,
            delta_pitch,
        )
        step_yaw = self._clamp(raw_step_yaw, -self.max_delta_yaw_deg, self.max_delta_yaw_deg)
        step_pitch = self._clamp(
            raw_step_pitch,
            -self.max_delta_pitch_deg,
            self.max_delta_pitch_deg,
        )
        # current_yaw/current_pitch 是 HeadController 最近一次发送的目标角；
        # 当前节点按增量方式更新，并再次夹到头部机械/软件限位内。
        yaw = self._clamp(
            head_controller.current_yaw + step_yaw,
            head_controller.yaw_min,
            head_controller.yaw_max,
        )
        pitch = self._clamp(
            head_controller.current_pitch + step_pitch,
            head_controller.pitch_min,
            head_controller.pitch_max,
        )

        self.ros_node.get_logger().info(
            f"[{self.config_label}] YOLO头部限幅控制: "
            f"head_frame={target_in_head.header.frame_id}, "
            f"head_xyz=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"delta_yaw={delta_yaw:.1f}, delta_pitch={delta_pitch:.1f}, "
            f"{control_debug}, "
            f"raw_step=({raw_step_yaw:.1f}, {raw_step_pitch:.1f}), "
            f"limited_step=({step_yaw:.1f}, {step_pitch:.1f}), "
            f"send_yaw={yaw:.1f}, send_pitch={pitch:.1f}"
        )
        return head_controller.set_head_target(yaw, pitch)

    def _compute_head_control_step(self, head_controller, error_yaw, error_pitch):
        """计算头部增量控制量。

        默认 P 模式完全兼容原逻辑：kp 使用 HeadController.head_tracking_gain。
        PD 模式额外根据误差变化率添加 D 项：
        - 误差快速变小时，D 项会抵消一部分 P，帮助接近目标时减速；
        - 误差快速变大时，D 项会增加响应，帮助移动目标更快跟上。
        """
        kp_yaw = (
            head_controller.head_tracking_gain
            if self.head_kp_yaw is None
            else self.head_kp_yaw
        )
        kp_pitch = (
            head_controller.head_tracking_gain
            if self.head_kp_pitch is None
            else self.head_kp_pitch
        )
        p_yaw = kp_yaw * error_yaw
        p_pitch = kp_pitch * error_pitch

        if self.head_control_mode != "pd":
            self._last_head_control_time = time.monotonic()
            self._last_error_yaw = error_yaw
            self._last_error_pitch = error_pitch
            return p_yaw, p_pitch, f"mode=p, kp=({kp_yaw:.2f}, {kp_pitch:.2f})"

        now = time.monotonic()
        d_yaw = 0.0
        d_pitch = 0.0
        if (
            self._last_head_control_time is not None
            and self._last_error_yaw is not None
            and self._last_error_pitch is not None
        ):
            dt = max(now - self._last_head_control_time, 1e-3)
            d_yaw = (error_yaw - self._last_error_yaw) / dt
            d_pitch = (error_pitch - self._last_error_pitch) / dt
            alpha = self.head_derivative_filter_alpha
            self._filtered_derivative_yaw = (
                alpha * d_yaw + (1.0 - alpha) * self._filtered_derivative_yaw
            )
            self._filtered_derivative_pitch = (
                alpha * d_pitch + (1.0 - alpha) * self._filtered_derivative_pitch
            )

        self._last_head_control_time = now
        self._last_error_yaw = error_yaw
        self._last_error_pitch = error_pitch

        d_step_yaw = self.head_kd_yaw * self._filtered_derivative_yaw
        d_step_pitch = self.head_kd_pitch * self._filtered_derivative_pitch
        raw_step_yaw = p_yaw + d_step_yaw
        raw_step_pitch = p_pitch + d_step_pitch
        return (
            raw_step_yaw,
            raw_step_pitch,
            "mode=pd, "
            f"kp=({kp_yaw:.2f}, {kp_pitch:.2f}), "
            f"kd=({self.head_kd_yaw:.3f}, {self.head_kd_pitch:.3f}), "
            f"p=({p_yaw:.1f}, {p_pitch:.1f}), "
            f"d=({d_step_yaw:.1f}, {d_step_pitch:.1f})",
        )

    def _sample_chassis_yaw_rate(self, head_controller):
        """采样底盘在 control_frame 下的 yaw 速率，用于判断是否降低锁点更新 alpha。

        这里只关心底盘是否正在快速转向，不参与头部角度求解。若 TF 查询失败，
        返回 None，锁点更新会继续依赖 residual 门控。
        """
        chassis_matrix = self._lookup_transform_matrix(
            head_controller,
            target_frame=self.control_frame,
            source_frame=self.chassis_frame,
        )
        if chassis_matrix is None:
            return None

        yaw_deg = math.degrees(tf_trans.euler_from_matrix(chassis_matrix)[2])
        now = time.monotonic()
        if self._last_chassis_yaw_deg is None or self._last_chassis_yaw_time is None:
            self._last_chassis_yaw_deg = yaw_deg
            self._last_chassis_yaw_time = now
            return None

        dt = now - self._last_chassis_yaw_time
        if dt <= 1e-6:
            return None

        yaw_delta = self._shortest_angle_delta_deg(yaw_deg, self._last_chassis_yaw_deg)
        yaw_rate = yaw_delta / dt
        self._last_chassis_yaw_deg = yaw_deg
        self._last_chassis_yaw_time = now
        return yaw_rate

    @staticmethod
    def _shortest_angle_delta_deg(current_deg, previous_deg):
        """返回 [-180, 180) 内的最短角度差，避免 yaw 跨 +/-180 时误判高速旋转。"""
        return (current_deg - previous_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def _point_distance(a, b):
        return math.sqrt(
            (a.x - b.x) * (a.x - b.x)
            + (a.y - b.y) * (a.y - b.y)
            + (a.z - b.z) * (a.z - b.z)
        )

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

    def _publish_debug_markers(self, head_controller, target_point_msg, target_to_head_matrix=None):
        """发布 YOLO 目标、camera 原点、目标连线和 camera x 轴方向线。

        注意：这里不参与头部控制，只读取与控制同源的 YOLO 目标和实机 TF。
        可视化默认画在 YOLO source_frame 下；如果配置 debug_frame=map，
        则会优先按 map<-melon_odom 与 base_link<-camera 分段组合显示，
        避免直接依赖不稳定或不连通的 map<-camera 整链 TF。
        """
        if not self.debug_enabled:
            return

        if self.debug_marker_pub is None:
            return

        source_frame = target_point_msg.header.frame_id
        display_frame = self.debug_frame or source_frame
        target_to_head = target_to_head_matrix
        if target_to_head is None and display_frame == "map" and source_frame == head_controller.head_frame:
            # debug_frame=map 且目标在 camera 下时，沿用分段 TF，避免 map->camera 整链跳变。
            target_to_head = self._build_split_tf_target_to_head(head_controller, display_frame)
            if target_to_head is None:
                return
            target_point = self._transform_point_with_matrix(
                target_to_head,
                [
                    target_point_msg.point.x,
                    target_point_msg.point.y,
                    target_point_msg.point.z,
                ],
            )
        elif source_frame == display_frame:
            # 目标已经在显示坐标系下，直接画，不需要 TF。
            target_point = Point(
                x=target_point_msg.point.x,
                y=target_point_msg.point.y,
                z=target_point_msg.point.z,
            )
        else:
            # 非默认显示坐标系，尽量使用标准 TF 转换。
            target_point = self._transform_point_between_frames(
                head_controller,
                target_point_msg,
                display_frame,
            )
            if target_point is None:
                return

        target_point_for_debug = PointStamped()
        target_point_for_debug.header.stamp = self.ros_node.now()
        target_point_for_debug.header.frame_id = display_frame
        target_point_for_debug.point = target_point
        if self.debug_point_pub is not None:
            self.debug_point_pub.publish(target_point_for_debug)

        # marker_id=0：橙色目标点。
        self.debug_marker_pub.publish(
            self._build_sphere_marker(
                marker_id=0,
                frame_id=display_frame,
                point=target_point,
                scale=0.16,
                color=(1.0, 0.2, 0.0, 1.0),
                namespace="head_track_yolo_target",
            )
        )

        if target_to_head is not None:
            # 已有 T_display_camera 时，直接用矩阵算 camera 原点和 camera x 轴终点。
            head_origin = self._transform_point_with_matrix(
                target_to_head,
                [0.0, 0.0, 0.0],
            )
            head_x_axis_end = self._transform_point_with_matrix(
                target_to_head,
                [self.axis_length_m, 0.0, 0.0],
            )
        else:
            # 没有分段矩阵时，退回标准 TF 查询 camera 在 display_frame 下的位置。
            head_origin = self._transform_head_point_to_frame(
                head_controller,
                display_frame,
                [0.0, 0.0, 0.0],
            )
            head_x_axis_end = self._transform_head_point_to_frame(
                head_controller,
                display_frame,
                [self.axis_length_m, 0.0, 0.0],
            )
        if head_origin is None or head_x_axis_end is None:
            return

        # marker_id=1：蓝色 camera 原点。
        self.debug_marker_pub.publish(
            self._build_sphere_marker(
                marker_id=1,
                frame_id=display_frame,
                point=head_origin,
                scale=0.10,
                color=(0.1, 0.5, 1.0, 1.0),
                namespace="head_track_yolo_origin",
            )
        )
        # marker_id=2：绿色“应该看向目标”的连线。
        self.debug_marker_pub.publish(
            self._build_line_marker(
                marker_id=2,
                frame_id=display_frame,
                start=head_origin,
                end=target_point,
                color=(0.1, 1.0, 0.2, 1.0),
                namespace="head_track_yolo_target_line",
            )
        )
        # marker_id=3：红色当前 camera x 轴方向，用来和绿色线比较是否对准。
        self.debug_marker_pub.publish(
            self._build_line_marker(
                marker_id=3,
                frame_id=display_frame,
                start=head_origin,
                end=head_x_axis_end,
                color=(1.0, 0.0, 0.0, 1.0),
                namespace="head_track_yolo_current_x_axis",
            )
        )

    def _build_split_tf_target_to_head(self, head_controller, target_frame):
        """分段构造 T_target_head，用于稳定获得 camera/head_frame 在 target_frame 下的位姿。

        camera 到 map 不能直接连通或 latest 时间不一致时，直接 transformPoint
        会失败或跳变。这里复用盯 map 点节点的思路，避开整链 TF 的时间戳问题：
        1. 查 T_map_melon_odom：底盘在 map 下的实时位姿。
        2. 查 T_base_link_camera：camera 相对机器人本体 base_link 的姿态。
        3. 假设 melon_odom 与 base_link 重合，组合：
           T_map_camera = T_map_melon_odom * T_base_link_camera

        该矩阵会用于两处：
        - control_mode=latched_map_target 时，把锁定的 map 目标重投影到当前 camera 下控制头部。
        - RViz Marker 中，在 map 下画出目标点、camera 原点、camera x 轴和连线。
        """
        target_to_chassis = self._lookup_transform_matrix(
            head_controller,
            target_frame=target_frame,
            source_frame=self.chassis_frame,
        )
        base_to_head = self._lookup_transform_matrix(
            head_controller,
            target_frame=head_controller.base_frame,
            source_frame=head_controller.head_frame,
        )
        if target_to_chassis is None or base_to_head is None:
            return None
        return tf_trans.concatenate_matrices(target_to_chassis, base_to_head)

    def _lookup_transform_matrix(self, head_controller, target_frame, source_frame):
        """查询 T_target_source 矩阵，将 source_frame 下的点转换到 target_frame。"""
        # rospy.Time(0) 表示取 TF 缓存中最新可用变换；这里为了避开整链时间戳不齐，
        # 只查短链 map->melon_odom 和 base_link->camera。
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
            self._log_throttled(
                "failure",
                f"[{self.config_label}] 分段可视化无法查询 {source_frame} -> "
                f"{target_frame}: {err}",
            )
            return None

        return tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(translation),
            tf_trans.quaternion_matrix(rotation),
        )

    @staticmethod
    def _transform_point_with_matrix(matrix, point_xyz):
        """用 T_target_source 矩阵把 source 点转换到 target，用于 RViz Marker。"""
        transformed = matrix.dot(
            [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0]
        )
        return Point(
            x=float(transformed[0]),
            y=float(transformed[1]),
            z=float(transformed[2]),
        )

    def _transform_point_between_frames(self, head_controller, point_msg, target_frame):
        """把 YOLO 目标点转换到 debug_frame；target_frame 等于原 frame 时直接返回。"""
        if point_msg.header.frame_id == target_frame:
            return Point(
                x=point_msg.point.x,
                y=point_msg.point.y,
                z=point_msg.point.z,
            )

        try:
            head_controller.tf_listener.waitForTransform(
                target_frame,
                point_msg.header.frame_id,
                self.ros_node.zero_time(),
                self.ros_node.duration(head_controller.tf_timeout),
            )
            transformed = head_controller.tf_listener.transformPoint(
                target_frame,
                point_msg,
            )
            return transformed.point
        except Exception as err:
            self._log_throttled(
                "failure",
                f"[{self.config_label}] 调试 Marker 无法转换 YOLO 目标 "
                f"{point_msg.header.frame_id} -> {target_frame}: {err}",
            )
            return None

    def _transform_head_point_to_frame(self, head_controller, target_frame, point_xyz):
        """把 camera/head_frame 下的点转换到 debug_frame，用于画当前头部状态。"""
        if target_frame == head_controller.head_frame:
            return Point(
                x=float(point_xyz[0]),
                y=float(point_xyz[1]),
                z=float(point_xyz[2]),
            )

        latest_tf_time = self.ros_node.zero_time()
        point_msg = PointStamped()
        point_msg.header.stamp = latest_tf_time
        point_msg.header.frame_id = head_controller.head_frame
        point_msg.point.x = float(point_xyz[0])
        point_msg.point.y = float(point_xyz[1])
        point_msg.point.z = float(point_xyz[2])
        try:
            head_controller.tf_listener.waitForTransform(
                target_frame,
                head_controller.head_frame,
                latest_tf_time,
                self.ros_node.duration(head_controller.tf_timeout),
            )
            transformed = head_controller.tf_listener.transformPoint(
                target_frame,
                point_msg,
            )
            return transformed.point
        except Exception as err:
            self._log_throttled(
                "failure",
                f"[{self.config_label}] 调试 Marker 无法转换 "
                f"{head_controller.head_frame} -> {target_frame}: {err}",
            )
            return None

    def _build_sphere_marker(self, marker_id, frame_id, point, scale, color, namespace):
        marker = Marker()
        marker.header.stamp = self.ros_node.now()
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    def _build_line_marker(self, marker_id, frame_id, start, end, color, namespace):
        marker = Marker()
        marker.header.stamp = self.ros_node.now()
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.points = [
            Point(x=start.x, y=start.y, z=start.z),
            Point(x=end.x, y=end.y, z=end.z),
        ]
        return marker

    def _log_throttled(self, kind, message):
        """按类别限频打印日志，避免 10Hz 控制循环把终端刷满。"""
        now = time.monotonic()
        if kind == "no_target":
            if now - self._last_no_target_log_time >= self.no_target_log_interval_sec:
                self.ros_node.get_logger().warning(message)
                self._last_no_target_log_time = now
            return

        if now - self._last_failure_log_time >= self.failure_log_interval_sec:
            self.ros_node.get_logger().warning(message)
            self._last_failure_log_time = now

    def describe_start(self):
        return (
            f"[{self.config_label}] 开始头部持续跟踪 YOLO 最近目标: "
            f"topic={self.yolo_topic}, "
            f"control_mode={self.control_mode}, "
            f"control_frame={self.control_frame}, "
            f"track_interval={self.track_interval_sec:.2f}s, "
            f"residual_fast={self.lock_residual_fast_m:.2f}m, "
            f"residual_slow={self.lock_residual_slow_m:.2f}m, "
            f"pending_confirm={self.lock_pending_confirm_count}, "
            f"debug_frame={self.debug_frame or 'source_frame'}, "
            f"debug_marker_topic={self.debug_marker_topic if self.debug_enabled else 'disabled'}"
        )
