"""让头部持续盯住 map/base_link 等坐标系下的固定点，并发布 RViz 调试可视化。"""

import ast
import time

import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import Point, PointStamped
from py_trees.common import Status
from visualization_msgs.msg import Marker

from ..base import TimedMockAction


class MoveBoxTrackHeadToMapPoint(TimedMockAction):
    """持续让头部跟踪 JSON 中配置的固定目标点。

    控制链路有两种模式：
    - tf: 直接复用 HeadController.turn_to_target()，由 TF 查询 target_frame -> camera。
    - split_tf: 分开查询 target_frame -> chassis_frame 与 base_frame -> camera，
      再自己组合成 target_frame -> camera，避开整条 TF 链 latest 时间不一致导致的跳变。

    RViz 调试只读取实机 TF，不重复实现头部控制算法：
    - 橙色球：目标点。
    - 蓝色球：camera/head_frame 原点。
    - 绿色线：camera/head_frame 原点到目标点。
    - 红色线：camera/head_frame 当前 x 轴方向。

    当红色线和绿色线方向重合时，就表示 camera/head_frame 的 x 轴基本对准目标点。
    """

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.tracking_mode = str(params.get("tracking_mode", "tf")).strip().lower()
        self.target_frame = str(params.get("target_frame", "map")).strip()
        self.target_point = self._parse_point(params.get("target_point", [2.0, 1.0, 1.0]))
        self.chassis_frame = str(params.get("chassis_frame", "melon_odom")).strip()
        # 不在节点内部限流，头部控制频率直接跟随行为树 tick。
        # tick_debug 日志仍可用于确认并行流程里本节点是否被持续 tick。
        self.failure_log_interval_sec = float(params.get("failure_log_interval_sec", 1.0))
        self.tick_debug_enabled = self._to_bool(params.get("tick_debug_enabled", False))
        self.tick_log_interval_sec = float(params.get("tick_log_interval_sec", 1.0))
        self.axis_length_m = float(params.get("axis_length_m", 1.0))
        # 与 YOLO 盯箱节点共用 HeadController 的 P/PD 增量控制：
        # 这里的参数只影响“目标点已经转换到 camera/head_frame 后怎么转头”，
        # 不参与 map 固定点 TF 组合和 RViz 可视化。
        self.max_delta_yaw_deg = self._optional_float(params.get("max_delta_yaw_deg", None))
        self.max_delta_pitch_deg = self._optional_float(params.get("max_delta_pitch_deg", None))
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
        self.debug_enabled = self._to_bool(params.get("debug_enabled", True))
        self.debug_point_topic = str(
            params.get("debug_point_topic", "/head_x_axis_target_point")
        ).strip()
        self.debug_marker_topic = str(
            params.get("debug_marker_topic", "/head_x_axis_alignment_marker")
        ).strip()

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

        self._last_track_time = 0.0
        self._last_failure_log_time = 0.0
        self._last_tick_log_time = 0.0
        self._tick_count = 0
        self._reset_head_controller_on_next_tick = True
        self._skip_logged = False

    def initialise(self):
        super().initialise()
        self._last_track_time = 0.0
        self._last_tick_log_time = 0.0
        self._tick_count = 0
        self._reset_head_controller_on_next_tick = True
        self._skip_logged = False

    def update(self):
        # 这个节点是持续 RUNNING 的跟踪节点，每次被行为树 tick 到都会尝试刷新头部目标。
        # 如果开启跳过头部动作，只保留节点运行状态，不会发布头部控制命令。
        now = time.monotonic()
        self._tick_count += 1
        self._log_tick_debug(now)
        if self.should_skip_head_motion():
            if not self._skip_logged:
                self.log_skip_head_motion()
                self._skip_logged = True
            return Status.RUNNING

        self._last_track_time = now

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "head_controller"):
            self._log_failure(
                f"[{self.config_label}] services 或 head_controller 缺失: key={self.services_key}"
            )
            return Status.RUNNING

        head_controller = services.head_controller
        if self._reset_head_controller_on_next_tick:
            head_controller.reset_tracking_control_state()
            self._reset_head_controller_on_next_tick = False
        target_point_msg = self._build_target_point_msg()
        split_target_to_head = None
        if self.tracking_mode == "split_tf":
            # split_tf 模式：不再让 TF 一次性查完整 target_frame -> camera。
            # 这里会先分段构造出 target_frame -> camera 的矩阵，并直接得到
            # camera/head_frame 下的目标点，供 HeadController 解算 yaw/pitch。
            target_in_head, split_target_to_head = self._build_split_tf_target_in_head(
                head_controller
            )
            self._publish_debug_markers(
                head_controller,
                target_point_msg,
                target_to_head_matrix=split_target_to_head,
            )
            ok = (
                self._turn_to_head_frame_point(head_controller, target_in_head)
                if target_in_head is not None
                else False
            )
        else:
            # tf 模式：保持旧逻辑，完全交给 HeadController 查询 target_frame -> camera。
            # 如果完整 TF 链时间戳不一致，这个模式更容易复现 camera 原点跳变。
            self._publish_debug_markers(head_controller, target_point_msg)
            target_in_head = self._target_point_to_head_frame(head_controller, target_point_msg)
            ok = (
                self._turn_to_head_frame_point(head_controller, target_in_head)
                if target_in_head is not None
                else False
            )
        if not ok:
            self._log_failure(
                f"[{self.config_label}] 头部跟踪执行失败: mode={self.tracking_mode}"
            )
        return Status.RUNNING

    @staticmethod
    def _parse_point(value):
        """解析 JSON 字符串/list 格式的 [x, y, z] 点。"""
        if isinstance(value, str):
            value = ast.literal_eval(value)
        point = [float(item) for item in value]
        if len(point) != 3:
            raise ValueError("target_point must be [x, y, z]")
        return point

    def _log_failure(self, message):
        now = time.monotonic()
        if now - self._last_failure_log_time >= self.failure_log_interval_sec:
            self.ros_node.get_logger().warning(message)
            self._last_failure_log_time = now

    def _log_tick_debug(self, now):
        """节流打印 update() 被 tick 的情况，用来排查并行节点是否被阻塞。"""
        if not self.tick_debug_enabled:
            return
        if now - self._last_tick_log_time < self.tick_log_interval_sec:
            return

        elapsed_from_control = (
            -1.0 if self._last_track_time <= 0.0 else now - self._last_track_time
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] tick alive: "
            f"count={self._tick_count}, mode={self.tracking_mode}, "
            f"elapsed_from_last_control={elapsed_from_control:.3f}s, "
            f"control_rate=tree_tick"
        )
        self._last_tick_log_time = now

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        return float(text)

    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def _build_target_point_msg(self):
        """把 JSON 目标点发布成 PointStamped，方便 rostopic/RViz 检查输入点。"""
        point_msg = PointStamped()
        point_msg.header.stamp = self.ros_node.now()
        point_msg.header.frame_id = self.target_frame
        point_msg.point.x = self.target_point[0]
        point_msg.point.y = self.target_point[1]
        point_msg.point.z = self.target_point[2]
        return point_msg

    def _turn_to_head_frame_point(self, head_controller, target_in_head):
        """复用 HeadController 通用 P/PD 控制，保持与 YOLO 盯箱节点一致的调参语义。"""
        return head_controller.turn_to_head_frame_point_controlled(
            target_in_head,
            control_mode=self.head_control_mode,
            kp_yaw=self.head_kp_yaw,
            kp_pitch=self.head_kp_pitch,
            kd_yaw=self.head_kd_yaw,
            kd_pitch=self.head_kd_pitch,
            derivative_filter_alpha=self.head_derivative_filter_alpha,
            max_delta_yaw_deg=self.max_delta_yaw_deg,
            max_delta_pitch_deg=self.max_delta_pitch_deg,
            log_prefix=f"[{self.config_label}] 固定点头部限幅控制",
        )

    def _target_point_to_head_frame(self, head_controller, target_point_msg):
        """tf 模式下把固定点转换到 camera/head_frame，再交给通用控制器。"""
        latest_tf_time = self.ros_node.zero_time()
        target_point_msg.header.stamp = latest_tf_time
        try:
            head_controller.tf_listener.waitForTransform(
                head_controller.head_frame,
                target_point_msg.header.frame_id,
                latest_tf_time,
                self.ros_node.duration(head_controller.tf_timeout),
            )
            return head_controller.tf_listener.transformPoint(
                head_controller.head_frame,
                target_point_msg,
            )
        except Exception as err:
            self._log_failure(
                f"[{self.config_label}] 目标点从 {target_point_msg.header.frame_id} "
                f"转到 {head_controller.head_frame} 失败: {err}"
            )
            return None

    def _publish_debug_markers(
        self,
        head_controller,
        target_point_msg,
        target_to_head_matrix=None,
    ):
        """发布目标点、camera 原点、目标连线和 camera x 轴方向线。

        这里完全使用 TF 中的实机数据，不推导 yaw/pitch：
        - 蓝色球 = 当前 head_frame 原点。
        - 红色线 = 当前 head_frame x 轴。
        - 绿色线 = 当前 head_frame 原点到目标点。
        """
        if not self.debug_enabled:
            return

        # PointStamped 用来确认 JSON 输入点本身是否正确；
        # Marker 用来同时看目标点、camera 原点、camera x 轴与目标连线。
        if self.debug_point_pub is not None:
            self.debug_point_pub.publish(target_point_msg)
        if self.debug_marker_pub is None:
            return

        # 橙色球：目标点，始终画在 target_frame 下，例如 map 下的 [2, 1, 1]。
        self.debug_marker_pub.publish(
            self._build_sphere_marker(
                marker_id=0,
                frame_id=self.target_frame,
                point=target_point_msg.point,
                scale=0.16,
                color=(1.0, 0.2, 0.0, 1.0),
                namespace="head_x_axis_target",
            )
        )

        if target_to_head_matrix is None and self.tracking_mode == "split_tf":
            # split_tf 下如果组合矩阵没拿到，说明分段 TF 查询失败。
            # 此时只保留橙色目标点，不退回旧整链 TF，避免 RViz 混入两套逻辑。
            return

        if target_to_head_matrix is not None:
            # split_tf 可视化：使用和控制完全相同的组合矩阵 T_target_head。
            # [0,0,0] 是 camera/head_frame 原点，乘 T_target_head 后得到 target_frame 坐标。
            head_origin = self._transform_point_with_matrix(
                target_to_head_matrix,
                [0.0, 0.0, 0.0],
            )
            # [axis_length,0,0] 是 camera/head_frame 的 x 轴端点。
            # 红线 start=head_origin, end=head_x_axis_end 表示当前 camera x 轴方向。
            head_x_axis_end = self._transform_point_with_matrix(
                target_to_head_matrix,
                [self.axis_length_m, 0.0, 0.0],
            )
        else:
            # tf 可视化：直接用完整 TF 链把 camera 下的点转换到 target_frame。
            # 这条路径主要保留用于与旧逻辑对比。
            head_origin = self._transform_head_point_to_target_frame(
                head_controller,
                [0.0, 0.0, 0.0],
            )
            head_x_axis_end = self._transform_head_point_to_target_frame(
                head_controller,
                [self.axis_length_m, 0.0, 0.0],
            )
        if head_origin is None or head_x_axis_end is None:
            return

        self.debug_marker_pub.publish(
            self._build_sphere_marker(
                marker_id=1,
                frame_id=self.target_frame,
                point=head_origin,
                scale=0.10,
                color=(0.1, 0.5, 1.0, 1.0),
                namespace="head_x_axis_origin",
            )
        )
        self.debug_marker_pub.publish(
            self._build_line_marker(
                marker_id=2,
                frame_id=self.target_frame,
                start=head_origin,
                end=target_point_msg.point,
                color=(0.1, 1.0, 0.2, 1.0),
                namespace="head_x_axis_target_line",
            )
        )
        self.debug_marker_pub.publish(
            self._build_line_marker(
                marker_id=3,
                frame_id=self.target_frame,
                start=head_origin,
                end=head_x_axis_end,
                color=(1.0, 0.0, 0.0, 1.0),
                namespace="head_x_axis_current",
            )
        )

    def _build_split_tf_target_in_head(self, head_controller):
        """方案 A：分段组合 TF，再把 map 目标点转换到 camera/head_frame。

        旧链路直接查询 target_frame -> head_frame，TF 会自行拼完整链路：
            target_frame -> melon_odom -> base_link -> ... -> head_frame

        在各段 TF 时间戳不一致时，这条完整 latest 查询可能出现跳变。
        split_tf 模式改为显式分两段查询：
            1. target_frame -> chassis_frame，使用底盘世界位姿。
            2. base_frame -> head_frame，使用本体内部 TF。
        然后认为 chassis_frame 与 base_frame 重合，自行组合：
            T_target_head = T_target_chassis * T_base_head

        最终仍把目标点转到 head_frame，复用 HeadController.turn_to_head_frame_point()
        进行 yaw/pitch 解算和发布。
        """
        base_frame = head_controller.base_frame
        head_frame = head_controller.head_frame

        # 1. 查询 T_target_chassis。
        # 约定：T_A_B 表示“把 B 坐标系下的点转换到 A 坐标系”。
        # 如果 target_frame=map、chassis_frame=melon_odom，这里得到的就是：
        #   T_map_melon_odom
        # 它代表底盘实时位姿在世界系下的位置和朝向。
        target_to_chassis = self._lookup_transform_matrix(
            head_controller,
            target_frame=self.target_frame,
            source_frame=self.chassis_frame,
        )

        # 2. 查询 T_base_head。
        # 如果 base_frame=base_link、head_frame=camera，这里得到的就是：
        #   T_base_link_camera
        # 它只依赖机器人本体内部 TF，用来描述 camera 相对 base_link 的安装/头部姿态。
        base_to_head = self._lookup_transform_matrix(
            head_controller,
            target_frame=base_frame,
            source_frame=head_frame,
        )
        if target_to_chassis is None or base_to_head is None:
            return None, None

        # 3. 组合 T_target_head。
        # 当前方案 A 的关键假设是：chassis_frame 与 base_frame 重合。
        # 也就是把 melon_odom 当作 base_link 使用，中间不再额外查询/发布
        # melon_odom -> base_link。
        #
        # 因此：
        #   T_target_head = T_target_chassis * T_base_head
        #
        # 对 target_frame=map 的典型配置来说就是：
        #   T_map_camera = T_map_melon_odom * T_base_link_camera
        #
        # concatenate_matrices(A, B) 的含义与矩阵左乘一致：
        #   point_in_target = A * B * point_in_head
        target_to_head = tf_trans.concatenate_matrices(target_to_chassis, base_to_head)

        # 4. 控制解算需要“目标点在 camera/head_frame 下的位置”。
        # 现在手里有 T_target_head，它能把 camera 点转换到 map。
        # 反过来要把 map 点转换到 camera，就取逆矩阵：
        #   T_head_target = inverse(T_target_head)
        head_to_target = tf_trans.inverse_matrix(target_to_head)

        # 5. 把 JSON 目标点 [x,y,z] 从 target_frame 转换到 head_frame。
        # 齐次坐标最后的 1.0 用于让平移也参与矩阵运算。
        # 得到的 target_xyz_in_head 会交给 HeadController.turn_to_head_frame_point()：
        #   x: 目标在 camera 前方的距离
        #   y: 目标相对 camera x 轴的左右偏差
        #   z: 目标相对 camera x 轴的上下偏差
        target_xyz_in_head = head_to_target.dot(
            [self.target_point[0], self.target_point[1], self.target_point[2], 1.0]
        )

        # 6. 封装成 HeadController 需要的 PointStamped。
        # 注意这里 frame_id=head_frame，是因为 point 已经在 camera/head_frame 下。
        target_in_head = PointStamped()
        target_in_head.header.stamp = self.ros_node.zero_time()
        target_in_head.header.frame_id = head_frame
        target_in_head.point.x = float(target_xyz_in_head[0])
        target_in_head.point.y = float(target_xyz_in_head[1])
        target_in_head.point.z = float(target_xyz_in_head[2])
        return target_in_head, target_to_head

    def _lookup_transform_matrix(self, head_controller, target_frame, source_frame):
        """查询 T_target_source 矩阵，将 source_frame 下的点转换到 target_frame。"""
        latest_tf_time = self.ros_node.zero_time()
        try:
            # waitForTransform/lookupTransform 的方向是：
            #   lookupTransform(target_frame, source_frame, time)
            # 返回 source_frame 在 target_frame 下的平移和旋转。
            # 也就是 T_target_source。
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
            self._log_failure(
                f"[{self.config_label}] split_tf 无法查询 {source_frame} -> "
                f"{target_frame}: {err}"
            )
            return None

        # tf 返回 translation + quaternion，这里显式转成 4x4 齐次矩阵。
        # translation_matrix 负责平移，quaternion_matrix 负责旋转。
        # T = Translate * Rotate，表示先在 source 坐标系表达点，再转换到 target。
        return tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(translation),
            tf_trans.quaternion_matrix(rotation),
        )

    @staticmethod
    def _transform_point_with_matrix(matrix, point_xyz):
        """用 T_target_source 矩阵把 source 点转换到 target，用于 RViz Marker。"""
        # 输入 point_xyz 是 source_frame 下的普通三维点。
        # 追加 1.0 变成齐次坐标后，矩阵乘法会同时处理旋转和平移。
        transformed = matrix.dot(
            [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0]
        )
        return Point(
            x=float(transformed[0]),
            y=float(transformed[1]),
            z=float(transformed[2]),
        )

    def _transform_head_point_to_target_frame(self, head_controller, point_xyz):
        """把 head_frame 下的点转换到 target_frame，用于 RViz 画当前 camera x 轴。"""
        # 旧 tf 模式可视化专用：
        # 直接让 TF 查询完整 target_frame <- head_frame 链。
        # split_tf 模式不走这里，避免混入整链 latest 查询导致的跳变。
        latest_tf_time = self.ros_node.zero_time()
        point_msg = PointStamped()
        point_msg.header.stamp = latest_tf_time
        point_msg.header.frame_id = head_controller.head_frame
        point_msg.point.x = float(point_xyz[0])
        point_msg.point.y = float(point_xyz[1])
        point_msg.point.z = float(point_xyz[2])
        try:
            head_controller.tf_listener.waitForTransform(
                self.target_frame,
                head_controller.head_frame,
                latest_tf_time,
                self.ros_node.duration(head_controller.tf_timeout),
            )
            transformed = head_controller.tf_listener.transformPoint(
                self.target_frame,
                point_msg,
            )
            return transformed.point
        except Exception as err:
            self._log_failure(
                f"[{self.config_label}] 调试 Marker 无法转换 {head_controller.head_frame} -> "
                f"{self.target_frame}: {err}"
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

    def describe_start(self):
        return (
            f"[{self.config_label}] 开始头部持续跟踪固定点: "
            f"mode={self.tracking_mode}, frame={self.target_frame}, "
            f"point={self.target_point}, chassis_frame={self.chassis_frame}, "
            f"debug_marker_topic={self.debug_marker_topic if self.debug_enabled else 'disabled'}"
        )
