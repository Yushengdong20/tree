#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque

import rospy
import tf
import tf.transformations as tf_trans
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from kuavo_humanoid_sdk.common.yolo_boxes import parse_yolo_boxes_string
from tree.constants import BASE_LINK_FRAME, MAP_FRAME


class YoloStringViewer:
    def __init__(self):
        rospy.init_node("yolo_boxes_printer")
        self.yolo_topic = rospy.get_param("~yolo_topic", "/yolo/target_boxes3d_string")
        self.marker_topic = rospy.get_param(
            "~marker_topic", "/move_box/show_yolo_string_markers"
        )
        self.target_frame = rospy.get_param("~target_frame", MAP_FRAME)
        self.base_frame = rospy.get_param("~base_frame", BASE_LINK_FRAME)
        self.odom_topic = rospy.get_param("~odom_topic", "melon_odom")
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 0.2))
        self.odom_history_duration_sec = max(
            float(rospy.get_param("~odom_history_duration_sec", 10.0)), 1.0
        )

        self.tf_listener = tf.TransformListener()
        self.latest_odom = None
        self.odom_history = deque()
        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True
        )
        rospy.Subscriber(self.odom_topic, Odometry, self._on_odom, queue_size=10)
        rospy.Subscriber(self.yolo_topic, String, self.callback, queue_size=1)

        rospy.loginfo(
            "show_yolo_string 已启动: yolo_topic=%s, marker_topic=%s, target_frame=%s, odom_topic=%s",
            self.yolo_topic,
            self.marker_topic,
            self.target_frame,
            self.odom_topic,
        )

    def _on_odom(self, msg):
        self.latest_odom = msg
        stamp_sec = self._odom_stamp_to_seconds(msg)
        self.odom_history.append((stamp_sec, msg))
        min_stamp_sec = stamp_sec - self.odom_history_duration_sec
        while len(self.odom_history) > 1 and self.odom_history[0][0] < min_stamp_sec:
            self.odom_history.popleft()

    def callback(self, msg):
        boxes = parse_yolo_boxes_string(msg.data)
        if not boxes:
            rospy.logwarn("收到空或非法 YOLO boxes String: topic=%s", self.yolo_topic)
            self._clear_markers()
            return

        self._print_boxes(boxes)
        self._publish_map_visualization(boxes)

    def _print_boxes(self, boxes):
        top_frame = boxes[0].get("frame_id", "") if boxes else ""
        top_stamp = boxes[0].get("stamp", 0.0) if boxes else 0.0

        print("\n================ YOLO Boxes3D ================")
        print(f"frame_id: {top_frame}")
        print(f"stamp:    {top_stamp}")
        print(f"box_count: {len(boxes)}")

        for i, box in enumerate(boxes):
            center_map, odom_delta_ms = self._solve_map_center(box)
            print(f"\nbox[{i}]")
            print(f"  frame_id : {box.get('frame_id')}")
            print(f"  stamp    : {box.get('stamp')}")
            print(f"  center   : {box['center']}")
            print(f"  quat     : {box['quat']}")
            print(f"  size     : {box['size']}")
            print(f"  score    : {box['score']:.3f}")
            print(f"  class_id : {box['class_id']}")
            if center_map is not None:
                print(
                    "  map      : [%.6f, %.6f, %.6f]  (matched odom Δt=%.1f ms)"
                    % (
                        center_map["x"],
                        center_map["y"],
                        center_map["z"],
                        odom_delta_ms,
                    )
                )
            else:
                print("  map      : <求解失败>")

        print("==============================================")

    def _publish_map_visualization(self, boxes):
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        marker_id = 1
        success_count = 0
        for index, box in enumerate(boxes):
            corners = self._box_corners_in_target_frame(box)
            if not corners:
                continue

            center, odom_delta_ms = self._solve_map_center(box)
            if center is None:
                continue

            outline = self._new_marker(
                marker_id, "show_yolo_string_outline", Marker.LINE_LIST
            )
            marker_id += 1
            outline.scale.x = 0.014
            self._set_color(outline, 0.20, 0.75, 1.0, 0.85)
            for start_idx, end_idx in self._box_edge_indices():
                outline.points.append(self._point_message(corners[start_idx]))
                outline.points.append(self._point_message(corners[end_idx]))
            marker_array.markers.append(outline)

            center_marker = self._new_marker(
                marker_id, "show_yolo_string_center", Marker.SPHERE
            )
            marker_id += 1
            center_marker.pose.position = Point(
                x=float(center["x"]),
                y=float(center["y"]),
                z=float(center["z"]),
            )
            center_marker.scale.x = 0.06
            center_marker.scale.y = 0.06
            center_marker.scale.z = 0.06
            self._set_color(center_marker, 1.0, 0.9, 0.0, 1.0)
            marker_array.markers.append(center_marker)

            top_z = max(corner[2] for corner in corners)
            text_marker = self._new_marker(
                marker_id, "show_yolo_string_text", Marker.TEXT_VIEW_FACING
            )
            marker_id += 1
            text_marker.pose.position.x = float(center["x"])
            text_marker.pose.position.y = float(center["y"])
            text_marker.pose.position.z = top_z + 0.10
            text_marker.scale.z = 0.08
            self._set_color(text_marker, 1.0, 1.0, 1.0, 1.0)
            text_marker.text = (
                f"YOLO #{index}\n"
                f"map=({center['x']:.2f}, {center['y']:.2f}, {center['z']:.2f})\n"
                f"class={box.get('class_id', '?')} score={float(box.get('score', 0.0)):.2f}\n"
                f"odom_dt={odom_delta_ms:.1f}ms"
            )
            marker_array.markers.append(text_marker)
            success_count += 1

        self.marker_pub.publish(marker_array)
        rospy.loginfo(
            "已发布 YOLO map 可视化: topic=%s, total=%d, success=%d, target_frame=%s",
            self.marker_topic,
            len(boxes),
            success_count,
            self.target_frame,
        )

    def _box_corners_in_target_frame(self, box):
        size = box.get("size") or [0.0, 0.0, 0.0]
        quat = box.get("quat") or [0.0, 0.0, 0.0, 1.0]
        center = box.get("center") or [0.0, 0.0, 0.0]
        source_frame = box.get("frame_id") or self.base_frame

        try:
            sx, sy, sz = [abs(float(v)) for v in size[:3]]
            if min(sx, sy, sz) < 0.01:
                return None
            rotation = tf_trans.quaternion_matrix(
                [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            )[:3, :3]
        except Exception as err:
            rospy.logwarn_throttle(1.0, "YOLO箱体姿态非法，跳过可视化: %s", err)
            return None

        half = [sx * 0.5, sy * 0.5, sz * 0.5]
        corners = []
        for dx in (-half[0], half[0]):
            for dy in (-half[1], half[1]):
                for dz in (-half[2], half[2]):
                    rotated = rotation.dot([dx, dy, dz])
                    source_point = {
                        "x": float(center[0]) + float(rotated[0]),
                        "y": float(center[1]) + float(rotated[1]),
                        "z": float(center[2]) + float(rotated[2]),
                    }
                    map_point, _ = self._transform_point_to_target(source_point, source_frame)
                    if map_point is None:
                        return None
                    corners.append(
                        [
                            float(map_point["x"]),
                            float(map_point["y"]),
                            float(map_point["z"]),
                        ]
                    )
        return corners

    def _solve_map_center(self, box):
        source_frame = box.get("frame_id") or self.base_frame
        center = box.get("center") or [0.0, 0.0, 0.0]
        return self._transform_point_to_target(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "_stamp_sec": float(box.get("stamp", 0.0)),
            },
            source_frame,
        )

    def _transform_point_to_target(self, point_xyz, source_frame):
        if source_frame == self.target_frame:
            return {
                "x": float(point_xyz["x"]),
                "y": float(point_xyz["y"]),
                "z": float(point_xyz["z"]),
            }, 0.0

        try:
            odom_msg, odom_delta_ms = self._get_nearest_odom_by_stamp_sec(
                point_xyz.get("_stamp_sec")
            )
            if odom_msg is None:
                raise RuntimeError("缺少可用 odom，无法执行 source->base->map 转换")

            self.tf_listener.waitForTransform(
                self.base_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_sec),
            )
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rospy.Time(0)
            point_stamped.point.x = float(point_xyz["x"])
            point_stamped.point.y = float(point_xyz["y"])
            point_stamped.point.z = float(point_xyz["z"])
            transformed = self.tf_listener.transformPoint(self.base_frame, point_stamped)
            map_point = self._transform_base_point_to_map_with_odom(
                odom_msg,
                {
                    "x": float(transformed.point.x),
                    "y": float(transformed.point.y),
                    "z": float(transformed.point.z),
                },
            )
            return {
                "x": float(map_point["x"]),
                "y": float(map_point["y"]),
                "z": float(map_point["z"]),
            }, odom_delta_ms
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
            RuntimeError,
        ) as err:
            rospy.logwarn_throttle(
                1.0,
                "YOLO点按 source->base->map 链路从 %s 转到 %s 失败，跳过该箱体: %s",
                source_frame,
                self.target_frame,
                err,
            )
            return None, 0.0

    def _clear_markers(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def _new_marker(self, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _set_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    @staticmethod
    def _point_message(point_xyz):
        return Point(
            x=float(point_xyz[0]),
            y=float(point_xyz[1]),
            z=float(point_xyz[2]),
        )

    @staticmethod
    def _box_edge_indices():
        return [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

    def _get_nearest_odom_by_stamp_sec(self, stamp_sec):
        if stamp_sec is None:
            if self.latest_odom is None:
                return None, 0.0
            return self.latest_odom, 0.0
        if not self.odom_history:
            if self.latest_odom is None:
                return None, 0.0
            return self.latest_odom, 0.0

        target_sec = float(stamp_sec)
        nearest_msg = None
        nearest_delta = None
        for history_stamp_sec, history_msg in self.odom_history:
            delta = abs(history_stamp_sec - target_sec)
            if nearest_delta is None or delta < nearest_delta:
                nearest_delta = delta
                nearest_msg = history_msg
        if nearest_msg is None:
            if self.latest_odom is None:
                return None, 0.0
            return self.latest_odom, 0.0
        return nearest_msg, float(nearest_delta) * 1000.0

    def _transform_base_point_to_map_with_odom(self, odom_msg, base_position):
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

    @staticmethod
    def _odom_stamp_to_seconds(odom_msg):
        header = getattr(odom_msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return 0.0
        if hasattr(stamp, "secs"):
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


if __name__ == "__main__":
    viewer = YoloStringViewer()
    rospy.spin()
