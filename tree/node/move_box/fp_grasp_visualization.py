"""FoundationPose 箱体与抓取目标 RViz 可视化工具。

这里只做诊断显示，不改变任何黑板数据或实际抓取目标。
输入的 FP 箱体中心、方向轴、抓取点仍然保持原来的 base_link 计算链路；
发布 marker 前再用当前 odom 位姿转换到 map，方便在全局视角下确认落点。
这里和 SelectAndPublishHighestYoloBox / ArmsToPose 夹爪诊断保持同一条转换链路：
base_link 与 melon_odom 视为重合，使用 odom.pose 的完整 4x4 矩阵构造
``map <- base_link``，避免 2D yaw 近似和其它可视化产生厘米级差异。
"""

import numpy as np
import tf.transformations as tf_trans
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME


def publish_fp_box_and_targets(
    *,
    ros_node,
    publisher,
    topic,
    config_label,
    odom_transformer,
    services,
    box_center,
    box_axes,
    strategy,
    grasp_pair=None,
    target_points=None,
    include_grasp_targets=False,
):
    """发布 FP 箱体轮廓和可选抓取目标点。

    :param include_grasp_targets: False 时只显示 FP 箱体本体；True 时显示抓取点/目标点。
    """
    if publisher is None:
        return

    center_base = _as_vector(box_center)
    left_axis_base = _normalize(_as_vector(box_axes.get("left") if box_axes else None))
    up_axis_base = _normalize(_as_vector(box_axes.get("up") if box_axes else None))
    raw_front_axis_base = _get_front_axis(services, left_axis_base, up_axis_base)
    if (
        center_base is None
        or left_axis_base is None
        or up_axis_base is None
        or raw_front_axis_base is None
    ):
        ros_node.get_logger().warning(f"[{config_label}] FP可视化跳过: center/axes 无效")
        return

    odom_msg = odom_transformer.get_latest_odom() if odom_transformer is not None else None
    if odom_msg is None:
        ros_node.get_logger().warning(f"[{config_label}] FP可视化跳过: 尚未获得 map 下底盘位姿")
        return
    map_from_base = _map_from_odom_message(odom_msg)

    left_axis_base = _normalize(
        left_axis_base - np.dot(left_axis_base, up_axis_base) * up_axis_base
    )
    front_axis_base = _normalize(np.cross(left_axis_base, up_axis_base))
    if front_axis_base is None:
        return
    if np.dot(front_axis_base, raw_front_axis_base) < 0.0:
        front_axis_base = -front_axis_base

    center = _point_base_to_map(center_base, map_from_base)
    left_axis = _axis_base_to_map(center_base, left_axis_base, map_from_base)
    front_axis = _axis_base_to_map(center_base, front_axis_base, map_from_base)
    up_axis = _axis_base_to_map(center_base, up_axis_base, map_from_base)
    if left_axis is None or front_axis is None or up_axis is None:
        ros_node.get_logger().warning(f"[{config_label}] FP可视化跳过: 坐标轴转换失败")
        return
    box_size, box_size_source = _get_fp_box_size(services)

    marker_array = MarkerArray()
    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)

    marker_id = 1
    corners = _fp_box_corners(center, left_axis, front_axis, up_axis, box_size)
    box_marker = _new_marker(ros_node, marker_id, "fp_box_outline", Marker.LINE_LIST)
    marker_id += 1
    box_marker.scale.x = 0.025
    _set_marker_color(box_marker, 0.0, 0.85, 1.0, 0.95)
    for start_index, end_index in _box_edge_indices():
        box_marker.points.append(_point_message(corners[start_index]))
        box_marker.points.append(_point_message(corners[end_index]))
    marker_array.markers.append(box_marker)

    center_marker = _new_marker(ros_node, marker_id, "fp_box_center", Marker.SPHERE)
    marker_id += 1
    center_marker.pose.position = _point_message(center)
    center_marker.scale.x = center_marker.scale.y = center_marker.scale.z = 0.08
    _set_marker_color(center_marker, 1.0, 1.0, 0.0, 1.0)
    marker_array.markers.append(center_marker)

    for label, axis, color in (
        ("left_axis", left_axis, (1.0, 0.1, 1.0)),
        ("front_axis", front_axis, (0.1, 1.0, 0.1)),
        ("up_axis", up_axis, (1.0, 0.7, 0.1)),
    ):
        marker_id = _append_axis_marker(ros_node, marker_array, marker_id, label, center, axis, color)

    if include_grasp_targets:
        left_grasp = None
        right_grasp = None
        if grasp_pair is not None and len(grasp_pair) >= 2:
            left_grasp = _as_vector(grasp_pair[0])
            right_grasp = _as_vector(grasp_pair[1])
        marker_id = _append_point_marker(
            ros_node,
            marker_array,
            marker_id,
            "left_grasp",
            _point_base_to_map(left_grasp, map_from_base) if left_grasp is not None else None,
            (1.0, 0.1, 1.0),
            "LEFT GRASP",
            0.095,
        )
        marker_id = _append_point_marker(
            ros_node,
            marker_array,
            marker_id,
            "right_grasp",
            _point_base_to_map(right_grasp, map_from_base) if right_grasp is not None else None,
            (1.0, 0.45, 0.05),
            "RIGHT GRASP",
            0.095,
        )

        target_colors = (
            (0.25, 0.75, 1.0),
            (0.2, 1.0, 0.45),
            (1.0, 0.85, 0.15),
            (1.0, 0.35, 0.15),
            (0.75, 0.45, 1.0),
            (0.9, 0.9, 0.9),
        )
        for index, (label, point_base) in enumerate((target_points or {}).items()):
            point = _as_vector(point_base)
            marker_id = _append_point_marker(
                ros_node,
                marker_array,
                marker_id,
                f"target_{label}",
                _point_base_to_map(point, map_from_base) if point is not None else None,
                target_colors[index % len(target_colors)],
                label,
                0.065,
            )

    text = _new_marker(ros_node, marker_id, "fp_debug_text", Marker.TEXT_VIEW_FACING)
    text.pose.position.x = float(center[0])
    text.pose.position.y = float(center[1])
    text.pose.position.z = float(center[2] + box_size[2] * 0.5 + 0.18)
    text.scale.z = 0.10
    _set_marker_color(text, 1.0, 1.0, 1.0, 1.0)
    text.text = (
        "FP DEBUG\n"
        f"center_map=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n"
        f"strategy={strategy}\n"
        f"box=({box_size[0]:.2f}, {box_size[1]:.2f}, {box_size[2]:.2f})\n"
        f"source={box_size_source}"
    )
    marker_array.markers.append(text)

    publisher.publish(marker_array)
    ros_node.get_logger().info(
        f"[{config_label}] 已发布FP RViz标记: topic={topic}, "
        f"center_map=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), "
        f"include_grasp_targets={include_grasp_targets}, "
        f"box_size_source={box_size_source}"
    )


def clear_marker_array(publisher):
    if publisher is None:
        return
    marker_array = MarkerArray()
    marker = Marker()
    marker.action = Marker.DELETEALL
    marker_array.markers.append(marker)
    publisher.publish(marker_array)


def _get_front_axis(services, left_axis, up_axis):
    front_axis = None
    if services is not None and hasattr(services, "box_detector"):
        getter = getattr(services.box_detector, "get_latest_box_front_axis", None)
        if callable(getter):
            front_axis = _normalize(_as_vector(getter()))
    if front_axis is None and left_axis is not None and up_axis is not None:
        front_axis = _normalize(np.cross(left_axis, up_axis))
    return front_axis


def _get_fp_box_size(services):
    if services is not None and hasattr(services, "box_detector"):
        detector = services.box_detector
        getter = getattr(detector, "get_latest_box_size", None)
        if callable(getter):
            parsed = _parse_box_size(getter())
            if parsed is not None:
                return parsed, "box_detector"
        parsed = _parse_box_size(
            {
                "x": getattr(detector, "box_size_x", None),
                "y": getattr(detector, "box_size_y", None),
                "z": getattr(detector, "box_size_z", None),
            }
        )
        if parsed is not None:
            return parsed, "box_detector_attrs"
    return np.array([0.60, 0.40, 0.34], dtype=float), "fallback_default"


def _parse_box_size(raw_size):
    if raw_size is None:
        return None
    try:
        if isinstance(raw_size, dict):
            size = np.array([raw_size.get("x"), raw_size.get("y"), raw_size.get("z")], dtype=float)
        else:
            if len(raw_size) < 3:
                return None
            size = np.array(raw_size[:3], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return None
    return size


def _map_from_odom_message(odom_msg):
    """从 odom.pose 构造 ``map <- base_link``，与选箱节点保持一致。"""
    odom_position = odom_msg.pose.pose.position
    odom_orientation = odom_msg.pose.pose.orientation
    return tf_trans.concatenate_matrices(
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


def _point_base_to_map(point, map_from_base):
    transformed = map_from_base.dot([float(point[0]), float(point[1]), float(point[2]), 1.0])
    return np.array([float(transformed[0]), float(transformed[1]), float(transformed[2])], dtype=float)


def _axis_base_to_map(center_base, axis_base, map_from_base):
    center_map = _point_base_to_map(center_base, map_from_base)
    end_map = _point_base_to_map(np.array(center_base, dtype=float) + np.array(axis_base, dtype=float), map_from_base)
    return _normalize(end_map - center_map)


def _fp_box_corners(center, left_axis, front_axis, up_axis, box_size):
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(
                    center
                    + left_axis * sx * box_size[0] * 0.5
                    + front_axis * sy * box_size[1] * 0.5
                    + up_axis * sz * box_size[2] * 0.5
                )
    return corners


def _append_axis_marker(ros_node, marker_array, marker_id, label, center, axis, color):
    arrow = _new_marker(ros_node, marker_id, f"fp_{label}", Marker.ARROW)
    marker_id += 1
    arrow.scale.x = 0.04
    arrow.scale.y = 0.08
    arrow.scale.z = 0.08
    arrow.points = [_point_message(center), _point_message(center + axis * 0.35)]
    _set_marker_color(arrow, color[0], color[1], color[2], 0.95)
    marker_array.markers.append(arrow)
    return marker_id


def _append_point_marker(ros_node, marker_array, marker_id, namespace, point, color, label, scale):
    if point is None:
        return marker_id
    sphere = _new_marker(ros_node, marker_id, namespace, Marker.SPHERE)
    marker_id += 1
    sphere.pose.position = _point_message(point)
    sphere.scale.x = sphere.scale.y = sphere.scale.z = float(scale)
    _set_marker_color(sphere, color[0], color[1], color[2], 1.0)
    marker_array.markers.append(sphere)

    text = _new_marker(ros_node, marker_id, f"{namespace}_text", Marker.TEXT_VIEW_FACING)
    marker_id += 1
    text.pose.position.x = float(point[0])
    text.pose.position.y = float(point[1])
    text.pose.position.z = float(point[2] + 0.10)
    text.scale.z = 0.075
    _set_marker_color(text, color[0], color[1], color[2], 1.0)
    text.text = str(label)
    marker_array.markers.append(text)
    return marker_id


def _new_marker(ros_node, marker_id, namespace, marker_type):
    marker = Marker()
    marker.header.frame_id = MAP_FRAME
    marker.header.stamp = ros_node.now()
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def _set_marker_color(marker, red, green, blue, alpha):
    marker.color.r = float(red)
    marker.color.g = float(green)
    marker.color.b = float(blue)
    marker.color.a = float(alpha)


def _point_message(vector):
    return Point(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))


def _box_edge_indices():
    return (
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    )


def _as_vector(value):
    if value is None:
        return None
    try:
        vector = np.array(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.ndim == 0 or vector.shape[0] < 3:
        return None
    return vector[:3]


def _normalize(vector):
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-6:
        return None
    return vector / norm
