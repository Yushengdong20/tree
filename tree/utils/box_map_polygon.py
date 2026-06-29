"""箱子 map 区域 polygon 判断工具。"""

import json
import math


def parse_map_polygon(raw_polygon):
    """解析配置中的 map polygon 点列表。"""
    if raw_polygon is None or raw_polygon == "":
        return []

    if isinstance(raw_polygon, str):
        raw_polygon = json.loads(raw_polygon)

    if not isinstance(raw_polygon, list):
        raise ValueError("valid_box_map_polygon 必须是点列表")

    polygon = []
    for point in raw_polygon:
        if isinstance(point, dict):
            if "x" not in point or "y" not in point:
                raise ValueError("valid_box_map_polygon 点字典必须包含 x/y")
            polygon.append({"x": float(point["x"]), "y": float(point["y"])})
            continue

        if isinstance(point, (list, tuple)) and len(point) >= 2:
            polygon.append({"x": float(point[0]), "y": float(point[1])})
            continue

        raise ValueError("valid_box_map_polygon 点必须是 {'x': ..., 'y': ...} 或 [x, y]")

    if polygon and len(polygon) < 3:
        raise ValueError("valid_box_map_polygon 至少需要 3 个点")
    return polygon


def is_map_position_in_polygon(map_position, polygon):
    """判断 map_position 是否在 polygon 内，未配置 polygon 时默认放行。"""
    if not polygon:
        return True
    if map_position is None or "x" not in map_position or "y" not in map_position:
        return False

    point_x = float(map_position["x"])
    point_y = float(map_position["y"])

    # 关键步骤：边界上的箱子也认为在区域内，避免边界检测抖动导致目标丢失。
    if _is_point_on_polygon_boundary(point_x, point_y, polygon):
        return True

    inside = False
    point_count = len(polygon)
    previous_index = point_count - 1
    for current_index in range(point_count):
        current = polygon[current_index]
        previous = polygon[previous_index]
        current_y = float(current["y"])
        previous_y = float(previous["y"])

        intersects_y_range = (current_y > point_y) != (previous_y > point_y)
        if intersects_y_range:
            current_x = float(current["x"])
            previous_x = float(previous["x"])
            cross_x = (
                (previous_x - current_x)
                * (point_y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            if point_x < cross_x:
                inside = not inside
        previous_index = current_index
    return inside


def _is_point_on_polygon_boundary(point_x, point_y, polygon):
    """判断点是否落在 polygon 边界上。"""
    point_count = len(polygon)
    previous_index = point_count - 1
    for current_index in range(point_count):
        current = polygon[current_index]
        previous = polygon[previous_index]
        if _is_point_on_segment(
            point_x,
            point_y,
            float(previous["x"]),
            float(previous["y"]),
            float(current["x"]),
            float(current["y"]),
        ):
            return True
        previous_index = current_index
    return False


def _is_point_on_segment(point_x, point_y, start_x, start_y, end_x, end_y):
    """判断点是否在线段上。"""
    cross = (point_x - start_x) * (end_y - start_y) - (point_y - start_y) * (end_x - start_x)
    if abs(cross) > 1e-8:
        return False

    min_x = min(start_x, end_x) - 1e-8
    max_x = max(start_x, end_x) + 1e-8
    min_y = min(start_y, end_y) - 1e-8
    max_y = max(start_y, end_y) + 1e-8
    if point_x < min_x or point_x > max_x or point_y < min_y or point_y > max_y:
        return False

    return math.hypot(point_x - start_x, point_y - start_y) <= math.hypot(
        end_x - start_x,
        end_y - start_y,
    ) + 1e-8
