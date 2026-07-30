"""根据 FoundationPose 托盘 OBB 规划“托盘内底面中心直放”目标。

该节点用于上开口浅托盘，不做四格码垛或左右推送：

1. 读取 ``ComputeFoundationPosePalletMapPolygon`` 写入黑板的托盘 map 位姿；
2. 按托盘 OBB 的真实尺寸和朝向，取托盘内底面中心作为本轮目标箱心的 XY；
3. 由托盘底面、内底高度补偿、待放箱高度计算放置平面与目标箱心 Z；
4. 找到托盘两条长边，在两条长边外侧各生成一个“正对长边”的底盘站位；
5. 丢弃仍落在托盘 polygon 内的站位，优先选择距离当前底盘更近的一侧；
6. 写出与 ``ComputeMoveBoxPalletPlaceStrategy`` 相同的关键 blackboard 输出，
   让后续 ``ComputeMoveBoxPlaceTargets`` 与直接放箱子树无需改动。

约定：FP 托盘 OBB 的 ``size`` 为实体外包围盒尺寸；托盘内底面默认从实体下表面
向上 ``pallet_inner_bottom_offset_m``，再额外使用 ``place_clearance_z`` 作为箱底
与内底之间的竖直安全间隙。
"""

import math

import numpy as np
import py_trees
from geometry_msgs.msg import Point
from py_trees.common import Status
from visualization_msgs.msg import Marker, MarkerArray

from tree.constants import MAP_FRAME, ROBOT_SERVICES_KEY
from tree.utils.box_map_polygon import is_map_position_in_polygon

from ..base import TimedMockAction


class ComputeMoveBoxPalletCenterPlaceTarget(TimedMockAction):
    """从真实 FP 托盘 OBB 规划托盘内底面中心直放。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.pallet_pose_key = str(
            params.get("pallet_pose_key", "move_box_detected_pallet_map_pose")
        ).strip()
        self.pallet_map_polygon_key = str(
            params.get("pallet_map_polygon_key", "move_box_detected_pallet_map_polygon")
        ).strip()

        # 抓取箱体仍使用流程中配置的人工尺寸；这里不会读取 FP 箱体 size。
        self.box_size_x = float(params.get("box_size_x", 0.60))
        self.box_size_y = float(params.get("box_size_y", 0.40))
        self.box_size_z = float(params.get("box_size_z", 0.34))
        self.pallet_inner_bottom_offset_m = float(
            params.get("pallet_inner_bottom_offset_m", 0.015)
        )
        self.place_clearance_z = float(params.get("place_clearance_z", 0.0))
        self.navigation_pallet_clearance_m = max(
            0.0, float(params.get("navigation_pallet_clearance_m", 0.10))
        )
        self.navigation_standoff_m = max(
            0.0, float(params.get("navigation_standoff_m", 0.90))
        )
        self.navigation_select_mode = str(
            params.get("navigation_select_mode", "nearest_feasible")
        ).strip().lower()

        self.navigation_target_key = str(
            params.get("navigation_target_key", "move_box_pallet_stack_navigation_target")
        ).strip()
        self.place_plane_height_key = str(
            params.get("place_plane_height_key", "move_box_pallet_stack_place_plane_height")
        ).strip()
        self.slot_pose_key = str(
            params.get("slot_pose_key", "move_box_pallet_stack_slot_pose")
        ).strip()
        self.expected_box_pose_key = str(
            params.get("expected_box_pose_key", "move_box_pallet_stack_expected_box_pose")
        ).strip()
        self.strategy_key = str(
            params.get("strategy_key", "move_box_pallet_place_strategy")
        ).strip()
        self.final_box_pose_key = str(
            params.get("final_box_pose_key", "move_box_pallet_place_final_box_pose")
        ).strip()
        self.pre_box_pose_key = str(
            params.get("pre_box_pose_key", "move_box_pallet_place_pre_box_pose")
        ).strip()

        self.visualization_enabled = self._to_bool(params.get("visualization_enabled", True))
        self.visualization_topic = str(
            params.get("visualization_topic", "/move_box/pallet_place_dynamic_estimate_markers")
        ).strip()

        for key in (self.services_key, self.pallet_pose_key, self.pallet_map_polygon_key):
            if key:
                self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)
        for key in (
            self.navigation_target_key,
            self.place_plane_height_key,
            self.slot_pose_key,
            self.expected_box_pose_key,
            self.strategy_key,
            self.final_box_pose_key,
            self.pre_box_pose_key,
        ):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame="base_link",
        )
        self.visualization_pub = None
        if self.visualization_enabled and self.visualization_topic:
            self.visualization_pub = self.ros_node.create_publisher(
                self.visualization_topic, MarkerArray, queue_size=1, latch=True
            )

    def update(self):
        pallet = self._read_pallet_pose()
        polygon = self._read_polygon()
        if pallet is None or polygon is None:
            return Status.RUNNING

        geometry = self._build_geometry(pallet)
        if geometry is None:
            return Status.FAILURE
        navigation_pose, candidates = self._select_navigation_pose(geometry, polygon)
        if navigation_pose is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 无可用托盘外侧导航站位，停止放箱规划"
            )
            return Status.FAILURE

        # 托盘的内底平面：OBB 中心沿世界竖直投影的下半高，再加内底厚度补偿。
        # 当前机器人在平整地面作业，后续 ComputeMoveBoxPlaceTargets 使用 map-z 垂直下降。
        inner_bottom_z = (
            geometry["center_z"] - geometry["height"] * 0.5 + self.pallet_inner_bottom_offset_m
        )
        place_plane_z = inner_bottom_z + self.place_clearance_z
        yaw = geometry["long_yaw"]
        slot_pose = {
            "x": geometry["center_x"],
            "y": geometry["center_y"],
            "z": place_plane_z,
            "yaw": yaw,
            "target_type": "pallet_inner_bottom_center",
        }
        expected_box_pose = dict(slot_pose)
        expected_box_pose["z"] = place_plane_z + self.box_size_z * 0.5
        expected_box_pose["box_size"] = [self.box_size_x, self.box_size_y, self.box_size_z]
        final_box_pose = dict(expected_box_pose)
        pre_box_pose = dict(expected_box_pose)

        self.blackboard.set(self.navigation_target_key, navigation_pose, overwrite=True)
        self.blackboard.set(self.place_plane_height_key, place_plane_z, overwrite=True)
        self.blackboard.set(self.slot_pose_key, slot_pose, overwrite=True)
        self.blackboard.set(self.expected_box_pose_key, expected_box_pose, overwrite=True)
        self.blackboard.set(self.strategy_key, "direct_place", overwrite=True)
        self.blackboard.set(self.final_box_pose_key, final_box_pose, overwrite=True)
        self.blackboard.set(self.pre_box_pose_key, pre_box_pose, overwrite=True)

        self._publish(geometry, polygon, expected_box_pose, navigation_pose, candidates)
        self.ros_node.get_logger().info(
            "\033[96m"
            f"[{self.config_label}] FP托盘中心直放规划: "
            f"托盘中心=({geometry['center_x']:.3f},{geometry['center_y']:.3f},{geometry['center_z']:.3f}), "
            f"尺寸=({geometry['long_size']:.3f},{geometry['short_size']:.3f},{geometry['height']:.3f}), "
            f"长边轴=({geometry['long_axis'][0]:.3f},{geometry['long_axis'][1]:.3f}), "
            f"内底面z={inner_bottom_z:.3f}, 放置平面z={place_plane_z:.3f}, "
            f"目标箱心=({expected_box_pose['x']:.3f},{expected_box_pose['y']:.3f},{expected_box_pose['z']:.3f}, "
            f"yaw={yaw:.1f}deg), "
            f"选中长边外侧站位=({navigation_pose['x']:.3f},{navigation_pose['y']:.3f}, "
            f"yaw={navigation_pose['yaw']:.1f}deg), side={navigation_pose['approach_side']}"
            "\033[0m"
        )
        return Status.SUCCESS

    def _read_pallet_pose(self):
        if not self.pallet_pose_key or not self.blackboard.exists(self.pallet_pose_key):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 FP 托盘 map 位姿: key={self.pallet_pose_key}"
            )
            return None
        raw = self.blackboard.get(self.pallet_pose_key)
        try:
            pose = dict(raw)
            pose["x"] = float(pose["x"])
            pose["y"] = float(pose["y"])
            pose["z"] = float(pose["z"])
            pose["size"] = [abs(float(value)) for value in pose["size"]]
            if len(pose["size"]) != 3:
                raise ValueError("size 长度不是 3")
            pose["side_axis"] = [float(value) for value in pose["side_axis"][:2]]
            pose["front_axis"] = [float(value) for value in pose["front_axis"][:2]]
            return pose
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] FP 托盘位姿格式无效: key={self.pallet_pose_key}, "
                f"value={raw!r}, error={exc}"
            )
            return None

    def _read_polygon(self):
        if not self.pallet_map_polygon_key or not self.blackboard.exists(self.pallet_map_polygon_key):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 等待 FP 托盘 map 四边形: key={self.pallet_map_polygon_key}"
            )
            return None
        try:
            polygon = self.blackboard.get(self.pallet_map_polygon_key)
            result = [(float(point["x"]), float(point["y"])) for point in polygon]
            if len(result) < 3:
                raise ValueError("顶点少于 3")
            return result
        except (TypeError, KeyError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] FP 托盘四边形格式无效: error={exc}"
            )
            return None

    def _build_geometry(self, pallet):
        side_axis = self._normalize(pallet["side_axis"], "side_axis")
        front_axis = np.asarray(pallet["front_axis"], dtype=float)
        front_axis -= side_axis * float(np.dot(front_axis, side_axis))
        front_axis = self._normalize(front_axis, "front_axis")
        side_size, front_size, height = pallet["size"]
        if min(side_size, front_size, height) <= 0.0:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] FP 托盘尺寸非法: size={pallet['size']}"
            )
            return None
        if side_size >= front_size:
            long_axis, short_axis = side_axis, front_axis
            long_size, short_size = side_size, front_size
        else:
            long_axis, short_axis = front_axis, side_axis
            long_size, short_size = front_size, side_size
        return {
            "center_x": pallet["x"], "center_y": pallet["y"], "center_z": pallet["z"],
            "long_axis": long_axis, "short_axis": short_axis,
            "long_size": long_size, "short_size": short_size, "height": height,
            "long_yaw": math.degrees(math.atan2(float(long_axis[1]), float(long_axis[0]))),
        }

    def _select_navigation_pose(self, geometry, polygon):
        center = np.asarray([geometry["center_x"], geometry["center_y"]], dtype=float)
        # 面对长边意味着机器人应站在两条长边外侧，即沿短边法向离开托盘。
        distance = geometry["short_size"] * 0.5 + self.navigation_pallet_clearance_m + self.navigation_standoff_m
        candidates = []
        for sign, label in ((1.0, "short_axis_positive"), (-1.0, "short_axis_negative")):
            point = center + sign * geometry["short_axis"] * distance
            inside = is_map_position_in_polygon(
                {"x": float(point[0]), "y": float(point[1])},
                [{"x": x, "y": y} for x, y in polygon],
            )
            yaw = math.degrees(math.atan2(geometry["center_y"] - point[1], geometry["center_x"] - point[0]))
            candidates.append({
                "x": float(point[0]), "y": float(point[1]), "yaw": yaw,
                "approach_side": label, "inside_pallet": bool(inside),
            })
        feasible = [item for item in candidates if not item["inside_pallet"]]
        if not feasible:
            return None, candidates
        pose = self.odom_transformer.get_current_pose()
        if pose is not None and self.navigation_select_mode == "nearest_feasible":
            selected = min(feasible, key=lambda item: math.hypot(item["x"] - pose[0], item["y"] - pose[1]))
        else:
            selected = feasible[0]
        selected = dict(selected)
        selected["approach_source"] = "fp_pallet_long_edge_two_sided"
        selected["approach_axis_x"] = float(geometry["short_axis"][0])
        selected["approach_axis_y"] = float(geometry["short_axis"][1])
        return selected, candidates

    def _publish(self, geometry, polygon, expected_box_pose, navigation_pose, candidates):
        if self.visualization_pub is None:
            return
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        stamp = self.ros_node.now()

        pallet = self._marker(1, "fp_pallet_center_plan", Marker.LINE_STRIP, stamp)
        pallet.scale.x = 0.025
        for x, y in polygon + [polygon[0]]:
            pallet.points.append(Point(x=x, y=y, z=geometry["center_z"]))
        self._color(pallet, 1.0, 0.45, 0.05, 1.0)
        markers.markers.append(pallet)

        target = self._marker(2, "fp_pallet_center_target", Marker.CUBE, stamp)
        target.pose.position.x = expected_box_pose["x"]
        target.pose.position.y = expected_box_pose["y"]
        target.pose.position.z = expected_box_pose["z"]
        yaw = math.radians(expected_box_pose["yaw"])
        target.pose.orientation.z = math.sin(yaw * 0.5)
        target.pose.orientation.w = math.cos(yaw * 0.5)
        target.scale.x, target.scale.y, target.scale.z = self.box_size_x, self.box_size_y, self.box_size_z
        self._color(target, 0.1, 0.9, 0.25, 0.32)
        markers.markers.append(target)

        for index, candidate in enumerate(candidates):
            marker = self._marker(10 + index, "fp_pallet_long_edge_candidates", Marker.ARROW, stamp)
            marker.pose.position.x = candidate["x"]
            marker.pose.position.y = candidate["y"]
            angle = math.radians(candidate["yaw"])
            marker.pose.orientation.z = math.sin(angle * 0.5)
            marker.pose.orientation.w = math.cos(angle * 0.5)
            marker.scale.x, marker.scale.y, marker.scale.z = 0.42, 0.09, 0.09
            selected = candidate["approach_side"] == navigation_pose["approach_side"]
            self._color(marker, 0.1, 0.9, 1.0, 1.0) if selected else self._color(marker, 0.45, 0.45, 0.45, 0.7)
            markers.markers.append(marker)

        text = self._marker(20, "fp_pallet_center_text", Marker.TEXT_VIEW_FACING, stamp)
        text.pose.position.x, text.pose.position.y = geometry["center_x"], geometry["center_y"]
        text.pose.position.z = expected_box_pose["z"] + self.box_size_z * 0.7
        text.scale.z = 0.10
        text.text = ("FP PALLET CENTER PLACE\\n"
                     f"target=({expected_box_pose['x']:.2f},{expected_box_pose['y']:.2f},{expected_box_pose['z']:.2f})\\n"
                     f"nav=({navigation_pose['x']:.2f},{navigation_pose['y']:.2f}) yaw={navigation_pose['yaw']:.1f}\\n"
                     f"face_long_edge side={navigation_pose['approach_side']}")
        self._color(text, 1.0, 1.0, 1.0, 1.0)
        markers.markers.append(text)
        self.visualization_pub.publish(markers)

    @staticmethod
    def _normalize(vector, name):
        value = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"{name} 无有效 map 平面方向: {vector}")
        return value / norm

    @staticmethod
    def _marker(marker_id, namespace, marker_type, stamp):
        marker = Marker()
        marker.header.frame_id = MAP_FRAME
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _color(marker, red, green, blue, alpha):
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = red, green, blue, alpha
