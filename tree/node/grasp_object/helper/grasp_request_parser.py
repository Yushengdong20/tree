"""grasp_object 抓取服务 payload 解析和排序。"""

import math

import numpy as np

from tree.constants import BASE_LINK_FRAME
from tree.utils.geometry import (
    lookup_base_from_map_via_chassis,
    lookup_map_from_source_via_chassis,
    lookup_transform_matrix,
)
from .grasp_request_errors import NoGraspObjectError


class GraspObjectPayloadParser:
    """封装 flat/multi 抓取结果解析、TF 转换、过滤和排序。"""

    def _parse_flat_grasp_poses(self, payload):
        poses = payload.get("poses")
        if not isinstance(poses, list) or not poses:
            raise NoGraspObjectError("服务未返回抓取位姿")

        grasp_poses = []
        for index, values in enumerate(poses):
            grasp_poses.append(self._validate_grasp_pose(values, f"第 {index + 1} 个抓取位姿"))
        return grasp_poses

    def _validate_grasp_pose(self, values, prefix):
        pose = np.asarray(values, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"{prefix}必须是有限数值组成的 4x4 矩阵")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise RuntimeError(f"{prefix}不是合法的齐次矩阵")
        rotation = pose[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-3
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
            raise RuntimeError(f"{prefix}旋转矩阵无效")
        return pose

    def _convert_grasp_poses_to_frame(self, camera_grasp_poses, source_frame, target_frame):
        """把抓取服务返回的 source_frame 位姿转换到指定目标坐标系。"""
        target_from_source = self._lookup_target_from_source(target_frame, source_frame)
        target_grasp_poses = []
        for index, grasp_pose in enumerate(camera_grasp_poses):
            target_grasp_pose = target_from_source @ grasp_pose
            camera_position = grasp_pose[:3, 3]
            target_position = target_grasp_pose[:3, 3]
            # 关键步骤：同时打印抓取位姿在相机坐标系和目标坐标系下的位置，便于现场核对 TF 转换结果。
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 第 {index + 1} 个 grasp pose: "
                f"{source_frame}(x={camera_position[0]:.4f}, "
                f"y={camera_position[1]:.4f}, z={camera_position[2]:.4f}), "
                f"{target_frame}(x={target_position[0]:.4f}, "
                f"y={target_position[1]:.4f}, z={target_position[2]:.4f})"
            )
            target_grasp_poses.append(target_grasp_pose)
        return target_grasp_poses

    def _lookup_target_from_source(self, target_frame, source_frame):
        """查询 target_frame<-source_frame；map/base 关系统一通过 melon_odom 中转。"""
        if target_frame == source_frame:
            return np.eye(4)
        if target_frame == self.map_frame:
            return lookup_map_from_source_via_chassis(
                self._get_tf_listener(),
                self.ros_node,
                source_frame,
                map_frame=self.map_frame,
                base_frame=BASE_LINK_FRAME,
                chassis_frame=self.chassis_frame,
                timeout=self.tf_timeout_sec,
            )
        if target_frame == BASE_LINK_FRAME and source_frame == self.map_frame:
            return lookup_base_from_map_via_chassis(
                self._get_tf_listener(),
                self.ros_node,
                map_frame=self.map_frame,
                chassis_frame=self.chassis_frame,
                timeout=self.tf_timeout_sec,
            )
        return self._lookup_transform_matrix(target_frame, source_frame)

    def _convert_grasp_poses_to_map_frame(self, camera_grasp_poses, source_frame):
        """通过 melon_odom 分段构造 map 抓取位姿，避免直接查询 map 到 base_link 的整链 TF。"""
        map_from_source = lookup_map_from_source_via_chassis(
            self._get_tf_listener(),
            self.ros_node,
            source_frame,
            map_frame=self.map_frame,
            base_frame=BASE_LINK_FRAME,
            chassis_frame=self.chassis_frame,
            timeout=self.tf_timeout_sec,
        )
        map_grasp_poses = []
        for index, grasp_pose in enumerate(camera_grasp_poses):
            map_grasp_pose = map_from_source @ grasp_pose
            source_position = grasp_pose[:3, 3]
            map_position = map_grasp_pose[:3, 3]
            # 关键步骤：map 下位姿按 map<-melon_odom 与 base_link<-source_frame 分段组合得到。
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 第 {index + 1} 个 map grasp pose: "
                f"{source_frame}(x={source_position[0]:.4f}, "
                f"y={source_position[1]:.4f}, z={source_position[2]:.4f}), "
                f"{self.map_frame}(x={map_position[0]:.4f}, "
                f"y={map_position[1]:.4f}, z={map_position[2]:.4f}), "
                f"via={self.chassis_frame}"
            )
            map_grasp_poses.append(map_grasp_pose)
        return map_grasp_poses

    def _build_sorted_grasp_objects(self, payload, source_frame):
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            # 关键步骤：兼容旧服务或 nearest 结果，把顶层 poses 包装成单物体列表。
            camera_grasp_poses = self._parse_flat_grasp_poses(payload)
            base_grasp_poses = self._convert_grasp_poses_to_frame(
                camera_grasp_poses,
                source_frame,
                self.target_frame,
            )
            map_grasp_poses = self._convert_grasp_poses_to_map_frame(
                camera_grasp_poses,
                source_frame,
            )
            objects = [
                {
                    "object_id": 0,
                    "bbox": None,
                    "camera_grasp_poses": camera_grasp_poses,
                    "base_grasp_poses": base_grasp_poses,
                    "map_grasp_poses": map_grasp_poses,
                    "scores": list(payload.get("scores", [])),
                    "openings": list(payload.get("openings", [])),
                }
            ]
            objects = self._filter_downward_grasp_objects(objects)
            return self._sort_grasp_objects(objects)

        parsed_objects = []
        for object_index, raw_object in enumerate(objects):
            if not isinstance(raw_object, dict):
                continue
            camera_grasp_poses = []
            for pose_index, values in enumerate(raw_object.get("poses", [])):
                prefix = f"第 {object_index + 1} 个物体的第 {pose_index + 1} 个抓取位姿"
                camera_grasp_poses.append(self._validate_grasp_pose(values, prefix))
            if not camera_grasp_poses:
                continue
            base_grasp_poses = self._convert_grasp_poses_to_frame(
                camera_grasp_poses,
                source_frame,
                self.target_frame,
            )
            map_grasp_poses = self._convert_grasp_poses_to_map_frame(
                camera_grasp_poses,
                source_frame,
            )
            parsed_objects.append(
                {
                    "object_id": int(raw_object.get("object_id", object_index)),
                    "bbox": raw_object.get("bbox"),
                    "camera_grasp_poses": camera_grasp_poses,
                    "base_grasp_poses": base_grasp_poses,
                    "map_grasp_poses": map_grasp_poses,
                    "scores": self._to_float_list(raw_object.get("scores", [])),
                    "openings": self._to_float_list(raw_object.get("openings", [])),
                }
            )

        if not parsed_objects:
            raise NoGraspObjectError("服务未返回包含抓取位姿的物体")
        parsed_objects = self._filter_downward_grasp_objects(parsed_objects)
        return self._sort_grasp_objects(parsed_objects)

    def _filter_downward_grasp_objects(self, grasp_objects):
        """按 base_link 下的抓取方向过滤，只保留斜向下到垂直向下的抓取位姿。"""
        if not self.filter_downward_grasp_poses:
            return grasp_objects

        filtered_objects = []
        for grasp_object in grasp_objects:
            base_grasp_poses = grasp_object.get("base_grasp_poses", [])
            keep_indices = self._downward_grasp_keep_indices(base_grasp_poses)
            if not keep_indices:
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 物体 object_id={grasp_object.get('object_id')} "
                    f"抓取位姿全部被向下角度过滤: pose_count={len(base_grasp_poses)}, "
                    f"max_angle={self.downward_grasp_max_angle_deg:.1f}deg"
                )
                continue

            filtered_object = dict(grasp_object)
            filtered_object["camera_grasp_poses"] = self._filter_indexed_list(
                grasp_object.get("camera_grasp_poses", []),
                keep_indices,
            )
            filtered_object["base_grasp_poses"] = self._filter_indexed_list(
                base_grasp_poses,
                keep_indices,
            )
            filtered_object["map_grasp_poses"] = self._filter_indexed_list(
                grasp_object.get("map_grasp_poses", []),
                keep_indices,
            )
            filtered_object["scores"] = self._filter_optional_indexed_list(
                grasp_object.get("scores", []),
                keep_indices,
                len(base_grasp_poses),
            )
            filtered_object["openings"] = self._filter_optional_indexed_list(
                grasp_object.get("openings", []),
                keep_indices,
                len(base_grasp_poses),
            )
            filtered_objects.append(filtered_object)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 向下抓取位姿过滤: "
                f"object_id={grasp_object.get('object_id')}, "
                f"kept={len(keep_indices)}/{len(base_grasp_poses)}, "
                f"max_angle={self.downward_grasp_max_angle_deg:.1f}deg"
            )

        if not filtered_objects:
            raise NoGraspObjectError(
                f"向下抓取过滤后无候选位姿: max_angle={self.downward_grasp_max_angle_deg:.1f}deg"
            )
        return filtered_objects

    def _downward_grasp_keep_indices(self, base_grasp_poses):
        """返回局部 Z 轴与 base_link 竖直向下方向夹角不超过阈值的候选下标。"""
        min_downward_dot = math.cos(math.radians(self.downward_grasp_max_angle_deg))
        keep_indices = []
        for index, grasp_pose in enumerate(base_grasp_poses):
            z_axis = np.asarray(grasp_pose[:3, 2], dtype=float)
            z_axis_norm = np.linalg.norm(z_axis)
            if z_axis_norm < 1e-8:
                continue
            downward_alignment = float(np.dot(z_axis / z_axis_norm, np.array([0.0, 0.0, -1.0])))
            if downward_alignment >= min_downward_dot:
                keep_indices.append(index)
        return keep_indices

    @staticmethod
    def _filter_indexed_list(values, keep_indices):
        return [values[index] for index in keep_indices]

    @staticmethod
    def _filter_optional_indexed_list(values, keep_indices, expected_count):
        if not isinstance(values, list) or len(values) != expected_count:
            return []
        return [values[index] for index in keep_indices]

    def _sort_grasp_objects(self, grasp_objects):
        for grasp_object in grasp_objects:
            distance = self._compute_object_distance(grasp_object["base_grasp_poses"])
            grasp_object["distance_m"] = distance
        sorted_objects = sorted(grasp_objects, key=lambda item: item["distance_m"])
        for rank_index, grasp_object in enumerate(sorted_objects):
            grasp_object["rank_index"] = rank_index
            self.ros_node.get_logger().info(
                f"[{self.config_label}] multi 物体排序: rank={rank_index + 1}/{len(sorted_objects)}, "
                f"object_id={grasp_object['object_id']}, distance={grasp_object['distance_m']:.4f}m, "
                f"pose_count={len(grasp_object['base_grasp_poses'])}"
            )
        return sorted_objects

    def _compute_object_distance(self, base_grasp_poses):
        translations = np.asarray([pose[:3, 3] for pose in base_grasp_poses], dtype=float)
        center = np.mean(translations, axis=0)
        if self.distance_metric == "xyz":
            return float(np.linalg.norm(center))
        return float(np.hypot(center[0], center[1]))

    @staticmethod
    def _to_float_list(values):
        if not isinstance(values, list):
            return []
        return [float(value) for value in values]

    def _lookup_transform_matrix(self, target_frame, source_frame):
        tf_listener = self._get_tf_listener()
        return lookup_transform_matrix(
            tf_listener,
            self.ros_node,
            target_frame,
            source_frame,
            timeout=self.tf_timeout_sec,
        )

    def _get_tf_listener(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        tf_listener = getattr(services, "tf_listener", None)
        if tf_listener is None and hasattr(services, "arm_controller"):
            tf_listener = getattr(services.arm_controller, "tf_listener", None)
        if tf_listener is None:
            raise RuntimeError(f"services 中没有可用的 tf_listener: key={self.services_key}")
        return tf_listener

