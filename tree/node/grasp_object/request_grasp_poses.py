"""请求抓取服务并把 multi 物体抓取结果写入 blackboard。"""

import time

import numpy as np
import py_trees
from py_trees.common import Status

from tree.constants import (
    BASE_LINK_FRAME,
    CAMERA_FRAME,
    CHASSIS_FRAME,
    MAP_FRAME,
    ROBOT_SERVICES_KEY,
)
from tree.utils.geometry import lookup_transform_matrix
from tree.utils.geometry import (
    lookup_base_from_map_via_chassis,
    lookup_map_from_source_via_chassis,
)

from ..base import TimedMockAction


class NoGraspObjectError(RuntimeError):
    """抓取服务明确表示当前没有可用抓取目标。"""


class RequestGraspPoses(TimedMockAction):
    """调用抓取服务，获取按距离排序后的物体抓取结果。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_url = str(
            params.get("grasp_url", "http://127.0.0.1:13603/grasp")
        ).strip()
        self.max_attempts = int(params.get("max_attempts", 3))
        self.http_timeout_sec = float(params.get("http_timeout_sec", 30.0))
        self.retry_interval_sec = float(params.get("retry_interval_sec", 1.0))
        self.request_delay_sec = float(params.get("request_delay_sec", 0.0))
        self.target_class_id = int(params.get("target_class_id", 0))
        self.grasp_mode = str(params.get("grasp_mode", "nearest")).strip().lower()
        self.reuse_sorted_objects = self._to_bool(params.get("reuse_sorted_objects", False))
        self.distance_metric = str(params.get("distance_metric", "horizontal")).strip().lower()
        self.sorted_grasp_objects_key = str(
            params.get("sorted_grasp_objects_key", "grasp_object_sorted_grasp_objects")
        ).strip()
        self.next_grasp_object_index_key = str(
            params.get("next_grasp_object_index_key", "grasp_object_next_grasp_object_index")
        ).strip()
        self.source_frame = str(params.get("source_frame", CAMERA_FRAME)).strip()
        self.target_frame = str(params.get("target_frame", BASE_LINK_FRAME)).strip()
        self.map_frame = str(params.get("map_frame", MAP_FRAME)).strip()
        self.chassis_frame = str(params.get("chassis_frame", CHASSIS_FRAME)).strip()
        self.tf_timeout_sec = float(params.get("tf_timeout_sec", 2.0))
        self.services_key = ROBOT_SERVICES_KEY
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if self.grasp_mode not in ("nearest", "multi"):
            raise ValueError("grasp_mode 必须是 nearest 或 multi")
        if self.distance_metric not in ("horizontal", "xyz"):
            raise ValueError("distance_metric 必须是 horizontal 或 xyz")
        if not self.sorted_grasp_objects_key:
            raise ValueError("sorted_grasp_objects_key 不能为空")
        if not self.next_grasp_object_index_key:
            raise ValueError("next_grasp_object_index_key 不能为空")
        if not self.source_frame:
            raise ValueError("source_frame 不能为空")
        if not self.target_frame:
            raise ValueError("target_frame 不能为空")
        if not self.map_frame:
            raise ValueError("map_frame 不能为空")
        if not self.chassis_frame:
            raise ValueError("chassis_frame 不能为空")
        self.blackboard.register_key(
            key=self.services_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.sorted_grasp_objects_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.sorted_grasp_objects_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.next_grasp_object_index_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.next_grasp_object_index_key,
            access=py_trees.common.Access.WRITE,
        )
        self._attempt_count = 0
        self._next_attempt_at = 0.0
        self._request_delay_until = None
        self._request_delay_done = False

    def initialise(self):
        super().initialise()
        self._attempt_count = 0
        self._next_attempt_at = 0.0
        self._request_delay_until = None
        self._request_delay_done = False

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        now = time.monotonic()
        if now < self._next_attempt_at:
            return Status.RUNNING

        if self.reuse_sorted_objects and self._has_unselected_cached_object():
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 复用已缓存的 multi 抓取物体列表: "
                f"key={self.sorted_grasp_objects_key}, "
                f"next_index={self.blackboard.get(self.next_grasp_object_index_key)}"
            )
            return Status.SUCCESS

        if not self._request_delay_done and self.request_delay_sec > 0.0:
            if self._request_delay_until is None:
                # 关键步骤：缓存未命中后才等待，避免复用已有物体列表时无意义地卡 1 秒。
                self._request_delay_until = now + self.request_delay_sec
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "WAIT_GRASP_REQUEST",
                    f"准备请求抓取位姿前等待 {self.request_delay_sec:.2f}s",
                )
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 缓存未命中，"
                    f"请求抓取位姿前等待 {self.request_delay_sec:.2f}s"
                )
                return Status.RUNNING
            if now < self._request_delay_until:
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "WAIT_GRASP_REQUEST",
                    f"等待抓取位姿请求延时结束: "
                    f"{self._request_delay_until - now:.2f}s",
                )
                return Status.RUNNING
            self._request_delay_done = True

        self._attempt_count += 1
        self.ros_node.set_live_runtime(
            self.config_label,
            "GRASP_REQUEST",
            f"Calling {self.grasp_url}, mode={self.grasp_mode}, "
            f"attempt {self._attempt_count}/{self.max_attempts}",
        )
        try:
            payload = self._request_grasp_payload()
            source_frame = str(payload.get("frame_id") or self.source_frame).strip()
            grasp_objects = self._build_sorted_grasp_objects(payload, source_frame)
        except Exception as exc:
            if isinstance(exc, NoGraspObjectError):
                # 关键步骤：A 点暂时没有可抓目标时保持 RUNNING，让行为树停在请求节点继续等待。
                self._next_attempt_at = time.monotonic() + self.retry_interval_sec
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "WAIT_GRASP_OBJECT",
                    f"等待抓取目标: {exc}",
                )
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 未发现可抓取目标，继续等待: {exc}"
                )
                return Status.RUNNING

            if self._attempt_count >= self.max_attempts:
                self.feedback_message = str(exc)
                self.ros_node.clear_live_runtime()
                self.ros_node.get_logger().error(
                    f"[{self.config_label}] 抓取位姿请求连续失败 "
                    f"{self._attempt_count} 次: {exc}"
                )
                return Status.FAILURE

            self._next_attempt_at = time.monotonic() + self.retry_interval_sec
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 抓取位姿请求失败 "
                f"({self._attempt_count}/{self.max_attempts}): {exc}"
            )
            return Status.RUNNING

        # 关键步骤：这里只缓存按物体分组后的完整结果；当前物体的扁平 pose key 由 SelectNextGraspObject 写入。
        self.blackboard.set(self.sorted_grasp_objects_key, grasp_objects, overwrite=True)
        self.blackboard.set(self.next_grasp_object_index_key, 0, overwrite=True)
        self.ros_node.clear_live_runtime()
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已获取 {len(grasp_objects)} 个物体，"
            f"已按 {self.target_frame} 距离排序并写入 blackboard: "
            f"objects={self.sorted_grasp_objects_key}"
        )
        return Status.SUCCESS

    def _request_grasp_payload(self):
        import requests

        # 关键步骤：新抓取服务通过 query 参数选择类别和 nearest/multi 模式。
        response = requests.get(
            self.grasp_url,
            params={
                "target_class_id": self.target_class_id,
                "mode": self.grasp_mode,
            },
            timeout=self.http_timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("success", False):
            error = (
                payload.get("error", "服务返回 success=false")
                if isinstance(payload, dict)
                else payload
            )
            error_text = str(error)
            lowered_error = error_text.lower()
            if (
                "no object" in lowered_error
                or "empty" in lowered_error
                or "没有" in error_text
                or "无目标" in error_text
                or "无抓取" in error_text
                or "无物体" in error_text
            ):
                raise NoGraspObjectError(error_text)
            raise RuntimeError(error_text)
        return payload

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
        return self._sort_grasp_objects(parsed_objects)

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

    def _has_unselected_cached_object(self):
        if not self.blackboard.exists(self.sorted_grasp_objects_key):
            return False
        if not self.blackboard.exists(self.next_grasp_object_index_key):
            return False
        objects = self.blackboard.get(self.sorted_grasp_objects_key)
        if not isinstance(objects, list) or not objects:
            return False
        try:
            next_index = int(self.blackboard.get(self.next_grasp_object_index_key))
        except (TypeError, ValueError):
            return False
        return 0 <= next_index < len(objects)

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

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] RequestGraspPoses start: "
            f"url={self.grasp_url}, mode={self.grasp_mode}, "
            f"target_class_id={self.target_class_id}, max_attempts={self.max_attempts}, "
            f"request_delay_sec={self.request_delay_sec:.2f}, "
            f"objects_key={self.sorted_grasp_objects_key}, "
            f"frame={self.target_frame}/{self.map_frame}<-{self.source_frame}, "
            f"chassis_frame={self.chassis_frame}"
        )
