"""请求抓取服务并把候选抓取位姿转到 base_link 后写入 blackboard。"""

import time

import numpy as np
import py_trees
from py_trees.common import Status

from tree.constants import BASE_LINK_FRAME, CAMERA_FRAME, ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class NoGraspObjectError(RuntimeError):
    """抓取服务明确表示当前没有可用抓取目标。"""


class RequestGraspPoses(TimedMockAction):
    """调用抓取服务，获取候选抓取位姿并缓存为 base_link 坐标系。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_url = str(
            params.get("grasp_url", "http://127.0.0.1:13603/grasp")
        ).strip()
        self.max_attempts = int(params.get("max_attempts", 3))
        self.http_timeout_sec = float(params.get("http_timeout_sec", 30.0))
        self.retry_interval_sec = float(params.get("retry_interval_sec", 1.0))
        self.base_grasp_poses_key = str(
            params.get(
                "base_grasp_poses_key",
                params.get("raw_grasp_poses_key", "grasp_object_base_grasp_poses"),
            )
        ).strip()
        self.camera_grasp_poses_key = str(
            params.get("camera_grasp_poses_key", "grasp_object_camera_grasp_poses")
        ).strip()
        self.source_frame = str(params.get("source_frame", CAMERA_FRAME)).strip()
        self.target_frame = str(params.get("target_frame", BASE_LINK_FRAME)).strip()
        self.tf_timeout_sec = float(params.get("tf_timeout_sec", 2.0))
        self.services_key = ROBOT_SERVICES_KEY
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if not self.base_grasp_poses_key:
            raise ValueError("base_grasp_poses_key 不能为空")
        if not self.camera_grasp_poses_key:
            raise ValueError("camera_grasp_poses_key 不能为空")
        if not self.source_frame:
            raise ValueError("source_frame 不能为空")
        if not self.target_frame:
            raise ValueError("target_frame 不能为空")
        self.blackboard.register_key(
            key=self.services_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.base_grasp_poses_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.camera_grasp_poses_key,
            access=py_trees.common.Access.WRITE,
        )
        self._attempt_count = 0
        self._next_attempt_at = 0.0

    def initialise(self):
        super().initialise()
        self._attempt_count = 0
        self._next_attempt_at = 0.0

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        now = time.monotonic()
        if now < self._next_attempt_at:
            return Status.RUNNING

        self._attempt_count += 1
        self.ros_node.set_live_runtime(
            self.config_label,
            "GRASP_REQUEST",
            f"Calling {self.grasp_url}, attempt {self._attempt_count}/{self.max_attempts}",
        )
        try:
            camera_grasp_poses = self._request_grasp_poses()
            grasp_poses = self._convert_grasp_poses_to_target_frame(camera_grasp_poses)
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

        self.blackboard.set(self.base_grasp_poses_key, grasp_poses, overwrite=True)
        # 关键步骤：保留抓取服务返回的 camera 原始矩阵，后续选中候选时可以直接打印同一 index 的原始坐标。
        self.blackboard.set(self.camera_grasp_poses_key, camera_grasp_poses, overwrite=True)
        self.ros_node.clear_live_runtime()
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已获取 {len(grasp_poses)} 个候选抓取位姿，"
            f"已转换到 {self.target_frame} 并写入 blackboard: "
            f"{self.base_grasp_poses_key}, camera_raw={self.camera_grasp_poses_key}"
        )
        return Status.SUCCESS

    def _request_grasp_poses(self):
        import requests

        response = requests.get(self.grasp_url, timeout=self.http_timeout_sec)
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

        poses = payload.get("poses")
        if not isinstance(poses, list) or not poses:
            raise NoGraspObjectError("服务未返回抓取位姿")

        grasp_poses = []
        for index, values in enumerate(poses):
            pose = np.asarray(values, dtype=float)
            prefix = f"第 {index + 1} 个抓取位姿"
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise RuntimeError(f"{prefix}必须是有限数值组成的 4x4 矩阵")
            if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
                raise RuntimeError(f"{prefix}不是合法的齐次矩阵")
            rotation = pose[:3, :3]
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-3
            ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
                raise RuntimeError(f"{prefix}旋转矩阵无效")
            grasp_poses.append(pose)
        return grasp_poses

    def _convert_grasp_poses_to_target_frame(self, camera_grasp_poses):
        """把抓取服务返回的 source_frame 位姿固定到 base_link，避免腰部运动后复用旧 camera 坐标。"""
        target_from_source = self._lookup_transform_matrix(self.target_frame, self.source_frame)
        target_grasp_poses = []
        for index, grasp_pose in enumerate(camera_grasp_poses):
            target_grasp_pose = target_from_source @ grasp_pose
            camera_position = grasp_pose[:3, 3]
            target_position = target_grasp_pose[:3, 3]
            # 关键步骤：同时打印抓取位姿在相机坐标系和目标坐标系下的位置，便于现场核对 TF 转换结果。
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 第 {index + 1} 个 grasp pose: "
                f"{self.source_frame}(x={camera_position[0]:.4f}, "
                f"y={camera_position[1]:.4f}, z={camera_position[2]:.4f}), "
                f"{self.target_frame}(x={target_position[0]:.4f}, "
                f"y={target_position[1]:.4f}, z={target_position[2]:.4f})"
            )
            target_grasp_poses.append(target_grasp_pose)
        return target_grasp_poses

    def _lookup_transform_matrix(self, target_frame, source_frame):
        import tf

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        tf_listener = getattr(services, "tf_listener", None)
        if tf_listener is None and hasattr(services, "arm_controller"):
            tf_listener = getattr(services.arm_controller, "tf_listener", None)
        if tf_listener is None:
            raise RuntimeError(f"services 中没有可用的 tf_listener: key={self.services_key}")

        stamp = self.ros_node.zero_time()
        tf_listener.waitForTransform(
            target_frame,
            source_frame,
            stamp,
            self.ros_node.duration(self.tf_timeout_sec),
        )
        translation, quaternion = tf_listener.lookupTransform(
            target_frame,
            source_frame,
            stamp,
        )
        return tf.transformations.concatenate_matrices(
            tf.transformations.translation_matrix(translation),
            tf.transformations.quaternion_matrix(quaternion),
        )

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] RequestGraspPoses start: "
            f"url={self.grasp_url}, max_attempts={self.max_attempts}, "
            f"base_grasp_poses_key={self.base_grasp_poses_key}, "
            f"camera_grasp_poses_key={self.camera_grasp_poses_key}, "
            f"frame={self.target_frame}<-{self.source_frame}"
        )
