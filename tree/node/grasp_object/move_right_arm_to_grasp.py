"""调用 ContactGraspNet，并计算右臂抓取相关目标位姿。"""

import math
import time

import numpy as np
import py_trees
from geometry_msgs.msg import PoseStamped
from py_trees.common import Status
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import IKAnalytical

from tree.constants import MODEL_TYPE_KEY, ROBOT_SERVICES_KEY

from ..base import TimedMockAction


SOURCE_FRAME = "camera"
TARGET_FRAME = "waist_yaw_link"
SDK_AXIS_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class MoveRightArmToGrasp(TimedMockAction):
    """按顺序选择右臂可达的抓取位姿，并写入后续 ArmsToPose 的目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_url = str(
            params.get("grasp_url", "http://127.0.0.1:13603/grasp")
        ).strip()
        self.max_attempts = int(params.get("max_attempts", 3))
        self.http_timeout_sec = float(params.get("http_timeout_sec", 30.0))
        self.retry_interval_sec = float(params.get("retry_interval_sec", 1.0))
        self.grasp_offset_m = float(params.get("grasp_offset_m", 0.05))
        self.pregrasp_offset_min_m = float(params.get("pregrasp_offset_min_m", 0.12))
        self.pregrasp_offset_max_m = float(params.get("pregrasp_offset_max_m", 0.18))
        self.pregrasp_offset_samples = int(params.get("pregrasp_offset_samples", 5))
        self.tf_timeout_sec = float(params.get("tf_timeout_sec", 2.0))
        self.pose_topic = str(
            params.get("pose_topic", "/grasp_object/right_pregrasp_pose")
        ).strip()
        self.grasp_pose_topic = str(
            params.get("grasp_pose_topic", "/grasp_object/grasp_pose")
        ).strip()
        self.raw_grasp_pose_topic = str(
            params.get("raw_grasp_pose_topic", "/grasp_object/raw_grasp_pose")
        ).strip()
        self.services_key = ROBOT_SERVICES_KEY
        self.model_type_key = MODEL_TYPE_KEY
        self.model_type = None
        self.left_pose_key = str(
            params.get("left_pose_key", "grasp_object_left_hold_pose")
        ).strip()
        self.pregrasp_pose_key = str(
            params.get("pregrasp_pose_key", "grasp_object_pregrasp_pose")
        ).strip()
        self.grasp_pose_key = str(
            params.get("grasp_pose_key", "grasp_object_grasp_pose")
        ).strip()
        self.lift_pose_key = str(
            params.get("lift_pose_key", "grasp_object_lift_pose")
        ).strip()
        self.left_shift_pose_key = str(
            params.get("left_shift_pose_key", "grasp_object_left_shift_pose")
        ).strip()
        self.lift_distance_m = float(params.get("lift_distance_m", 0.05))
        self.left_offset_m = float(params.get("left_offset_m", 0.10))
        self.left_shift_z_offset_m = float(params.get("left_shift_z_offset_m", 0.02))
        self.blackboard.register_key(
            key=self.services_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.model_type_key,
            access=py_trees.common.Access.READ,
        )
        for key in [
            self.left_pose_key,
            self.pregrasp_pose_key,
            self.grasp_pose_key,
            self.lift_pose_key,
            self.left_shift_pose_key,
        ]:
            if not key:
                raise ValueError("blackboard pose key 不能为空")
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if self.pregrasp_offset_min_m <= self.grasp_offset_m:
            raise ValueError("pregrasp_offset_min_m 必须大于 grasp_offset_m")
        if self.pregrasp_offset_max_m < self.pregrasp_offset_min_m:
            raise ValueError("pregrasp_offset_max_m 必须大于等于 pregrasp_offset_min_m")
        if self.pregrasp_offset_samples < 1:
            raise ValueError("pregrasp_offset_samples 必须大于等于 1")
        offsets = np.linspace(
            self.pregrasp_offset_min_m,
            self.pregrasp_offset_max_m,
            self.pregrasp_offset_samples,
            endpoint=True,
        )
        middle = len(offsets) // 2
        self.pregrasp_offsets_m = [offsets[middle]]
        for step in range(1, len(offsets)):
            for offset_index in (middle + step, middle - step):
                if 0 <= offset_index < len(offsets):
                    self.pregrasp_offsets_m.append(offsets[offset_index])
        self.pose_pub = self.ros_node.create_publisher(
            self.pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self.grasp_pose_pub = self.ros_node.create_publisher(
            self.grasp_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self.raw_grasp_pose_pub = self.ros_node.create_publisher(
            self.raw_grasp_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )

        self._attempt_count = 0
        self._next_attempt_at = 0.0
        self._tf = None
        self._tf_listener = None
        self._arm_controller = None

    def initialise(self):
        super().initialise()
        self._attempt_count = 0
        self._next_attempt_at = 0.0

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()
        if self.should_skip_arm_motion():
            self.log_skip_arm_motion()
            return Status.SUCCESS

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
            grasp_poses = self._request_grasp_poses()
        except Exception as exc:
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

        try:
            self._ensure_arm_controller()
            target_from_camera = self._lookup_transform_matrix()
            selected = None

            self.ros_node.set_live_runtime(
                self.config_label,
                "GRASP_COMPUTE",
                "Computing pregrasp and grasp targets",
            )
            left_target = self._arm_controller.get_current_end_effector_pose(
                "left",
                target_frame=TARGET_FRAME,
            )
            if left_target is None or len(left_target) != 6:
                raise RuntimeError("无法获取左臂当前末端位姿")

            for index, grasp_pose in enumerate(grasp_poses):
                for rotate_z_180 in (False, True):
                    grasp_target, grasp_target_pose = self._build_right_target(
                        grasp_pose,
                        target_from_camera,
                        offset_m=self.grasp_offset_m,
                        rotate_z_180=rotate_z_180,
                    )
                    pose_label = "z轴旋转180度后" if rotate_z_180 else "原始"
                    try:
                        grasp_valid, _, pos_error, angle_error = IKAnalytical.check_pose_validity(
                            eef_pos=grasp_target_pose[:3, 3],
                            eef_quat_xyzw=self._tf.transformations.quaternion_from_matrix(grasp_target_pose),
                            eef_frame="zarm_r7_link",
                            model_type=self.model_type,
                            pos_threshold=0.01,
                            angle_threshold=0.05,
                        )
                    except Exception as exc:
                        self.ros_node.get_logger().warning(
                            f"[{self.config_label}] 第 {index + 1} 个{pose_label}抓取位姿检查失败: {exc}"
                        )
                        continue
                    if not grasp_valid:
                        self.ros_node.get_logger().warning(
                            f"[{self.config_label}] 第 {index + 1} 个{pose_label}抓取位姿IK-FK检查不通过: "
                            f"pos={pos_error:.4f}m/0.0100m, angle={angle_error:.4f}rad/0.0500rad"
                        )
                        continue

                    for pregrasp_offset_m in self.pregrasp_offsets_m:
                        pregrasp_pose = np.array(grasp_target_pose, copy=True)
                        pregrasp_pose[:3, 3] += (
                            pregrasp_offset_m - self.grasp_offset_m
                        ) * grasp_target_pose[:3, 2]
                        pregrasp_target = self._target_from_pose(pregrasp_pose)
                        try:
                            pregrasp_valid, _, pos_error, angle_error = IKAnalytical.check_pose_validity(
                                eef_pos=pregrasp_pose[:3, 3],
                                eef_quat_xyzw=self._tf.transformations.quaternion_from_matrix(pregrasp_pose),
                                eef_frame="zarm_r7_link",
                                model_type=self.model_type,
                                pos_threshold=0.03,
                                angle_threshold=0.20,
                            )
                        except Exception as exc:
                            self.ros_node.get_logger().warning(
                                f"[{self.config_label}] 第 {index + 1} 个{pose_label}预抓取位姿 "
                                f"offset={pregrasp_offset_m:.3f}m 检查失败: {exc}"
                            )
                            continue
                        if not pregrasp_valid:
                            self.ros_node.get_logger().info(
                                f"[{self.config_label}] 第 {index + 1} 个{pose_label}预抓取位姿 "
                                f"offset={pregrasp_offset_m:.3f}m IK-FK检查不通过: "
                                f"pos={pos_error:.4f}m/0.0300m, angle={angle_error:.4f}rad/0.2000rad"
                            )
                            continue

                        self._publish_target_pose(pregrasp_pose)
                        self._publish_grasp_pose(grasp_target_pose, target_from_camera)
                        self._publish_raw_grasp_pose(grasp_pose)

                        lift_target = list(grasp_target)
                        lift_target[2] += self.lift_distance_m
                        left_shift_target = list(grasp_target)
                        left_shift_target[1] += self.left_offset_m
                        left_shift_target[2] += self.left_shift_z_offset_m
                        selected = (
                            index,
                            pose_label,
                            pregrasp_offset_m,
                            list(left_target),
                            pregrasp_target,
                            grasp_target,
                            lift_target,
                            left_shift_target,
                        )
                        self.blackboard.set(
                            self.left_pose_key,
                            list(left_target),
                            overwrite=True,
                        )
                        self.blackboard.set(
                            self.pregrasp_pose_key,
                            list(pregrasp_target),
                            overwrite=True,
                        )
                        self.blackboard.set(
                            self.grasp_pose_key,
                            list(grasp_target),
                            overwrite=True,
                        )
                        self.blackboard.set(
                            self.lift_pose_key,
                            lift_target,
                            overwrite=True,
                        )
                        self.blackboard.set(
                            self.left_shift_pose_key,
                            left_shift_target,
                            overwrite=True,
                        )
                        self.ros_node.get_logger().info(
                            f"[{self.config_label}] 选择第 {index + 1}/{len(grasp_poses)} 个"
                            f"{pose_label}右臂预抓取位姿 offset={pregrasp_offset_m:.3f}m 和抓取位姿，"
                            f"已写入 blackboard: left={self.left_pose_key}, "
                            f"pregrasp={self.pregrasp_pose_key}, grasp={self.grasp_pose_key}, "
                            f"lift={self.lift_pose_key}, left_shift={self.left_shift_pose_key}"
                        )
                        break
                    if selected is not None:
                        break
                if selected is not None:
                    break
            if selected is None:
                raise RuntimeError(f"{len(grasp_poses)} 个抓取位姿及其z轴180度变体均未通过抓取/预抓取检查")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 预抓取或抓取位姿转换、检查或写入失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.clear_live_runtime()
        _, _, _, _, _, grasp_target, lift_target, left_shift_target = selected
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算右臂抓取目标: "
            f"grasp={grasp_target}, lift={lift_target}, left_shift={left_shift_target}"
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
            raise RuntimeError(str(error))

        poses = payload.get("poses")
        if not isinstance(poses, list) or not poses:
            raise RuntimeError("服务未返回抓取位姿")

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

    def _ensure_arm_controller(self):
        if self._arm_controller is not None:
            return

        import tf

        self._tf = tf
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None or not hasattr(services, "arm_controller"):
            raise RuntimeError(f"services 或 arm_controller 缺失: key={self.services_key}")
        self._arm_controller = services.arm_controller
        self.model_type = (
            self.blackboard.get(self.model_type_key)
            if self.blackboard.exists(self.model_type_key)
            else getattr(services, "model_type", None)
        )
        if not self.model_type:
            self.model_type = getattr(self._arm_controller, "model_type", None)
        if not self.model_type:
            raise RuntimeError("model_type 缺失")
        self._tf_listener = getattr(services, "tf_listener", None)
        if self._tf_listener is None:
            self._tf_listener = getattr(self._arm_controller, "tf_listener", None)
        if self._tf_listener is None:
            raise RuntimeError("services 中没有可用的 tf_listener")

    def _build_right_target(self, grasp_pose, target_from_camera, offset_m, rotate_z_180=False):
        offset = np.eye(4)
        offset[2, 3] = -offset_m
        camera_target = grasp_pose @ offset @ SDK_AXIS_TRANSFORM

        target_pose = target_from_camera @ camera_target
        if rotate_z_180:
            rotate_z_180_transform = np.eye(4)
            rotate_z_180_transform[:3, :3] = np.diag([-1.0, -1.0, 1.0])
            target_pose = target_pose @ rotate_z_180_transform

        x_axis_offset = 0.01 * target_pose[:3, 0]
        positive_translation = target_pose[:3, 3] + x_axis_offset
        negative_translation = target_pose[:3, 3] - x_axis_offset
        if np.linalg.norm(positive_translation) < np.linalg.norm(negative_translation):
            target_pose[:3, 3] = positive_translation
        else:
            target_pose[:3, 3] = negative_translation

        return self._target_from_pose(target_pose), target_pose

    def _target_from_pose(self, target_pose):
        roll, pitch, yaw = self._tf.transformations.euler_from_matrix(target_pose)
        return [
            float(target_pose[0, 3]),
            float(target_pose[1, 3]),
            float(target_pose[2, 3]),
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        ]

    def _lookup_transform_matrix(self):
        stamp = self.ros_node.zero_time()
        self._tf_listener.waitForTransform(
            TARGET_FRAME,
            SOURCE_FRAME,
            stamp,
            self.ros_node.duration(self.tf_timeout_sec),
        )
        translation, quaternion = self._tf_listener.lookupTransform(
            TARGET_FRAME,
            SOURCE_FRAME,
            stamp,
        )
        return self._tf.transformations.concatenate_matrices(
            self._tf.transformations.translation_matrix(translation),
            self._tf.transformations.quaternion_matrix(quaternion),
        )

    def _publish_target_pose(self, target_pose):
        quaternion = self._tf.transformations.quaternion_from_matrix(target_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = TARGET_FRAME
        message.pose.position.x = float(target_pose[0, 3])
        message.pose.position.y = float(target_pose[1, 3])
        message.pose.position.z = float(target_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.pose_pub.publish(message)

    def _publish_grasp_pose(self, target_pose, target_from_camera):
        source_pose = np.linalg.inv(target_from_camera) @ target_pose
        quaternion = self._tf.transformations.quaternion_from_matrix(source_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = 'camera_link'
        message.pose.position.x = float(source_pose[0, 3])
        message.pose.position.y = float(source_pose[1, 3])
        message.pose.position.z = float(source_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.grasp_pose_pub.publish(message)

    def _publish_raw_grasp_pose(self, grasp_pose):
        quaternion = self._tf.transformations.quaternion_from_matrix(grasp_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = 'camera_link'
        message.pose.position.x = float(grasp_pose[0, 3])
        message.pose.position.y = float(grasp_pose[1, 3])
        message.pose.position.z = float(grasp_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.raw_grasp_pose_pub.publish(message)

    def terminate(self, new_status):
        self.ros_node.clear_live_runtime()
        super().terminate(new_status)

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveRightArmToGrasp start: "
            f"url={self.grasp_url}, frames={SOURCE_FRAME}->{TARGET_FRAME}, "
            f"max_attempts={self.max_attempts}"
        )
