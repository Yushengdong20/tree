"""调用 ContactGraspNet，并计算右臂抓取相关目标位姿。"""

import math
import time

import numpy as np
import py_trees
from geometry_msgs.msg import PoseStamped
from py_trees.common import Status
from kuavo_humanoid_sdk.common.three_link_torso_ik import ThreeLinkTorsoIk
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import IKAnalytical

from tree.constants import MODEL_TYPE_KEY, ROBOT_SERVICES_KEY

from ..base import TimedMockAction


SOURCE_FRAME = "camera"
TARGET_FRAME = "waist_yaw_link"
KNEE_FRAME = "knee_link"
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
        self.torso_pose_key = str(
            params.get("torso_pose_key", "grasp_object_torso_pose")
        ).strip()
        self.torso_enabled_key = str(
            params.get("torso_enabled_key", "grasp_object_use_torso")
        ).strip()
        self.torso_sample_x_min_m = float(params.get("torso_sample_x_min_m", -0.2))
        self.torso_sample_x_max_m = float(params.get("torso_sample_x_max_m", 0.2))
        self.torso_sample_z_min_m = float(params.get("torso_sample_z_min_m", 0.5))
        self.torso_sample_z_max_m = float(params.get("torso_sample_z_max_m", 0.9))
        self.torso_sample_step_m = float(params.get("torso_sample_step_m", 0.05))
        self.torso_pitch_abs_max_rad = float(params.get("torso_pitch_abs_max_rad", 0.05))
        self.knee_origin_x = float(params.get("knee_origin_x", 0.098))
        self.knee_origin_z = float(params.get("knee_origin_z", 0.376))
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
            self.torso_pose_key,
        ]:
            if not key:
                raise ValueError("blackboard pose key 不能为空")
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
        if not self.torso_enabled_key:
            raise ValueError("torso_enabled_key 不能为空")
        self.blackboard.register_key(
            key=self.torso_enabled_key,
            access=py_trees.common.Access.WRITE,
        )
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须大于等于 1")
        if self.pregrasp_offset_min_m <= self.grasp_offset_m:
            raise ValueError("pregrasp_offset_min_m 必须大于 grasp_offset_m")
        if self.pregrasp_offset_max_m < self.pregrasp_offset_min_m:
            raise ValueError("pregrasp_offset_max_m 必须大于等于 pregrasp_offset_min_m")
        if self.pregrasp_offset_samples < 1:
            raise ValueError("pregrasp_offset_samples 必须大于等于 1")
        if self.torso_sample_x_max_m < self.torso_sample_x_min_m:
            raise ValueError("torso_sample_x_max_m 必须大于等于 torso_sample_x_min_m")
        if self.torso_sample_z_max_m < self.torso_sample_z_min_m:
            raise ValueError("torso_sample_z_max_m 必须大于等于 torso_sample_z_min_m")
        if self.torso_sample_step_m <= 0.0:
            raise ValueError("torso_sample_step_m 必须大于 0")
        if self.torso_pitch_abs_max_rad < 0.0:
            raise ValueError("torso_pitch_abs_max_rad 必须大于等于 0")
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
        self._torso_controller = None

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
            knee_from_camera = self._lookup_transform_matrix(KNEE_FRAME, SOURCE_FRAME)
            current_knee_from_waist = self._lookup_transform_matrix(KNEE_FRAME, TARGET_FRAME)

            self.ros_node.set_live_runtime(
                self.config_label,
                "GRASP_COMPUTE",
                "Computing pregrasp and grasp targets with current torso",
            )
            left_target = self._arm_controller.get_current_end_effector_pose(
                "left",
                target_frame=TARGET_FRAME,
            )
            if left_target is None or len(left_target) != 6:
                raise RuntimeError("无法获取左臂当前末端位姿")

            check_sample_count = 0
            check_total_sec = 0.0
            check_start = time.monotonic()
            selected = self._find_valid_targets_for_torso_sample(
                self._current_torso_sample(current_knee_from_waist),
                grasp_poses,
                knee_from_camera,
                left_target,
            )
            check_total_sec += time.monotonic() - check_start
            check_sample_count += 1
            if selected is None:
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 当前腰部位姿下纯右臂未找到可达抓取位姿，开始腰部采样"
                )
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "TORSO_SAMPLE",
                    "Sampling torso-assisted grasp targets",
                )
                current_torso_pose = self._get_current_torso_pose()
                for sample in self._torso_ik_samples(current_knee_from_waist, current_torso_pose):
                    check_start = time.monotonic()
                    selected = self._find_valid_targets_for_torso_sample(
                        sample,
                        grasp_poses,
                        knee_from_camera,
                        left_target,
                    )
                    check_total_sec += time.monotonic() - check_start
                    check_sample_count += 1
                    if selected is not None:
                        break
            if check_sample_count > 0:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 抓取位姿sample检查耗时: "
                    f"total={check_total_sec:.3f}s, samples={check_sample_count}, "
                    f"avg={check_total_sec / check_sample_count:.3f}s/sample"
                )
            if selected is None:
                raise RuntimeError(
                    f"{len(grasp_poses)} 个抓取位姿及其z轴180度变体在当前腰部和腰部采样下均未通过检查"
                )
            self._write_selected_targets(selected)
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 预抓取或抓取位姿转换、检查或写入失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.clear_live_runtime()
        sample = selected["sample"]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算右臂抓取目标: "
            f"torso={sample['label']}, use_torso={sample['enabled']}, "
            f"grasp={selected['grasp_target']}, lift={selected['lift_target']}, "
            f"left_shift={selected['left_shift_target']}"
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
        self._torso_controller = getattr(services, "torso_controller", None)
        if self._torso_controller is None:
            raise RuntimeError("services 中没有 torso_controller")

    def _get_current_torso_pose(self):
        pose = list(getattr(self._torso_controller, "current_pose", []))
        if len(pose) != 6:
            raise RuntimeError(f"当前腰部位姿长度异常: {pose}")
        return [float(value) for value in pose]

    def _current_torso_sample(self, knee_from_waist):
        _, pitch, _ = self._tf.transformations.euler_from_matrix(knee_from_waist)
        return {
            "label": (
                "当前腰部"
                f"(x={knee_from_waist[0, 3]:.3f}, z={knee_from_waist[2, 3]:.3f}, "
                f"pitch={pitch:.3f})"
            ),
            "enabled": False,
            "knee_from_waist": knee_from_waist,
            "torso_pose": None,
            "x": float(knee_from_waist[0, 3]),
            "z": float(knee_from_waist[2, 3]),
            "pitch": float(pitch),
        }

    def _torso_ik_samples(self, current_knee_from_waist, current_torso_pose):
        current_x = float(current_knee_from_waist[0, 3])
        current_z = float(current_knee_from_waist[2, 3])
        samples = []
        for x in self._sample_axis_values(self.torso_sample_x_min_m, self.torso_sample_x_max_m):
            for z in self._sample_axis_values(self.torso_sample_z_min_m, self.torso_sample_z_max_m):
                ik_pose = ThreeLinkTorsoIk().solve_exact(
                    x,
                    z,
                    angle_step=math.radians(1.0),
                )
                if ik_pose is None:
                    continue

                pitch = float(ik_pose["torso_pitch"])
                if abs(pitch) > self.torso_pitch_abs_max_rad:
                    continue

                waist_x = float(ik_pose["end_world_x"])
                waist_z = float(ik_pose["end_world_z"])
                torso_pose = list(current_torso_pose)
                torso_pose[0] = waist_x + self.knee_origin_x
                torso_pose[2] = waist_z + self.knee_origin_z
                torso_pose[4] = pitch
                sample = {
                    "label": f"腰部采样(x={waist_x:.3f}, z={waist_z:.3f}, pitch={pitch:.3f})",
                    "enabled": True,
                    "knee_from_waist": self._make_knee_from_waist(waist_x, waist_z, pitch),
                    "torso_pose": torso_pose,
                    "x": waist_x,
                    "z": waist_z,
                    "pitch": pitch,
                }
                score = (waist_x - current_x) ** 2 + (waist_z - current_z) ** 2
                samples.append((score, sample))

        samples.sort(key=lambda item: item[0])
        return [sample for _, sample in samples]

    def _sample_axis_values(self, lower, upper):
        values = []
        value = float(lower)
        while value <= upper + 1e-9:
            values.append(round(value, 10))
            value += self.torso_sample_step_m
        if values and values[-1] < upper - 1e-9:
            values.append(float(upper))
        return values

    def _make_knee_from_waist(self, x, z, pitch):
        knee_from_waist = self._tf.transformations.euler_matrix(0.0, pitch, 0.0)
        knee_from_waist[:3, 3] = [float(x), 0.0, float(z)]
        return knee_from_waist

    def _find_valid_targets_for_torso_sample(self, sample, grasp_poses, knee_from_camera, left_target):
        waist_from_camera = np.linalg.inv(sample["knee_from_waist"]) @ knee_from_camera
        for index, grasp_pose in enumerate(grasp_poses):
            for rotate_z_180 in (False, True):
                grasp_target, grasp_target_pose = self._build_right_target(
                    grasp_pose,
                    waist_from_camera,
                    offset_m=self.grasp_offset_m,
                    rotate_z_180=rotate_z_180,
                )
                pose_label = "z轴旋转180度后" if rotate_z_180 else "原始"
                check_label = f"{sample['label']} 第 {index + 1} 个{pose_label}"
                try:
                    grasp_valid, _, pos_error, angle_error = self._check_right_pose(
                        grasp_target_pose,
                        pos_threshold=0.01,
                        angle_threshold=0.05,
                    )
                except Exception as exc:
                    continue
                if not grasp_valid:
                    continue

                for pregrasp_offset_m in self.pregrasp_offsets_m:
                    pregrasp_pose = np.array(grasp_target_pose, copy=True)
                    pregrasp_pose[:3, 3] += (
                        pregrasp_offset_m - self.grasp_offset_m
                    ) * grasp_target_pose[:3, 2]
                    pregrasp_target = self._target_from_pose(pregrasp_pose)
                    try:
                        pregrasp_valid, _, pos_error, angle_error = self._check_right_pose(
                            pregrasp_pose,
                            pos_threshold=0.03,
                            angle_threshold=0.20,
                        )
                    except Exception as exc:
                        self.ros_node.get_logger().warning(
                            f"[{self.config_label}] {check_label}预抓取位姿 "
                            f"offset={pregrasp_offset_m:.3f}m 检查失败: {exc}"
                        )
                        continue
                    if not pregrasp_valid:
                        self.ros_node.get_logger().info(
                            f"[{self.config_label}] {check_label}预抓取位姿 "
                            f"offset={pregrasp_offset_m:.3f}m IK-FK检查不通过: "
                            f"pos={pos_error:.4f}m/0.0300m, angle={angle_error:.4f}rad/0.2000rad"
                        )
                        continue

                    lift_target = list(grasp_target)
                    lift_target[2] += self.lift_distance_m
                    left_shift_target = list(grasp_target)
                    left_shift_target[1] += self.left_offset_m
                    left_shift_target[2] += self.left_shift_z_offset_m
                    return {
                        "sample": sample,
                        "index": index,
                        "pose_label": pose_label,
                        "pregrasp_offset_m": pregrasp_offset_m,
                        "left_target": list(left_target),
                        "pregrasp_pose": pregrasp_pose,
                        "pregrasp_target": pregrasp_target,
                        "grasp_pose": grasp_pose,
                        "grasp_target_pose": grasp_target_pose,
                        "grasp_target": grasp_target,
                        "lift_target": lift_target,
                        "left_shift_target": left_shift_target,
                        "waist_from_camera": waist_from_camera,
                    }
        return None

    def _check_right_pose(self, target_pose, pos_threshold, angle_threshold):
        return IKAnalytical.check_pose_validity(
            eef_pos=target_pose[:3, 3],
            eef_quat_xyzw=self._tf.transformations.quaternion_from_matrix(target_pose),
            eef_frame="zarm_r7_link",
            model_type=self.model_type,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )

    def _write_selected_targets(self, selected):
        sample = selected["sample"]
        self._publish_target_pose(selected["pregrasp_pose"])
        self._publish_grasp_pose(selected["grasp_target_pose"], selected["waist_from_camera"])
        self._publish_raw_grasp_pose(selected["grasp_pose"])

        self.blackboard.set(self.left_pose_key, selected["left_target"], overwrite=True)
        self.blackboard.set(self.pregrasp_pose_key, list(selected["pregrasp_target"]), overwrite=True)
        self.blackboard.set(self.grasp_pose_key, list(selected["grasp_target"]), overwrite=True)
        self.blackboard.set(self.lift_pose_key, selected["lift_target"], overwrite=True)
        self.blackboard.set(self.left_shift_pose_key, selected["left_shift_target"], overwrite=True)
        self.blackboard.set(self.torso_enabled_key, bool(sample["enabled"]), overwrite=True)
        if sample["enabled"]:
            self.blackboard.set(self.torso_pose_key, list(sample["torso_pose"]), overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选择{sample['label']}下第 "
            f"{selected['index'] + 1} 个{selected['pose_label']}右臂预抓取位姿 "
            f"offset={selected['pregrasp_offset_m']:.3f}m 和抓取位姿，"
            f"已写入 blackboard: left={self.left_pose_key}, "
            f"pregrasp={self.pregrasp_pose_key}, grasp={self.grasp_pose_key}, "
            f"lift={self.lift_pose_key}, left_shift={self.left_shift_pose_key}, "
            f"use_torso={self.torso_enabled_key}"
        )

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

    def _lookup_transform_matrix(self, target_frame, source_frame):
        stamp = self.ros_node.zero_time()
        self._tf_listener.waitForTransform(
            target_frame,
            source_frame,
            stamp,
            self.ros_node.duration(self.tf_timeout_sec),
        )
        translation, quaternion = self._tf_listener.lookupTransform(
            target_frame,
            source_frame,
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
            f"url={self.grasp_url}, frames={SOURCE_FRAME}->{KNEE_FRAME}->{TARGET_FRAME}, "
            f"max_attempts={self.max_attempts}"
        )
