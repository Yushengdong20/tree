"""右臂抓取目标计算工具，供多个行为树节点复用。"""

import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import py_trees
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from kuavo_humanoid_sdk.common.three_link_torso_ik import ThreeLinkTorsoIk
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import IKAnalytical

from tree.constants import BASE_LINK_FRAME, MODEL_TYPE_KEY, ROBOT_SERVICES_KEY

SOURCE_FRAME = BASE_LINK_FRAME
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


def _target_from_pose_for_worker(target_pose):
    """子进程中从矩阵生成 yaw/pitch/roll 目标。"""
    roll, pitch, yaw = tf_trans.euler_from_matrix(target_pose)
    return [
        float(target_pose[0, 3]),
        float(target_pose[1, 3]),
        float(target_pose[2, 3]),
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    ]


def _build_right_target_for_worker(grasp_pose, target_from_source, offset_m, rotate_z_180=False):
    """子进程中构造右臂候选目标矩阵。"""
    offset = np.eye(4)
    offset[2, 3] = -offset_m
    source_target = grasp_pose @ offset @ SDK_AXIS_TRANSFORM

    target_pose = target_from_source @ source_target
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

    return _target_from_pose_for_worker(target_pose), target_pose


def _find_valid_grasp_target_for_torso_sample_worker(args):
    """子进程检查单个腰部 sample，避免把 ROS/blackboard 对象传入进程池。"""
    (
        sample,
        grasp_poses,
        knee_from_base,
        left_target,
        grasp_offset_m,
        left_offset_m,
        left_shift_z_offset_m,
        model_type,
    ) = args
    waist_from_base = np.linalg.inv(sample["knee_from_waist"]) @ knee_from_base
    print(f'{waist_from_base=} {knee_from_base=} {np.linalg.inv(sample["knee_from_waist"])=}')
    candidates = []
    for index, grasp_pose in enumerate(grasp_poses):
        for rotate_z_180 in (False, True):
            grasp_target, grasp_target_pose = _build_right_target_for_worker(
                grasp_pose,
                waist_from_base,
                offset_m=grasp_offset_m,
                rotate_z_180=rotate_z_180,
            )
            pose_label = "z轴旋转180度后" if rotate_z_180 else "原始"
            candidates.append(
                {
                    "index": index,
                    "pose_label": pose_label,
                    "rotate_z_180": rotate_z_180,
                    "grasp_pose": grasp_pose,
                    "grasp_target": grasp_target,
                    "grasp_target_pose": grasp_target_pose,
                    "position": grasp_target_pose[:3, 3],
                    "quaternion": tf_trans.quaternion_from_matrix(grasp_target_pose),
                }
            )

    if not candidates:
        return None

    try:
        # 关键步骤：同一个腰部 sample 下的抓取候选一次性做 IK-FK 检查，减少 Python 循环和重复函数调用。
        valid_mask, _, _, _ = IKAnalytical.check_pose_validity_batch(
            eef_pos_array=np.stack([candidate["position"] for candidate in candidates], axis=0),
            eef_quat_xyzw_array=np.stack([candidate["quaternion"] for candidate in candidates], axis=0),
            eef_frame="zarm_r7_link",
            model_type=model_type,
            pos_threshold=0.01,
            angle_threshold=0.05,
        )
    except Exception as exc:
        # 关键步骤：子进程不能直接使用 ROS logger，把异常带回主进程统一记录。
        return {
            "__worker_error__": True,
            "sample_label": str(sample.get("label", "")),
            "candidate_count": len(candidates),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    for candidate_index, grasp_valid in enumerate(valid_mask):
        if not grasp_valid:
            continue
        candidate = candidates[candidate_index]
        grasp_target = candidate["grasp_target"]
        grasp_target_pose = candidate["grasp_target_pose"]
        left_shift_target = list(grasp_target)
        left_shift_target[1] += left_offset_m
        left_shift_target[2] += left_shift_z_offset_m
        return {
            "sample": sample,
            "index": candidate["index"],
            "pose_label": candidate["pose_label"],
            "rotate_z_180": bool(candidate["rotate_z_180"]),
            "left_target": list(left_target),
            "grasp_pose": candidate["grasp_pose"],
            "grasp_target_pose": grasp_target_pose,
            "grasp_target": grasp_target,
            "left_shift_target": left_shift_target,
            "waist_from_base": waist_from_base,
        }
    return None


class RightGraspTargetComputer:
    """封装抓取目标选择、腰部采样和调试位姿发布。"""

    def __init__(self, config_label, ros_node, blackboard, params):
        self.config_label = config_label
        self.ros_node = ros_node
        self.blackboard = blackboard
        self.base_grasp_poses_key = str(
            params.get(
                "base_grasp_poses_key",
                params.get("raw_grasp_poses_key", "grasp_object_base_grasp_poses"),
            )
        ).strip()
        self.camera_grasp_poses_key = str(
            params.get("camera_grasp_poses_key", "grasp_object_camera_grasp_poses")
        ).strip()
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
        self.grasp_pose_key = str(
            params.get("grasp_pose_key", "grasp_object_grasp_pose")
        ).strip()
        self.left_shift_pose_key = str(
            params.get("left_shift_pose_key", "grasp_object_left_shift_pose")
        ).strip()
        self.left_offset_m = float(params.get("left_offset_m", 0.0))
        self.left_shift_z_offset_m = float(params.get("left_shift_z_offset_m", 0.03))
        self.torso_pose_key = str(
            params.get("torso_pose_key", "grasp_object_torso_pose")
        ).strip()
        self.torso_enabled_key = str(
            params.get("torso_enabled_key", "grasp_object_use_torso")
        ).strip()
        self.selected_grasp_pose_key = str(
            params.get("selected_grasp_pose_key", "grasp_object_selected_grasp_pose")
        ).strip()
        self.selected_base_grasp_pose_key = str(
            params.get("selected_base_grasp_pose_key", "grasp_object_selected_base_grasp_pose")
        ).strip()
        self.selected_rotate_z_180_key = str(
            params.get("selected_rotate_z_180_key", "grasp_object_selected_rotate_z_180")
        ).strip()
        self.selected_waist_from_base_key = str(
            params.get(
                "selected_waist_from_base_key",
                params.get("selected_waist_from_camera_key", "grasp_object_selected_waist_from_base"),
            )
        ).strip()
        self.torso_sample_x_min_m = float(params.get("torso_sample_x_min_m", -0.2))
        self.torso_sample_x_max_m = float(params.get("torso_sample_x_max_m", 0.2))
        self.torso_sample_z_min_m = float(params.get("torso_sample_z_min_m", 0.5))
        self.torso_sample_z_max_m = float(params.get("torso_sample_z_max_m", 0.9))
        self.torso_sample_step_m = float(params.get("torso_sample_step_m", 0.05))
        self.torso_pitch_abs_max_rad = float(params.get("torso_pitch_abs_max_rad", 0.05))
        self.torso_sample_check_workers = max(
            1,
            int(params.get("torso_sample_check_workers", params.get("grasp_check_workers", 8))),
        )
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
        if not self.base_grasp_poses_key:
            raise ValueError("base_grasp_poses_key 不能为空")
        if not self.camera_grasp_poses_key:
            raise ValueError("camera_grasp_poses_key 不能为空")
        self.blackboard.register_key(
            key=self.base_grasp_poses_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.camera_grasp_poses_key,
            access=py_trees.common.Access.READ,
        )
        for key in [
            self.left_pose_key,
            self.grasp_pose_key,
            self.left_shift_pose_key,
            self.torso_pose_key,
            self.selected_grasp_pose_key,
            self.selected_base_grasp_pose_key,
            self.selected_rotate_z_180_key,
            self.selected_waist_from_base_key,
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
        self._tf = None
        self._tf_listener = None
        self._arm_controller = None
        self._torso_controller = None

    def _read_grasp_poses(self):
        if not self.blackboard.exists(self.base_grasp_poses_key):
            raise RuntimeError(f"base_link 候选抓取位姿不存在: key={self.base_grasp_poses_key}")
        grasp_poses = self.blackboard.get(self.base_grasp_poses_key)
        if not isinstance(grasp_poses, list) or not grasp_poses:
            raise RuntimeError(f"base_link 候选抓取位姿为空: key={self.base_grasp_poses_key}")
        return grasp_poses

    def _read_camera_grasp_pose_by_index(self, index):
        """读取选中候选在 camera 坐标系下的原始矩阵；缺失时返回 None，避免影响主流程。"""
        if not self.blackboard.exists(self.camera_grasp_poses_key):
            return None
        camera_grasp_poses = self.blackboard.get(self.camera_grasp_poses_key)
        if not isinstance(camera_grasp_poses, list):
            return None
        if index < 0 or index >= len(camera_grasp_poses):
            return None
        return camera_grasp_poses[index]

    def prepare_context(self):
        """准备 TF、控制器和当前左臂目标。"""
        self._ensure_arm_controller()
        knee_from_base = self._lookup_transform_matrix(KNEE_FRAME, SOURCE_FRAME)
        knee_from_base[:3, :3] = np.eye(3)
        print(f'{KNEE_FRAME=} {SOURCE_FRAME=} {knee_from_base=}\n\n\n\n')
        current_knee_from_waist = self._lookup_transform_matrix(BASE_LINK_FRAME, TARGET_FRAME)
        current_knee_from_waist = knee_from_base @ current_knee_from_waist
        left_target = self._arm_controller.get_current_end_effector_pose(
            "left",
            target_frame=TARGET_FRAME,
        )
        if left_target is None or len(left_target) != 6:
            raise RuntimeError("无法获取左臂当前末端位姿")
        return knee_from_base, current_knee_from_waist, left_target

    def compute_current_torso_target(self):
        """仅使用当前腰部位姿计算可达抓取目标。"""
        grasp_poses = self._read_grasp_poses()
        knee_from_base, current_knee_from_waist, left_target = self.prepare_context()
        selected = self._find_valid_grasp_target_for_torso_sample(
            self._current_torso_sample(current_knee_from_waist),
            grasp_poses,
            knee_from_base,
            left_target,
        )
        if selected is None:
            raise RuntimeError(f"{len(grasp_poses)} 个抓取位姿在当前腰部下均未通过检查")
        self._write_grasp_targets(selected)
        return selected

    def compute_torso_sample_target(self):
        """通过腰部采样寻找可达抓取目标。"""
        grasp_poses = self._read_grasp_poses()
        knee_from_base, current_knee_from_waist, left_target = self.prepare_context()
        current_torso_pose = self._get_current_torso_pose()
        check_sample_count = 0
        check_total_sec = 0.0
        selected = None
        samples = self._torso_ik_samples(current_knee_from_waist, current_torso_pose)
        total_sample_count = len(samples)
        # self.ros_node.get_logger().info(
        #     f"[{self.config_label}] 腰部采样抓取位姿检查开始: "
        #     f"samples={total_sample_count}, workers={self.torso_sample_check_workers}, executor=process"
        # )
        worker_count = min(self.torso_sample_check_workers, max(1, total_sample_count))
        if worker_count <= 1:
            selected, check_sample_count, check_total_sec = self._check_torso_sample_batches(
                samples,
                grasp_poses,
                knee_from_base,
                left_target,
                None,
            )
        else:
            # 关键步骤：进程池只创建一次，避免每个 batch 重复 fork/spawn 带来额外开销。
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                selected, check_sample_count, check_total_sec = self._check_torso_sample_batches(
                    samples,
                    grasp_poses,
                    knee_from_base,
                    left_target,
                    executor,
                )
        if check_sample_count > 0:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 腰部采样抓取位姿检查耗时: "
                f"total={check_total_sec:.3f}s, samples={check_sample_count}, "
                f"avg={check_total_sec / check_sample_count:.3f}s/sample"
            )
        if selected is None:
            raise RuntimeError(f"{len(grasp_poses)} 个抓取位姿在腰部采样下均未通过检查")
        self._write_grasp_targets(selected)
        return selected

    def _check_torso_sample_batches(self, samples, grasp_poses, knee_from_base, left_target, executor):
        """按 batch 检查腰部采样；executor 为 None 时走串行回退。"""
        check_sample_count = 0
        check_total_sec = 0.0
        selected = None
        for batch_index, batch in enumerate(
            self._sample_batches(samples, self.torso_sample_check_workers),
            start=1,
        ):
            check_start = time.monotonic()
            selected = self._find_valid_grasp_target_for_sample_batch(
                batch,
                grasp_poses,
                knee_from_base,
                left_target,
                executor,
            )
            batch_elapsed_sec = time.monotonic() - check_start
            check_total_sec += batch_elapsed_sec
            check_sample_count += len(batch)
            if selected is None:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 腰部采样批次检查完成: "
                    f"batch={batch_index}, batch_size={len(batch)}, "
                    f"elapsed={batch_elapsed_sec:.3f}s, selected=False"
                )
            else:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 腰部采样批次检查完成: "
                    f"batch={batch_index}, batch_size={len(batch)}, "
                    f"elapsed={batch_elapsed_sec:.3f}s, selected=True, "
                    f"sample={selected['sample']['label']}, "
                    f"candidate=第 {selected['index'] + 1} 个{selected['pose_label']}"
                )
            if selected is not None:
                break
        return selected, check_sample_count, check_total_sec

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

    def _sample_batches(self, samples, batch_size):
        """按顺序把腰部采样拆成批次，批内并行检查，批间保留原始优先级。"""
        for start_index in range(0, len(samples), max(1, int(batch_size))):
            yield samples[start_index : start_index + max(1, int(batch_size))]

    def _find_valid_grasp_target_for_sample_batch(self, samples, grasp_poses, knee_from_base, left_target, executor):
        """并行检查一批腰部采样，并按采样原始顺序选择第一个有效结果。"""
        if not samples:
            return None
        if executor is None or len(samples) <= 1:
            return self._find_valid_grasp_target_for_sample_batch_serial(
                samples,
                grasp_poses,
                knee_from_base,
                left_target,
            )

        results = [None] * len(samples)
        # 关键步骤：并行粒度放在腰部 sample 层；每个 sample 内仍按抓取候选原顺序检查。
        future_to_index = {}
        for sample_index, sample in enumerate(samples):
            future = executor.submit(
                _find_valid_grasp_target_for_torso_sample_worker,
                (
                    sample,
                    grasp_poses,
                    knee_from_base,
                    left_target,
                    self.grasp_offset_m,
                    self.left_offset_m,
                    self.left_shift_z_offset_m,
                    self.model_type,
                ),
            )
            future_to_index[future] = sample_index

        for future in as_completed(future_to_index):
            sample_index = future_to_index[future]
            try:
                result = future.result()
                if isinstance(result, dict) and result.get("__worker_error__"):
                    self.ros_node.get_logger().warning(
                        f"[{self.config_label}] 腰部采样子进程批量 IK 检查异常: "
                        f"sample={result.get('sample_label', '')}, "
                        f"candidates={result.get('candidate_count', 0)}, "
                        f"error={result.get('error', '')}\n{result.get('traceback', '')}"
                    )
                    results[sample_index] = None
                else:
                    results[sample_index] = result
            except Exception as exc:
                self.ros_node.get_logger().warning(
                    f"[{self.config_label}] 腰部采样子进程执行异常: "
                    f"sample={samples[sample_index].get('label', '')}, error={exc}\n"
                    f"{traceback.format_exc()}"
                )
                results[sample_index] = None

        for selected in results:
            if selected is not None:
                return selected
        return None

    def _find_valid_grasp_target_for_sample_batch_serial(self, samples, grasp_poses, knee_from_base, left_target):
        """串行检查一批腰部采样，作为单 worker 的回退路径。"""
        for sample in samples:
            selected = self._find_valid_grasp_target_for_torso_sample(
                sample,
                grasp_poses,
                knee_from_base,
                left_target,
            )
            if selected is not None:
                return selected
        return None

    def _find_valid_grasp_target_for_torso_sample(self, sample, grasp_poses, knee_from_base, left_target):
        # 关键步骤：候选抓取位姿固定保存在 base_link 下；腰部采样时只更新 waist <- base 变换。
        waist_from_base = np.linalg.inv(sample["knee_from_waist"]) @ knee_from_base
        for index, grasp_pose in enumerate(grasp_poses):
            for rotate_z_180 in (False, True):
                grasp_target, grasp_target_pose = self._build_right_target(
                    grasp_pose,
                    waist_from_base,
                    offset_m=self.grasp_offset_m,
                    rotate_z_180=rotate_z_180,
                )
                pose_label = "z轴旋转180度后" if rotate_z_180 else "原始"
                try:
                    grasp_valid, _, _, _ = self._check_right_pose(
                        grasp_target_pose,
                        pos_threshold=0.01,
                        angle_threshold=0.05,
                    )
                except Exception:
                    continue
                if not grasp_valid:
                    continue

                left_shift_target = list(grasp_target)
                left_shift_target[1] += self.left_offset_m
                left_shift_target[2] += self.left_shift_z_offset_m
                return {
                    "sample": sample,
                    "index": index,
                    "pose_label": pose_label,
                    "rotate_z_180": rotate_z_180,
                    "left_target": list(left_target),
                    "grasp_pose": grasp_pose,
                    "grasp_target_pose": grasp_target_pose,
                    "grasp_target": grasp_target,
                    "left_shift_target": left_shift_target,
                    "waist_from_base": waist_from_base,
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

    def _write_grasp_targets(self, selected):
        sample = selected["sample"]
        self._publish_grasp_pose(selected["grasp_target_pose"], selected["waist_from_base"])
        self._publish_raw_grasp_pose(selected["grasp_pose"])

        self.blackboard.set(self.left_pose_key, selected["left_target"], overwrite=True)
        self.blackboard.set(self.grasp_pose_key, list(selected["grasp_target"]), overwrite=True)
        self.blackboard.set(self.left_shift_pose_key, selected["left_shift_target"], overwrite=True)
        self.blackboard.set(self.selected_grasp_pose_key, selected["grasp_target_pose"], overwrite=True)
        self.blackboard.set(self.selected_base_grasp_pose_key, selected["grasp_pose"], overwrite=True)
        self.blackboard.set(self.selected_rotate_z_180_key, bool(selected["rotate_z_180"]), overwrite=True)
        self.blackboard.set(self.selected_waist_from_base_key, selected["waist_from_base"], overwrite=True)
        self.blackboard.set(self.torso_enabled_key, bool(sample["enabled"]), overwrite=True)
        if sample["enabled"]:
            self.blackboard.set(self.torso_pose_key, list(sample["torso_pose"]), overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选择{sample['label']}下第 "
            f"{selected['index'] + 1} 个{selected['pose_label']}右臂抓取位姿，"
            f"已写入 blackboard: left={self.left_pose_key}, "
            f"grasp={self.grasp_pose_key}, "
            f"left_shift={self.left_shift_pose_key}, "
            f"use_torso={self.torso_enabled_key}"
        )
        self._log_selected_grasp_source_poses(selected)

    def _log_selected_grasp_source_poses(self, selected):
        """打印选中候选在 camera、base_link 和 waist_yaw_link 下的位姿，便于核对坐标转换。"""
        camera_grasp_pose = self._read_camera_grasp_pose_by_index(int(selected["index"]))
        base_grasp_target = self._target_from_pose(selected["grasp_pose"])
        waist_grasp_target = self._target_from_pose(selected["grasp_target_pose"])
        if camera_grasp_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 选中候选缺少 camera 原始抓取位姿: "
                f"key={self.camera_grasp_poses_key}, index={selected['index'] + 1}"
            )
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 选中候选抓取位姿坐标: "
                f"camera=<missing>, "
                f"base_link={self._format_target_pose(base_grasp_target)}, "
                f"waist_yaw_link={self._format_target_pose(waist_grasp_target)}"
            )
            return

        camera_grasp_target = self._target_from_pose(camera_grasp_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选中候选抓取位姿坐标: "
            f"camera={self._format_target_pose(camera_grasp_target)}, "
            f"base_link={self._format_target_pose(base_grasp_target)}, "
            f"waist_yaw_link={self._format_target_pose(waist_grasp_target)}"
        )

    def refresh_selected_grasp_target(self):
        """腰部运动后基于当前 waist_yaw_link<-base_link 重新计算最终抓取目标。"""
        self._ensure_arm_controller()
        if not self.blackboard.exists(self.selected_base_grasp_pose_key):
            raise RuntimeError(f"选中的 base_link 抓取矩阵不存在: key={self.selected_base_grasp_pose_key}")
        if not self.blackboard.exists(self.selected_rotate_z_180_key):
            raise RuntimeError(f"选中的抓取翻转标记不存在: key={self.selected_rotate_z_180_key}")

        grasp_pose = self.blackboard.get(self.selected_base_grasp_pose_key)
        rotate_z_180 = bool(self.blackboard.get(self.selected_rotate_z_180_key))
        waist_from_base = self._lookup_transform_matrix(TARGET_FRAME, SOURCE_FRAME)
        left_target = self._arm_controller.get_current_end_effector_pose(
            "left",
            target_frame=TARGET_FRAME,
        )
        if left_target is None or len(left_target) != 6:
            raise RuntimeError("腰部运动后无法获取左臂当前末端位姿")

        # 关键步骤：目标物仍固定在 base_link，下发前按当前腰部 TF 重投影到 waist_yaw_link。
        grasp_target, grasp_target_pose = self._build_right_target(
            grasp_pose,
            waist_from_base,
            offset_m=self.grasp_offset_m,
            rotate_z_180=rotate_z_180,
        )
        base_grasp_target_pose = np.linalg.inv(waist_from_base) @ grasp_target_pose
        base_grasp_target = self._target_from_pose(base_grasp_target_pose)
        # 关键步骤：先打印重算目标，再做 IK-FK 检查，失败时也能定位目标坐标是否异常。
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 计算预抓取前最终抓取目标 pose: "
            f"base_link={self._format_target_pose(base_grasp_target)}, "
            f"waist_yaw_link={self._format_target_pose(grasp_target)}"
        )
        grasp_valid, _, pos_error, angle_error = self._check_right_pose(
            grasp_target_pose,
            pos_threshold=0.01,
            angle_threshold=0.05,
        )
        if not grasp_valid:
            raise RuntimeError(
                "腰部运动后重算抓取位姿未通过 IK-FK 检查: "
                f"pos={pos_error:.4f}m/0.0100m, angle={angle_error:.4f}rad/0.0500rad"
            )

        left_shift_target = list(grasp_target)
        left_shift_target[1] += self.left_offset_m
        left_shift_target[2] += self.left_shift_z_offset_m

        self._publish_grasp_pose(grasp_target_pose, waist_from_base)
        self.blackboard.set(self.left_pose_key, list(left_target), overwrite=True)
        self.blackboard.set(self.grasp_pose_key, list(grasp_target), overwrite=True)
        self.blackboard.set(self.left_shift_pose_key, left_shift_target, overwrite=True)
        self.blackboard.set(self.selected_grasp_pose_key, grasp_target_pose, overwrite=True)
        self.blackboard.set(self.selected_waist_from_base_key, waist_from_base, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 腰部运动后已按当前 TF 刷新右臂抓取目标: "
            f"grasp={grasp_target}, left_shift={left_shift_target}"
        )
        return grasp_target

    def compute_pregrasp_target(self, pregrasp_pose_key):
        """根据最终抓取位姿计算预抓取目标。"""
        self._ensure_arm_controller()
        if not self.blackboard.exists(self.selected_grasp_pose_key):
            raise RuntimeError(f"最终抓取矩阵不存在: key={self.selected_grasp_pose_key}")
        grasp_target_pose = self.blackboard.get(self.selected_grasp_pose_key)
        offsets = np.linspace(
            self.pregrasp_offset_min_m,
            self.pregrasp_offset_max_m,
            self.pregrasp_offset_samples,
            endpoint=True,
        )
        middle = len(offsets) // 2
        pregrasp_offsets_m = [offsets[middle]]
        for step in range(1, len(offsets)):
            for offset_index in (middle + step, middle - step):
                if 0 <= offset_index < len(offsets):
                    pregrasp_offsets_m.append(offsets[offset_index])

        for pregrasp_offset_m in pregrasp_offsets_m:
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
                    f"[{self.config_label}] 预抓取位姿 offset={pregrasp_offset_m:.3f}m 检查失败: {exc}"
                )
                continue
            if not pregrasp_valid:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 预抓取位姿 offset={pregrasp_offset_m:.3f}m "
                    f"IK-FK检查不通过: pos={pos_error:.4f}m/0.0300m, "
                    f"angle={angle_error:.4f}rad/0.2000rad"
                )
                continue
            # 关键步骤：预抓取目标在真正执行前才写入，确保使用的是最终抓取位姿。
            self._publish_target_pose(pregrasp_pose)
            self.blackboard.set(pregrasp_pose_key, list(pregrasp_target), overwrite=True)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已计算预抓取位姿: "
                f"offset={pregrasp_offset_m:.3f}m, key={pregrasp_pose_key}"
            )
            return pregrasp_target
        raise RuntimeError("所有预抓取 offset 均未通过 IK-FK 检查")

    def _build_right_target(self, grasp_pose, target_from_source, offset_m, rotate_z_180=False):
        offset = np.eye(4)
        offset[2, 3] = -offset_m
        source_target = grasp_pose @ offset @ SDK_AXIS_TRANSFORM

        target_pose = target_from_source @ source_target
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

    @staticmethod
    def _format_target_pose(target):
        return (
            f"x={target[0]:.4f}, y={target[1]:.4f}, z={target[2]:.4f}, "
            f"yaw={target[3]:.2f}deg, pitch={target[4]:.2f}deg, roll={target[5]:.2f}deg"
        )

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

    def _publish_grasp_pose(self, target_pose, target_from_base):
        source_pose = np.linalg.inv(target_from_base) @ target_pose
        quaternion = self._tf.transformations.quaternion_from_matrix(source_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = SOURCE_FRAME
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
        message.header.frame_id = SOURCE_FRAME
        message.pose.position.x = float(grasp_pose[0, 3])
        message.pose.position.y = float(grasp_pose[1, 3])
        message.pose.position.z = float(grasp_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.raw_grasp_pose_pub.publish(message)
