"""双臂抓取目标计算工具，供多个行为树节点复用。"""

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

from tree.constants import (
    BASE_LINK_FRAME,
    CHASSIS_FRAME,
    FIXED_KNEE_FRAME,
    KNEE_LINK_FRAME,
    MAP_FRAME,
    MODEL_TYPE_KEY,
    ROBOT_SERVICES_KEY,
    WAIST_YAW_LINK_FRAME,
)
from tree.utils.geometry import (
    format_xyz_ypr,
    lookup_base_from_map_via_chassis,
    lookup_transform_matrix,
    make_xz_pitch_transform,
    matrix_to_xyz_ypr,
)
from .grasp_math import build_grasp_target


_CPP_SEARCH_UNAVAILABLE = object()
_ARM_SEARCH_ORDER = ("right", "left")
_ARM_EEF_FRAMES = {
    "right": "zarm_r7_link",
    "left": "zarm_l7_link",
}


def _ordered_pregrasp_offsets(offset_min_m, offset_max_m, offset_samples):
    """生成从中间向两侧扩展的预抓取 offset 顺序。"""
    offsets = np.linspace(offset_min_m, offset_max_m, offset_samples, endpoint=True)
    middle = len(offsets) // 2
    ordered = [float(offsets[middle])]
    for step in range(1, len(offsets)):
        for offset_index in (middle + step, middle - step):
            if 0 <= offset_index < len(offsets):
                ordered.append(float(offsets[offset_index]))
    return ordered


def _build_pregrasp_pose(grasp_target_pose, grasp_offset_m, pregrasp_offset_m):
    """沿抓取目标 z 轴生成预抓取位姿。"""
    pregrasp_pose = np.array(grasp_target_pose, copy=True)
    pregrasp_pose[:3, 3] += (
        pregrasp_offset_m - grasp_offset_m
    ) * grasp_target_pose[:3, 2]
    return pregrasp_pose


def _find_valid_grasp_target_for_torso_sample_worker(args):
    """子进程检查单个腰部 sample，避免把 ROS/blackboard 对象传入进程池。"""
    (
        sample,
        grasp_poses,
        poses_frame,
        fixed_knee_from_base,
        arm_targets,
        grasp_offset_m,
        pregrasp_offsets_m,
        model_type,
    ) = args
    waist_from_base = np.linalg.inv(sample["fixed_knee_from_waist"]) @ fixed_knee_from_base
    base_grasp_poses = _project_grasp_poses_to_sample_base(grasp_poses, poses_frame, sample)
    for arm_side in _ARM_SEARCH_ORDER:
        candidates = []
        for index, grasp_pose in enumerate(base_grasp_poses):
            for rotate_z_180 in (False, True):
                grasp_target, grasp_target_pose = build_grasp_target(
                    grasp_pose,
                    waist_from_base,
                    offset_m=grasp_offset_m,
                    rotate_z_180=rotate_z_180,
                )
                candidates.append(
                    {
                        "index": index,
                        "pose_label": "z轴旋转180度后" if rotate_z_180 else "原始",
                        "rotate_z_180": rotate_z_180,
                        "source_grasp_pose": grasp_poses[index],
                        "grasp_pose": grasp_pose,
                        "grasp_target": grasp_target,
                        "grasp_target_pose": grasp_target_pose,
                    }
                )
        if not candidates:
            continue
        try:
            valid_mask, _, _, _ = IKAnalytical.check_pose_validity_batch(
                eef_pos_array=np.stack([item["grasp_target_pose"][:3, 3] for item in candidates]),
                eef_quat_xyzw_array=np.stack([
                    tf_trans.quaternion_from_matrix(item["grasp_target_pose"])
                    for item in candidates
                ]),
                eef_frame=_ARM_EEF_FRAMES[arm_side],
                model_type=model_type,
                pos_threshold=0.01,
                angle_threshold=0.05,
            )
        except Exception as exc:
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
            for pregrasp_offset_m in pregrasp_offsets_m:
                pregrasp_pose = _build_pregrasp_pose(
                    candidate["grasp_target_pose"], grasp_offset_m, pregrasp_offset_m
                )
                try:
                    pregrasp_valid, _, _, _ = IKAnalytical.check_pose_validity(
                        eef_pos=pregrasp_pose[:3, 3],
                        eef_quat_xyzw=tf_trans.quaternion_from_matrix(pregrasp_pose),
                        eef_frame=_ARM_EEF_FRAMES[arm_side],
                        model_type=model_type,
                        pos_threshold=0.03,
                        angle_threshold=0.50,
                    )
                except Exception:
                    continue
                if not pregrasp_valid:
                    continue
                return {
                    "sample": sample,
                    "arm_side": arm_side,
                    "index": candidate["index"],
                    "pose_label": candidate["pose_label"],
                    "rotate_z_180": bool(candidate["rotate_z_180"]),
                    "poses_frame": poses_frame,
                    "arm_targets": {side: list(target) for side, target in arm_targets.items()},
                    "source_grasp_pose": candidate["source_grasp_pose"],
                    "grasp_pose": candidate["grasp_pose"],
                    "grasp_target_pose": candidate["grasp_target_pose"],
                    "grasp_target": candidate["grasp_target"],
                    "pregrasp_target_pose": pregrasp_pose,
                    "pregrasp_target": matrix_to_xyz_ypr(pregrasp_pose),
                    "pregrasp_offset_m": pregrasp_offset_m,
                    "waist_from_base": waist_from_base,
                }
    return None


def _project_grasp_poses_to_sample_base(grasp_poses, poses_frame, sample):
    """按 sample 的假设底盘位姿，把抓取位姿投影到对应 base_link 坐标系。"""
    if poses_frame == MAP_FRAME:
        base_from_map = sample.get("base_from_map")
        if base_from_map is None:
            raise RuntimeError("map 抓取位姿需要 sample.base_from_map")
        return [base_from_map @ grasp_pose for grasp_pose in grasp_poses]
    return grasp_poses


class GraspTargetComputer:
    """封装双手抓取目标选择、腰部采样和调试位姿发布。"""

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
        self.map_grasp_poses_key = str(
            params.get("map_grasp_poses_key", "grasp_object_map_grasp_poses")
        ).strip()
        self.chassis_frame = str(params.get("chassis_frame", CHASSIS_FRAME)).strip()
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
        self.pregrasp_pose_key = str(
            params.get("pregrasp_pose_key", "grasp_object_pregrasp_pose")
        ).strip()
        self.selected_arm_side_key = str(
            params.get("selected_arm_side_key", "grasp_object_selected_arm_side")
        ).strip()
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
        self.chassis_lateral_sample_min_m = float(params.get("chassis_lateral_sample_min_m", -0.2))
        self.chassis_lateral_sample_max_m = float(params.get("chassis_lateral_sample_max_m", 0.2))
        self.chassis_lateral_sample_step_m = float(params.get("chassis_lateral_sample_step_m", 0.1))
        self.chassis_lateral_offset_key = str(
            params.get("chassis_lateral_offset_key", "grasp_object_chassis_lateral_offset")
        ).strip()
        self.chassis_lateral_enabled_key = str(
            params.get("chassis_lateral_enabled_key", "grasp_object_use_chassis_lateral")
        ).strip()
        self.use_cpp_grasp_search = self._to_bool(params.get("use_cpp_grasp_search", True))
        self.cpp_search_fallback_on_error = self._to_bool(
            params.get("cpp_search_fallback_on_error", True)
        )
        self.cpp_search_workers = max(
            1,
            int(params.get("cpp_search_workers", self.torso_sample_check_workers)),
        )
        self.cpp_search_batch_size = max(
            0,
            int(params.get("cpp_search_batch_size", 0)),
        )
        self.preserve_motion_flags = self._to_bool(
            params.get("preserve_motion_flags", False)
        )
        self.selected_map_grasp_pose_key = str(
            params.get("selected_map_grasp_pose_key", "grasp_object_selected_map_grasp_pose")
        ).strip()
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
        if not self.map_grasp_poses_key:
            raise ValueError("map_grasp_poses_key 不能为空")
        if not self.chassis_frame:
            raise ValueError("chassis_frame 不能为空")
        self.blackboard.register_key(
            key=self.base_grasp_poses_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.camera_grasp_poses_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.map_grasp_poses_key,
            access=py_trees.common.Access.READ,
        )
        for key in [
            self.left_pose_key,
            self.grasp_pose_key,
            self.pregrasp_pose_key,
            self.torso_pose_key,
            self.chassis_lateral_offset_key,
            self.selected_grasp_pose_key,
            self.selected_base_grasp_pose_key,
            self.selected_map_grasp_pose_key,
            self.selected_rotate_z_180_key,
            self.selected_waist_from_base_key,
            self.selected_arm_side_key,
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
        if not self.chassis_lateral_enabled_key:
            raise ValueError("chassis_lateral_enabled_key 不能为空")
        self.blackboard.register_key(
            key=self.chassis_lateral_enabled_key,
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
        if self.chassis_lateral_sample_max_m < self.chassis_lateral_sample_min_m:
            raise ValueError("chassis_lateral_sample_max_m 必须大于等于 chassis_lateral_sample_min_m")
        if self.chassis_lateral_sample_step_m <= 0.0:
            raise ValueError("chassis_lateral_sample_step_m 必须大于 0")
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
        if self.blackboard.exists(self.map_grasp_poses_key):
            grasp_poses = self.blackboard.get(self.map_grasp_poses_key)
            if isinstance(grasp_poses, list) and grasp_poses:
                return grasp_poses, MAP_FRAME
        if not self.blackboard.exists(self.base_grasp_poses_key):
            raise RuntimeError(f"base_link 候选抓取位姿不存在: key={self.base_grasp_poses_key}")
        grasp_poses = self.blackboard.get(self.base_grasp_poses_key)
        if not isinstance(grasp_poses, list) or not grasp_poses:
            raise RuntimeError(f"base_link 候选抓取位姿为空: key={self.base_grasp_poses_key}")
        return grasp_poses, BASE_LINK_FRAME

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
        """准备 TF、控制器和当前双臂目标。"""
        self._ensure_runtime_context()
        fixed_knee_from_base = self._lookup_fixed_knee_from_base()
        base_from_waist = self._lookup_transform_matrix(BASE_LINK_FRAME, WAIST_YAW_LINK_FRAME)
        base_from_map = self._lookup_base_from_map_via_chassis()
        # 关键步骤：fixed_knee 是计算用虚拟坐标系，原点跟随 knee_link，坐标轴保持与 base_link 对齐。
        current_fixed_knee_from_waist = fixed_knee_from_base @ base_from_waist
        arm_targets = {}
        for arm_side in ("left", "right"):
            target = self._arm_controller.get_current_end_effector_pose(
                arm_side,
                target_frame=WAIST_YAW_LINK_FRAME,
            )
            if target is None or len(target) != 6:
                raise RuntimeError(f"无法获取{arm_side}臂当前末端位姿")
            arm_targets[arm_side] = list(target)
        return fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map

    def compute_current_torso_target(self):
        """仅使用当前腰部位姿计算可达抓取目标。"""
        grasp_poses, poses_frame = self._read_grasp_poses()
        fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map = self.prepare_context()
        current_sample = self._current_torso_sample(current_fixed_knee_from_waist)
        current_sample["base_from_map"] = base_from_map
        selected = None
        if self.use_cpp_grasp_search:
            selected = self._find_valid_grasp_target_cpp(
                [current_sample],
                grasp_poses,
                poses_frame,
                fixed_knee_from_base,
                arm_targets,
                fallback_label="当前腰部",
            )
        if selected is _CPP_SEARCH_UNAVAILABLE:
            selected = self._find_valid_grasp_target_for_torso_sample(
                current_sample,
                grasp_poses,
                poses_frame,
                fixed_knee_from_base,
                arm_targets,
            )
        if selected is None:
            raise RuntimeError(f"{len(grasp_poses)} 个抓取位姿在当前腰部下均未通过检查")
        self._write_grasp_targets(selected)
        return selected

    def compute_torso_sample_target(self):
        """通过腰部采样寻找可达抓取目标。"""
        grasp_poses, poses_frame = self._read_grasp_poses()
        fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map = self.prepare_context()
        current_torso_pose = self._get_current_torso_pose()
        check_sample_count = 0
        check_total_sec = 0.0
        selected = None
        torso_samples = [self._current_torso_sample(current_fixed_knee_from_waist)]
        torso_samples.extend(self._torso_ik_samples(current_fixed_knee_from_waist, current_torso_pose))
        samples = self._chassis_lateral_torso_samples(torso_samples, base_from_map, poses_frame)
        total_sample_count = len(samples)
        # 关键步骤：腰部采样会对每个原始 grasp pose 检查原始姿态和 z 轴旋转 180 度后的姿态。
        grasp_candidate_count = len(grasp_poses) * 2
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 腰部采样目标物抓取位姿统计: "
            f"poses_frame={poses_frame}, grasp_poses={len(grasp_poses)}, "
            f"target_candidates_per_sample={grasp_candidate_count}, "
            f"samples={total_sample_count}, "
            f"total_checks={total_sample_count * grasp_candidate_count}"
        )
        # self.ros_node.get_logger().info(
        #     f"[{self.config_label}] 腰部采样抓取位姿检查开始: "
        #     f"samples={total_sample_count}, workers={self.torso_sample_check_workers}, executor=process"
        # )
        if self.use_cpp_grasp_search:
            check_start = time.monotonic()
            selected = self._find_valid_grasp_target_cpp(
                samples,
                grasp_poses,
                poses_frame,
                fixed_knee_from_base,
                arm_targets,
                fallback_label="腰部采样",
            )
            check_total_sec = time.monotonic() - check_start
            check_sample_count = total_sample_count
            self.ros_node.get_logger().info(
                f"[{self.config_label}] C++腰部采样抓取位姿检查完成: "
                f"elapsed={check_total_sec:.3f}s, samples={check_sample_count}, "
                f"selected={selected is not None and selected is not _CPP_SEARCH_UNAVAILABLE}"
            )

        if selected is _CPP_SEARCH_UNAVAILABLE:
            worker_count = min(self.torso_sample_check_workers, max(1, total_sample_count))
            if worker_count <= 1:
                selected, check_sample_count, check_total_sec = self._check_torso_sample_batches(
                    samples,
                    grasp_poses,
                    poses_frame,
                    fixed_knee_from_base,
                    arm_targets,
                    None,
                )
            else:
                # 关键步骤：进程池只创建一次，避免每个 batch 重复 fork/spawn 带来额外开销。
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    selected, check_sample_count, check_total_sec = self._check_torso_sample_batches(
                        samples,
                        grasp_poses,
                        poses_frame,
                        fixed_knee_from_base,
                        arm_targets,
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

    @staticmethod
    def _to_bool(value):
        """解析 JSON 参数里的布尔值。"""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    def _find_valid_grasp_target_cpp(
        self,
        samples,
        grasp_poses,
        poses_frame,
        fixed_knee_from_base,
        arm_targets,
        fallback_label,
    ):
        """调用 C++ 内核搜索可达抓取目标，失败时按配置回退 Python。"""
        if not samples:
            return None
        try:
            from . import _grasp_search_cpp

            sample_fixed_knee_from_waist = np.asarray(
                [sample["fixed_knee_from_waist"] for sample in samples],
                dtype=float,
            )
            sample_base_from_map = np.asarray(
                [sample.get("base_from_map", np.eye(4)) for sample in samples],
                dtype=float,
            )
            sample_enabled = np.asarray(
                [bool(sample.get("enabled", False)) for sample in samples],
                dtype=bool,
            )
            sample_chassis_lateral_enabled = np.asarray(
                [bool(sample.get("chassis_lateral_enabled", False)) for sample in samples],
                dtype=bool,
            )
            grasp_pose_array = np.asarray(grasp_poses, dtype=float)
            result = _grasp_search_cpp.find_grasp_target(
                sample_fixed_knee_from_waist=sample_fixed_knee_from_waist,
                sample_base_from_map=sample_base_from_map,
                sample_enabled=sample_enabled,
                sample_chassis_lateral_enabled=sample_chassis_lateral_enabled,
                grasp_poses=grasp_pose_array,
                poses_frame=poses_frame,
                fixed_knee_from_base=np.asarray(fixed_knee_from_base, dtype=float),
                grasp_offset_m=self.grasp_offset_m,
                model_type=self.model_type,
                pos_threshold=0.01,
                angle_threshold=0.05,
                pregrasp_offset_min_m=self.pregrasp_offset_min_m,
                pregrasp_offset_max_m=self.pregrasp_offset_max_m,
                pregrasp_offset_samples=self.pregrasp_offset_samples,
                cpp_search_workers=self.cpp_search_workers,
                cpp_search_batch_size=self.cpp_search_batch_size,
            )
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] C++{fallback_label}抓取搜索不可用，"
                f"{'回退 Python' if self.cpp_search_fallback_on_error else '不回退'}: {exc}"
            )
            if self.cpp_search_fallback_on_error:
                return _CPP_SEARCH_UNAVAILABLE
            raise

        if result is None:
            return None

        sample_index = int(result["sample_index"])
        grasp_index = int(result["grasp_index"])
        rotate_z_180 = bool(result["rotate_z_180"])
        pose_label = "z轴旋转180度后" if rotate_z_180 else "原始"
        return {
            "sample": samples[sample_index],
            "arm_side": str(result["arm_side"]),
            "index": grasp_index,
            "pose_label": pose_label,
            "rotate_z_180": rotate_z_180,
            "poses_frame": poses_frame,
            "arm_targets": {side: list(target) for side, target in arm_targets.items()},
            "source_grasp_pose": grasp_poses[grasp_index],
            "grasp_pose": np.asarray(result["grasp_pose"], dtype=float),
            "grasp_target_pose": np.asarray(result["grasp_target_pose"], dtype=float),
            "grasp_target": list(result["grasp_target"]),
            "pregrasp_target_pose": np.asarray(result["pregrasp_target_pose"], dtype=float),
            "pregrasp_target": list(result["pregrasp_target"]),
            "pregrasp_offset_m": float(result["pregrasp_offset_m"]),
            "waist_from_base": np.asarray(result["waist_from_base"], dtype=float),
            "pos_error": float(result.get("pos_error", 0.0)),
            "angle_error": float(result.get("angle_error", 0.0)),
        }

    def _check_torso_sample_batches(self, samples, grasp_poses, poses_frame, fixed_knee_from_base, arm_targets, executor):
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
                poses_frame,
                fixed_knee_from_base,
                arm_targets,
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

    def _ensure_runtime_context(self):
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

    def _current_torso_sample(self, fixed_knee_from_waist):
        _, pitch, _ = self._tf.transformations.euler_from_matrix(fixed_knee_from_waist)
        return {
            "label": (
                "当前腰部"
                f"(x={fixed_knee_from_waist[0, 3]:.3f}, z={fixed_knee_from_waist[2, 3]:.3f}, "
                f"pitch={pitch:.3f})"
            ),
            "enabled": False,
            "fixed_knee_from_waist": fixed_knee_from_waist,
            "knee_from_waist": fixed_knee_from_waist,
            "torso_pose": None,
            "x": float(fixed_knee_from_waist[0, 3]),
            "z": float(fixed_knee_from_waist[2, 3]),
            "pitch": float(pitch),
            "motion_cost": 0.0,
        }

    def _torso_ik_samples(self, current_fixed_knee_from_waist, current_torso_pose):
        current_x = float(current_fixed_knee_from_waist[0, 3])
        current_z = float(current_fixed_knee_from_waist[2, 3])
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
                    "fixed_knee_from_waist": make_xz_pitch_transform(waist_x, waist_z, pitch),
                    "torso_pose": torso_pose,
                    "x": waist_x,
                    "z": waist_z,
                    "pitch": pitch,
                }
                sample["knee_from_waist"] = sample["fixed_knee_from_waist"]
                score = (waist_x - current_x) ** 2 + (waist_z - current_z) ** 2
                sample["motion_cost"] = float(score)
                samples.append((score, sample))

        samples.sort(key=lambda item: item[0])
        return [sample for _, sample in samples]

    def _chassis_lateral_torso_samples(self, torso_samples, current_base_from_map, poses_frame):
        """组合底盘横向位移和腰部采样，优先检查横移距离更小的组合。"""
        offsets = self._chassis_lateral_offsets()
        combined = []
        for offset_index, lateral_offset in enumerate(offsets):
            base_from_map = self._base_from_map_after_lateral_offset(
                current_base_from_map,
                lateral_offset,
            )
            for torso_index, torso_sample in enumerate(torso_samples):
                sample = dict(torso_sample)
                sample["base_from_map"] = base_from_map
                sample["chassis_lateral_offset_m"] = float(lateral_offset)
                sample["chassis_lateral_enabled"] = abs(float(lateral_offset)) > 1e-6
                sample["poses_frame"] = poses_frame
                sample["label"] = (
                    f"底盘横移(y={lateral_offset:.3f}) + {torso_sample['label']}"
                )
                # 关键步骤：使用字典序严格优先最小底盘横移，再比较 torso x/z 位移平方和。
                combined.append(
                    (
                        abs(float(lateral_offset)),
                        float(torso_sample.get("motion_cost", 0.0)),
                        offset_index,
                        torso_index,
                        sample,
                    )
                )
        combined.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [sample for _, _, _, _, sample in combined]

    def _chassis_lateral_offsets(self):
        """生成底盘横向采样值，顺序为 0、-step、+step、-2step、+2step。"""
        values = self._sample_axis_values(
            self.chassis_lateral_sample_min_m,
            self.chassis_lateral_sample_max_m,
            step=self.chassis_lateral_sample_step_m,
        )
        values.sort(key=lambda value: (abs(float(value)), float(value)))
        return values

    @staticmethod
    def _base_from_map_after_lateral_offset(current_base_from_map, lateral_offset_m):
        """计算底盘横移后的 sample_base_link <- map 变换。"""
        base_from_sample_base = np.eye(4)
        base_from_sample_base[1, 3] = -float(lateral_offset_m)
        return base_from_sample_base @ current_base_from_map

    def _sample_axis_values(self, lower, upper, step=None):
        values = []
        value = float(lower)
        sample_step = float(self.torso_sample_step_m if step is None else step)
        while value <= upper + 1e-9:
            values.append(round(value, 10))
            value += sample_step
        if values and values[-1] < upper - 1e-9:
            values.append(float(upper))
        return values

    def _sample_batches(self, samples, batch_size):
        """按顺序把腰部采样拆成批次，批内并行检查，批间保留原始优先级。"""
        for start_index in range(0, len(samples), max(1, int(batch_size))):
            yield samples[start_index : start_index + max(1, int(batch_size))]

    def _find_valid_grasp_target_for_sample_batch(self, samples, grasp_poses, poses_frame, fixed_knee_from_base, arm_targets, executor):
        """并行检查一批腰部采样，并按采样原始顺序选择第一个有效结果。"""
        if not samples:
            return None
        if executor is None or len(samples) <= 1:
            for sample in samples:
                selected = self._find_valid_grasp_target_for_torso_sample(
                    sample,
                    grasp_poses,
                    poses_frame,
                    fixed_knee_from_base,
                    arm_targets,
                )
                if selected is not None:
                    return selected
            return None

        results = [None] * len(samples)
        # 关键步骤：并行粒度放在腰部 sample 层；每个 sample 内仍按抓取候选原顺序检查。
        future_to_index = {}
        for sample_index, sample in enumerate(samples):
            future = executor.submit(
                _find_valid_grasp_target_for_torso_sample_worker,
                (
                    sample,
                    grasp_poses,
                    poses_frame,
                    fixed_knee_from_base,
                    arm_targets,
                    self.grasp_offset_m,
                    _ordered_pregrasp_offsets(
                        self.pregrasp_offset_min_m,
                        self.pregrasp_offset_max_m,
                        self.pregrasp_offset_samples,
                    ),
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

    def _find_valid_grasp_target_for_torso_sample(self, sample, grasp_poses, poses_frame, fixed_knee_from_base, arm_targets):
        # 关键步骤：map 抓取位姿按 sample 的假设底盘位姿投影到 base_link，再做腰部采样检查。
        waist_from_base = np.linalg.inv(sample["fixed_knee_from_waist"]) @ fixed_knee_from_base
        base_grasp_poses = _project_grasp_poses_to_sample_base(grasp_poses, poses_frame, sample)
        pregrasp_offsets_m = _ordered_pregrasp_offsets(
            self.pregrasp_offset_min_m,
            self.pregrasp_offset_max_m,
            self.pregrasp_offset_samples,
        )
        # 关键步骤：同一 sample 内右手优先，但必须同时具备抓取和预抓取可达性。
        for arm_side in _ARM_SEARCH_ORDER:
            for index, grasp_pose in enumerate(base_grasp_poses):
                for rotate_z_180 in (False, True):
                    grasp_target, grasp_target_pose = build_grasp_target(
                        grasp_pose,
                        waist_from_base,
                        offset_m=self.grasp_offset_m,
                        rotate_z_180=rotate_z_180,
                    )
                    pose_label = "z轴旋转180度后" if rotate_z_180 else "原始"
                    try:
                        grasp_valid, _, _, _ = self._check_arm_pose(
                            arm_side,
                            grasp_target_pose,
                            pos_threshold=0.01,
                            angle_threshold=0.05,
                        )
                    except Exception:
                        continue
                    if not grasp_valid:
                        continue
                    for pregrasp_offset_m in pregrasp_offsets_m:
                        pregrasp_pose = _build_pregrasp_pose(
                            grasp_target_pose,
                            self.grasp_offset_m,
                            pregrasp_offset_m,
                        )
                        try:
                            pregrasp_valid, _, _, _ = self._check_arm_pose(
                                arm_side,
                                pregrasp_pose,
                                pos_threshold=0.03,
                                angle_threshold=0.50,
                            )
                        except Exception:
                            continue
                        if not pregrasp_valid:
                            continue
                        return {
                            "sample": sample,
                            "arm_side": arm_side,
                            "index": index,
                            "pose_label": pose_label,
                            "rotate_z_180": rotate_z_180,
                            "poses_frame": poses_frame,
                            "arm_targets": {
                                side: list(target) for side, target in arm_targets.items()
                            },
                            "source_grasp_pose": grasp_poses[index],
                            "grasp_pose": grasp_pose,
                            "grasp_target_pose": grasp_target_pose,
                            "grasp_target": grasp_target,
                            "pregrasp_target_pose": pregrasp_pose,
                            "pregrasp_target": matrix_to_xyz_ypr(pregrasp_pose),
                            "pregrasp_offset_m": pregrasp_offset_m,
                            "waist_from_base": waist_from_base,
                        }
        return None

    def _check_arm_pose(self, arm_side, target_pose, pos_threshold, angle_threshold):
        """按左右手选择 SDK 末端 frame 执行 IK-FK 检查。"""
        return IKAnalytical.check_pose_validity(
            eef_pos=target_pose[:3, 3],
            eef_quat_xyzw=self._tf.transformations.quaternion_from_matrix(target_pose),
            eef_frame=_ARM_EEF_FRAMES[arm_side],
            model_type=self.model_type,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )

    def _write_grasp_targets(self, selected, preserve_motion_flags=None):
        """写入选中抓取方案，到位后复核时可保留原底盘/torso 执行标记。"""
        sample = selected["sample"]
        preserve_flags = (
            self.preserve_motion_flags
            if preserve_motion_flags is None
            else bool(preserve_motion_flags)
        )
        self._publish_grasp_pose(selected["grasp_target_pose"], selected["waist_from_base"])
        self._publish_raw_grasp_pose(selected["grasp_pose"])

        self.blackboard.set(
            self.left_pose_key,
            selected["arm_targets"]["left"],
            overwrite=True,
        )
        self.blackboard.set(self.grasp_pose_key, list(selected["grasp_target"]), overwrite=True)
        self.blackboard.set(
            self.pregrasp_pose_key,
            list(selected["pregrasp_target"]),
            overwrite=True,
        )
        self.blackboard.set(
            self.selected_arm_side_key,
            selected["arm_side"],
            overwrite=True,
        )
        self.blackboard.set(self.selected_grasp_pose_key, selected["grasp_target_pose"], overwrite=True)
        self.blackboard.set(self.selected_base_grasp_pose_key, selected["grasp_pose"], overwrite=True)
        selected_map_grasp_pose = (
            selected["source_grasp_pose"]
            if selected.get("poses_frame") == MAP_FRAME
            else None
        )
        self.blackboard.set(self.selected_map_grasp_pose_key, selected_map_grasp_pose, overwrite=True)
        self.blackboard.set(self.selected_rotate_z_180_key, bool(selected["rotate_z_180"]), overwrite=True)
        self.blackboard.set(self.selected_waist_from_base_key, selected["waist_from_base"], overwrite=True)
        lateral_offset = float(sample.get("chassis_lateral_offset_m", 0.0))
        if not preserve_flags:
            self.blackboard.set(self.torso_enabled_key, bool(sample["enabled"]), overwrite=True)
            self.blackboard.set(self.chassis_lateral_offset_key, lateral_offset, overwrite=True)
            self.blackboard.set(
                self.chassis_lateral_enabled_key,
                bool(sample.get("chassis_lateral_enabled", False)),
                overwrite=True,
            )
            if sample["enabled"]:
                self.blackboard.set(self.torso_pose_key, list(sample["torso_pose"]), overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选择{sample['label']}下第 "
            f"{selected['index'] + 1} 个{selected['pose_label']}{selected['arm_side']}臂抓取位姿，"
            f"已写入 blackboard: side={self.selected_arm_side_key}, "
            f"grasp={self.grasp_pose_key}, "
            f"use_torso={self.torso_enabled_key}, "
            f"chassis_lateral={lateral_offset:.3f}m"
        )
        self._log_selected_grasp_source_poses(selected)

    def _log_selected_grasp_source_poses(self, selected):
        """打印选中候选在 camera、base_link 和 waist_yaw_link 下的位姿，便于核对坐标转换。"""
        camera_grasp_pose = self._read_camera_grasp_pose_by_index(int(selected["index"]))
        base_grasp_target = matrix_to_xyz_ypr(selected["grasp_pose"])
        waist_grasp_target = matrix_to_xyz_ypr(selected["grasp_target_pose"])
        if camera_grasp_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 选中候选缺少 camera 原始抓取位姿: "
                f"key={self.camera_grasp_poses_key}, index={selected['index'] + 1}"
            )
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 选中候选抓取位姿坐标: "
                f"camera=<missing>, "
                f"base_link={format_xyz_ypr(base_grasp_target)}, "
                f"waist_yaw_link={format_xyz_ypr(waist_grasp_target)}"
            )
            return

        camera_grasp_target = matrix_to_xyz_ypr(camera_grasp_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选中候选抓取位姿坐标: "
            f"camera={format_xyz_ypr(camera_grasp_target)}, "
            f"base_link={format_xyz_ypr(base_grasp_target)}, "
            f"waist_yaw_link={format_xyz_ypr(waist_grasp_target)}"
        )

    def refresh_selected_grasp_target(self):
        """底盘/腰部运动后按真实 TF 复核抓取手和最终目标。"""
        self._ensure_runtime_context()
        if not self.blackboard.exists(self.selected_rotate_z_180_key):
            raise RuntimeError(f"选中的抓取翻转标记不存在: key={self.selected_rotate_z_180_key}")
        if not self.blackboard.exists(self.selected_arm_side_key):
            raise RuntimeError(f"选中的手臂侧别不存在: key={self.selected_arm_side_key}")

        if self.blackboard.exists(self.selected_map_grasp_pose_key) and self.blackboard.get(self.selected_map_grasp_pose_key) is not None:
            map_grasp_pose = self.blackboard.get(self.selected_map_grasp_pose_key)
            base_from_map = self._lookup_base_from_map_via_chassis()
            grasp_pose = base_from_map @ map_grasp_pose
            self.blackboard.set(self.selected_base_grasp_pose_key, grasp_pose, overwrite=True)
        elif self.blackboard.exists(self.selected_base_grasp_pose_key):
            grasp_pose = self.blackboard.get(self.selected_base_grasp_pose_key)
        else:
            raise RuntimeError(
                f"选中的抓取矩阵不存在: map_key={self.selected_map_grasp_pose_key}, "
                f"base_key={self.selected_base_grasp_pose_key}"
            )
        rotate_z_180 = bool(self.blackboard.get(self.selected_rotate_z_180_key))
        arm_side = str(self.blackboard.get(self.selected_arm_side_key)).strip().lower()
        waist_from_base = self._lookup_transform_matrix(WAIST_YAW_LINK_FRAME, BASE_LINK_FRAME)
        arm_targets = {}
        for current_side in ("left", "right"):
            target = self._arm_controller.get_current_end_effector_pose(
                current_side, target_frame=WAIST_YAW_LINK_FRAME
            )
            if target is None or len(target) != 6:
                raise RuntimeError(f"运动后无法获取{current_side}臂当前末端位姿")
            arm_targets[current_side] = list(target)

        # 关键步骤：目标物仍固定在 base_link，下发前按当前腰部 TF 重投影到 waist_yaw_link。
        grasp_target, grasp_target_pose = build_grasp_target(
            grasp_pose,
            waist_from_base,
            offset_m=self.grasp_offset_m,
            rotate_z_180=rotate_z_180,
        )
        base_grasp_target_pose = np.linalg.inv(waist_from_base) @ grasp_target_pose
        base_grasp_target = matrix_to_xyz_ypr(base_grasp_target_pose)
        # 关键步骤：先打印重算目标，再做 IK-FK 检查，失败时也能定位目标坐标是否异常。
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 计算预抓取前最终抓取目标 pose: "
            f"base_link={format_xyz_ypr(base_grasp_target)}, "
            f"waist_yaw_link={format_xyz_ypr(grasp_target)}"
        )
        grasp_valid, _, pos_error, angle_error = self._check_arm_pose(
            arm_side,
            grasp_target_pose,
            pos_threshold=0.01,
            angle_threshold=0.05,
        )
        pregrasp_result = self._find_valid_pregrasp(arm_side, grasp_target_pose)
        if not grasp_valid or pregrasp_result is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 腰部运动后原抓取候选复核失败，"
                f"开始在当前物体的双手候选中重选: "
                f"pos={pos_error:.4f}m/0.0100m, angle={angle_error:.4f}rad/0.0500rad"
            )
            selected = self._reselect_current_object_grasp_target_after_torso_move(
                arm_targets
            )
            if selected is not None:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 腰部运动后已重选当前物体候选: "
                    f"第 {selected['index'] + 1} 个{selected['pose_label']}，"
                    f"grasp={selected['grasp_target']}"
                )
                return selected["grasp_target"]
            raise RuntimeError(
                "腰部运动后重算抓取位姿未通过 IK-FK 检查: "
                f"pos={pos_error:.4f}m/0.0100m, angle={angle_error:.4f}rad/0.0500rad"
            )

        pregrasp_pose, pregrasp_target, pregrasp_offset_m = pregrasp_result
        self._publish_grasp_pose(grasp_target_pose, waist_from_base)
        self.blackboard.set(self.left_pose_key, arm_targets["left"], overwrite=True)
        self.blackboard.set(self.grasp_pose_key, list(grasp_target), overwrite=True)
        self.blackboard.set(self.pregrasp_pose_key, list(pregrasp_target), overwrite=True)
        self.blackboard.set(self.selected_grasp_pose_key, grasp_target_pose, overwrite=True)
        self.blackboard.set(self.selected_waist_from_base_key, waist_from_base, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 运动后已按当前 TF 刷新{arm_side}臂抓取目标: "
            f"grasp={grasp_target}, pregrasp_offset={pregrasp_offset_m:.3f}m"
        )
        return grasp_target

    def _reselect_current_object_grasp_target_after_torso_move(self, arm_targets):
        """实际到位后，在当前物体的双手候选中重选。"""
        try:
            grasp_poses, poses_frame = self._read_grasp_poses()
            fixed_knee_from_base = self._lookup_fixed_knee_from_base()
            base_from_waist = self._lookup_transform_matrix(
                BASE_LINK_FRAME,
                WAIST_YAW_LINK_FRAME,
            )
            base_from_map = self._lookup_base_from_map_via_chassis()
            # 关键步骤：用当前真实腰部 TF 构造 sample，只重选当前物体候选，不推进物体游标。
            current_fixed_knee_from_waist = fixed_knee_from_base @ base_from_waist
            current_sample = self._current_torso_sample(current_fixed_knee_from_waist)
            current_sample["base_from_map"] = base_from_map
            selected = self._find_valid_grasp_target_for_torso_sample(
                current_sample,
                grasp_poses,
                poses_frame,
                fixed_knee_from_base,
                arm_targets,
            )
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 腰部运动后重选当前物体候选异常: {exc}"
            )
            return None

        if selected is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 腰部运动后当前物体没有可达候选"
            )
            return None

        # 关键步骤：复用统一写入函数，刷新最终抓取目标和 selected_* blackboard。
        self._write_grasp_targets(selected, preserve_motion_flags=True)
        return selected

    def compute_pregrasp_target(self, pregrasp_pose_key):
        """根据最终抓取位姿计算预抓取目标。"""
        self._ensure_runtime_context()
        if not self.blackboard.exists(self.selected_grasp_pose_key):
            raise RuntimeError(f"最终抓取矩阵不存在: key={self.selected_grasp_pose_key}")
        if not self.blackboard.exists(self.selected_arm_side_key):
            raise RuntimeError(f"选中手臂不存在: key={self.selected_arm_side_key}")
        grasp_target_pose = self.blackboard.get(self.selected_grasp_pose_key)
        arm_side = str(self.blackboard.get(self.selected_arm_side_key)).strip().lower()
        result = self._find_valid_pregrasp(arm_side, grasp_target_pose)
        if result is None:
            raise RuntimeError(f"{arm_side}臂所有预抓取 offset 均未通过 IK-FK 检查")
        pregrasp_pose, pregrasp_target, pregrasp_offset_m = result
        # 关键步骤：执行前用当前选手结果再次写入预抓取目标。
        self._publish_target_pose(pregrasp_pose)
        self.blackboard.set(pregrasp_pose_key, list(pregrasp_target), overwrite=True)
        self.blackboard.set(self.pregrasp_pose_key, list(pregrasp_target), overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算{arm_side}臂预抓取位姿: "
            f"offset={pregrasp_offset_m:.3f}m, key={pregrasp_pose_key}"
        )
        return pregrasp_target

    def _find_valid_pregrasp(self, arm_side, grasp_target_pose):
        """按稳定 offset 顺序返回选中手臂第一个可达预抓取位姿。"""
        for pregrasp_offset_m in _ordered_pregrasp_offsets(
            self.pregrasp_offset_min_m,
            self.pregrasp_offset_max_m,
            self.pregrasp_offset_samples,
        ):
            pregrasp_pose = _build_pregrasp_pose(
                grasp_target_pose, self.grasp_offset_m, pregrasp_offset_m
            )
            try:
                pregrasp_valid, _, _, _ = self._check_arm_pose(
                    arm_side,
                    pregrasp_pose,
                    pos_threshold=0.03,
                    angle_threshold=0.50,
                )
            except Exception:
                continue
            if pregrasp_valid:
                return pregrasp_pose, matrix_to_xyz_ypr(pregrasp_pose), pregrasp_offset_m
        return None

    def _lookup_transform_matrix(self, target_frame, source_frame):
        return lookup_transform_matrix(
            self._tf_listener,
            self.ros_node,
            target_frame,
            source_frame,
            timeout=self.tf_timeout_sec,
        )

    def _lookup_base_from_map_via_chassis(self):
        return lookup_base_from_map_via_chassis(
            self._tf_listener,
            self.ros_node,
            map_frame=MAP_FRAME,
            chassis_frame=self.chassis_frame,
            timeout=self.tf_timeout_sec,
        )

    def _lookup_fixed_knee_from_base(self):
        """构造 fixed_knee <- base_link 虚拟变换。"""
        knee_from_base = self._lookup_transform_matrix(KNEE_LINK_FRAME, BASE_LINK_FRAME)
        fixed_knee_from_base = np.array(knee_from_base, copy=True)
        # 关键步骤：fixed_knee 只复用 knee_link 的位置，旋转保持与 base_link 对齐。
        fixed_knee_from_base[:3, :3] = np.eye(3)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 使用虚拟 {FIXED_KNEE_FRAME}: "
            f"position_from_{BASE_LINK_FRAME}="
            f"[x={fixed_knee_from_base[0, 3]:.4f}, "
            f"y={fixed_knee_from_base[1, 3]:.4f}, "
            f"z={fixed_knee_from_base[2, 3]:.4f}], "
            f"rotation=identity"
        )
        return fixed_knee_from_base

    def _publish_target_pose(self, target_pose):
        quaternion = self._tf.transformations.quaternion_from_matrix(target_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = WAIST_YAW_LINK_FRAME
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
        message.header.frame_id = BASE_LINK_FRAME
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
        message.header.frame_id = BASE_LINK_FRAME
        message.pose.position.x = float(grasp_pose[0, 3])
        message.pose.position.y = float(grasp_pose[1, 3])
        message.pose.position.z = float(grasp_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.raw_grasp_pose_pub.publish(message)
