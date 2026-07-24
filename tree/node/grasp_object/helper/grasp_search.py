"""grasp_object 抓取候选搜索和 IK 检查。"""

import contextlib
import io
import time
import traceback
from concurrent.futures import as_completed

import numpy as np
import tf.transformations as tf_trans
from kuavo_humanoid_sdk.common.constants import (
    ARM_IK_SOLVER,
    ARM_IK_SOLVER_LEGACY,
    ARM_IK_SOLVER_NEW,
)
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import IKAnalytical
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.new_ik import NewIKAnalytical

from tree.constants import BASE_LINK_FRAME, MAP_FRAME
from tree.utils.geometry import matrix_to_xyz_ypr
from .grasp_math import build_grasp_target


_CPP_SEARCH_UNAVAILABLE = object()
_ARM_SEARCH_ORDER = ("right", "left")
_ARM_EEF_FRAMES = {
    "right": "zarm_r7_link",
    "left": "zarm_l7_link",
}
_NEW_IK_REFERENCE_JOINTS = np.zeros(7, dtype=float)


def _pose_matrix_from_pos_quat(eef_pos, eef_quat_xyzw):
    """把位置和四元数转成 waist_yaw_link 语义下的 4x4 目标矩阵。"""
    quat_xyzw = np.asarray(eef_quat_xyzw, dtype=float)
    quat_norm = np.linalg.norm(quat_xyzw)
    if quat_norm < 1e-8:
        raise ValueError("pose quaternion is invalid")
    target_pose = tf_trans.quaternion_matrix(quat_xyzw / quat_norm)
    target_pose[:3, 3] = np.asarray(eef_pos, dtype=float)
    return target_pose


def _new_ik_check_pose_validity(eef_pos, eef_quat_xyzw, eef_frame, pos_threshold, angle_threshold):
    """使用 NewIKAnalytical 做单点 IK-FK 回代检查。"""
    target_pose = _pose_matrix_from_pos_quat(eef_pos, eef_quat_xyzw)
    if eef_frame == _ARM_EEF_FRAMES["left"]:
        with contextlib.redirect_stdout(io.StringIO()):
            ik_valid, joint_angles = NewIKAnalytical.left_ik(
                _NEW_IK_REFERENCE_JOINTS,
                target_pose,
            )
        fk_pose = NewIKAnalytical.left_fk(joint_angles)
    elif eef_frame == _ARM_EEF_FRAMES["right"]:
        with contextlib.redirect_stdout(io.StringIO()):
            ik_valid, joint_angles = NewIKAnalytical.right_ik(
                _NEW_IK_REFERENCE_JOINTS,
                target_pose,
            )
        fk_pose = NewIKAnalytical.right_fk(joint_angles)
    else:
        raise ValueError(f"unsupported eef_frame: {eef_frame}")

    joint_angles = np.asarray(joint_angles, dtype=float)
    if not ik_valid or joint_angles.shape != (7,) or not np.all(np.isfinite(joint_angles)):
        return False, joint_angles, float("inf"), float("inf")

    fk_quat_xyzw = tf_trans.quaternion_from_matrix(fk_pose)
    target_quat_xyzw = np.asarray(eef_quat_xyzw, dtype=float)
    target_quat_xyzw = target_quat_xyzw / np.linalg.norm(target_quat_xyzw)
    pos_error = float(np.linalg.norm(fk_pose[:3, 3] - np.asarray(eef_pos, dtype=float)))
    quat_dot = abs(float(np.dot(target_quat_xyzw, fk_quat_xyzw)))
    angle_error = float(2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0)))
    return (
        pos_error <= pos_threshold and angle_error <= angle_threshold,
        joint_angles,
        pos_error,
        angle_error,
    )


def _check_pose_validity_with_selected_ik(
    eef_pos,
    eef_quat_xyzw,
    eef_frame,
    model_type,
    pos_threshold,
    angle_threshold,
):
    """按 SDK 全局常量选择 legacy_ik 或 new_ik 做单点 IK-FK 检查。"""
    if ARM_IK_SOLVER == ARM_IK_SOLVER_NEW:
        return _new_ik_check_pose_validity(
            eef_pos=eef_pos,
            eef_quat_xyzw=eef_quat_xyzw,
            eef_frame=eef_frame,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )
    if ARM_IK_SOLVER == ARM_IK_SOLVER_LEGACY:
        return IKAnalytical.check_pose_validity(
            eef_pos=eef_pos,
            eef_quat_xyzw=eef_quat_xyzw,
            eef_frame=eef_frame,
            model_type=model_type,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )
    raise ValueError(
        f"不支持的 ARM_IK_SOLVER={ARM_IK_SOLVER!r}，"
        f"可选值为 {ARM_IK_SOLVER_NEW!r} 或 {ARM_IK_SOLVER_LEGACY!r}"
    )


def _check_pose_validity_batch_with_selected_ik(
    eef_pos_array,
    eef_quat_xyzw_array,
    eef_frame,
    model_type,
    pos_threshold,
    angle_threshold,
):
    """按 SDK 全局常量选择 IK，并返回和旧 batch 接口一致的四元组。"""
    if ARM_IK_SOLVER == ARM_IK_SOLVER_LEGACY:
        return IKAnalytical.check_pose_validity_batch(
            eef_pos_array=eef_pos_array,
            eef_quat_xyzw_array=eef_quat_xyzw_array,
            eef_frame=eef_frame,
            model_type=model_type,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )

    eef_pos_array = np.asarray(eef_pos_array, dtype=float)
    eef_quat_xyzw_array = np.asarray(eef_quat_xyzw_array, dtype=float)
    valid_mask = np.zeros(eef_pos_array.shape[0], dtype=bool)
    joint_angles_array = np.full((eef_pos_array.shape[0], 7), np.nan, dtype=float)
    pos_errors = np.full(eef_pos_array.shape[0], np.inf, dtype=float)
    angle_errors = np.full(eef_pos_array.shape[0], np.inf, dtype=float)
    for index, (eef_pos, eef_quat_xyzw) in enumerate(zip(eef_pos_array, eef_quat_xyzw_array)):
        valid, joint_angles, pos_error, angle_error = _check_pose_validity_with_selected_ik(
            eef_pos=eef_pos,
            eef_quat_xyzw=eef_quat_xyzw,
            eef_frame=eef_frame,
            model_type=model_type,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )
        valid_mask[index] = valid
        joint_angles_array[index] = joint_angles
        pos_errors[index] = pos_error
        angle_errors[index] = angle_error
    return valid_mask, joint_angles_array, pos_errors, angle_errors


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
        allowed_arm_side,
    ) = args
    waist_from_base = np.linalg.inv(sample["fixed_knee_from_waist"]) @ fixed_knee_from_base
    base_grasp_poses = _project_grasp_poses_to_sample_base(grasp_poses, poses_frame, sample)
    arm_search_order = (
        (allowed_arm_side,)
        if allowed_arm_side in _ARM_SEARCH_ORDER
        else _ARM_SEARCH_ORDER
    )
    for arm_side in arm_search_order:
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
            valid_mask, _, _, _ = _check_pose_validity_batch_with_selected_ik(
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
                    pregrasp_valid, _, _, _ = _check_pose_validity_with_selected_ik(
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



class GraspSearch:
    """封装 C++/Python 抓取搜索和预抓取 IK 复核。"""

    def _find_valid_grasp_target_cpp(
        self,
        samples,
        grasp_poses,
        poses_frame,
        fixed_knee_from_base,
        arm_targets,
        fallback_label,
        allowed_arm_side=None,
    ):
        """调用 C++ 内核搜索可达抓取目标，失败时按配置回退 Python。"""
        if not samples:
            return None
        try:
            grasp_search_cpp = self._load_grasp_search_cpp_module()

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
            result = grasp_search_cpp.find_grasp_target(
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
                allowed_arm_side=allowed_arm_side or self._resolve_allowed_arm_side(),
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
            "sample_order": sample_index,
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

    def _load_grasp_search_cpp_module(self):
        """按 ARM_IK_SOLVER 选择抓取搜索 C++ 扩展。"""
        if ARM_IK_SOLVER == ARM_IK_SOLVER_NEW:
            from .. import _grasp_search_new_ik_cpp as grasp_search_cpp

            module_name = "_grasp_search_new_ik_cpp"
        elif ARM_IK_SOLVER == ARM_IK_SOLVER_LEGACY:
            from .. import _grasp_search_cpp as grasp_search_cpp

            module_name = "_grasp_search_cpp"
        else:
            raise ValueError(
                f"不支持的 ARM_IK_SOLVER={ARM_IK_SOLVER!r}，"
                f"可选值为 {ARM_IK_SOLVER_NEW!r} 或 {ARM_IK_SOLVER_LEGACY!r}"
            )

        if self._logged_cpp_search_module_name != module_name:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 抓取搜索使用 IK solver={ARM_IK_SOLVER}, "
                f"C++ module={module_name}"
            )
            self._logged_cpp_search_module_name = module_name
        return grasp_search_cpp

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
                    self._resolve_allowed_arm_side(),
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

    def _find_valid_grasp_target_for_torso_sample(
        self,
        sample,
        grasp_poses,
        poses_frame,
        fixed_knee_from_base,
        arm_targets,
        allowed_arm_side=None,
    ):
        # 关键步骤：map 抓取位姿按 sample 的假设底盘位姿投影到 base_link，再做腰部采样检查。
        waist_from_base = np.linalg.inv(sample["fixed_knee_from_waist"]) @ fixed_knee_from_base
        base_grasp_poses = _project_grasp_poses_to_sample_base(grasp_poses, poses_frame, sample)
        pregrasp_offsets_m = _ordered_pregrasp_offsets(
            self.pregrasp_offset_min_m,
            self.pregrasp_offset_max_m,
            self.pregrasp_offset_samples,
        )
        # 关键步骤：分阶段抓取时可锁定单侧手臂，both 仍保持右手优先。
        allowed_arm_side = allowed_arm_side or self._resolve_allowed_arm_side()
        arm_search_order = (
            (allowed_arm_side,)
            if allowed_arm_side in _ARM_SEARCH_ORDER
            else _ARM_SEARCH_ORDER
        )
        for arm_side in arm_search_order:
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

    def _find_all_valid_grasp_targets_for_torso_sample(
        self,
        sample,
        grasp_poses,
        poses_frame,
        fixed_knee_from_base,
        arm_targets,
        allowed_arm_side,
    ):
        """枚举单个 sample 下指定手臂的全部可达抓取候选。"""
        if allowed_arm_side not in _ARM_SEARCH_ORDER:
            raise ValueError(f"allowed_arm_side 必须是 left 或 right: {allowed_arm_side!r}")

        # 关键步骤：复核重选只在当前真实腰部 sample 内执行，需要保留同一物体的所有候选。
        valid_targets = []
        waist_from_base = np.linalg.inv(sample["fixed_knee_from_waist"]) @ fixed_knee_from_base
        base_grasp_poses = _project_grasp_poses_to_sample_base(grasp_poses, poses_frame, sample)
        for index, grasp_pose in enumerate(base_grasp_poses):
            for rotate_z_180 in (False, True):
                grasp_target, grasp_target_pose = build_grasp_target(
                    grasp_pose,
                    waist_from_base,
                    offset_m=self.grasp_offset_m,
                    rotate_z_180=rotate_z_180,
                )
                try:
                    grasp_valid, _, _, _ = self._check_arm_pose(
                        allowed_arm_side,
                        grasp_target_pose,
                        pos_threshold=0.01,
                        angle_threshold=0.05,
                    )
                except Exception:
                    continue
                if not grasp_valid:
                    continue

                pregrasp_result = self._find_valid_pregrasp(allowed_arm_side, grasp_target_pose)
                if pregrasp_result is None:
                    continue
                pregrasp_pose, pregrasp_target, pregrasp_offset_m = pregrasp_result
                valid_targets.append(
                    {
                        "sample": sample,
                        "arm_side": allowed_arm_side,
                        "index": index,
                        "pose_label": "z轴旋转180度后" if rotate_z_180 else "原始",
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
                        "pregrasp_target": pregrasp_target,
                        "pregrasp_offset_m": pregrasp_offset_m,
                        "waist_from_base": waist_from_base,
                    }
                )
        return valid_targets

    def _check_arm_pose(self, arm_side, target_pose, pos_threshold, angle_threshold):
        """按左右手选择 SDK 末端 frame 执行 IK-FK 检查。"""
        return _check_pose_validity_with_selected_ik(
            eef_pos=target_pose[:3, 3],
            eef_quat_xyzw=self._tf.transformations.quaternion_from_matrix(target_pose),
            eef_frame=_ARM_EEF_FRAMES[arm_side],
            model_type=self.model_type,
            pos_threshold=pos_threshold,
            angle_threshold=angle_threshold,
        )

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
