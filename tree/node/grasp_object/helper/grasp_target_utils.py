"""双臂抓取目标计算工具，供多个行为树节点复用。"""

import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import py_trees
from geometry_msgs.msg import PoseStamped

from tree.constants import (
    BASE_LINK_FRAME,
    CHASSIS_FRAME,
    MAP_FRAME,
    MODEL_TYPE_KEY,
    ODOM_POSE_TRANSFORMER_KEY,
    ROBOT_SERVICES_KEY,
    WAIST_YAW_LINK_FRAME,
)
from tree.utils.geometry import (
    format_xyz_ypr,
    matrix_to_xyz_ypr,
)
from .grasp_math import build_grasp_target
from .grasp_config import (
    register_blackboard_keys,
    register_blackboard_read_write_keys,
    require_non_empty,
)
from .grasp_context import GraspContext
from .grasp_object_store import GraspObjectStore
from .grasp_sampling import GraspSampling
from .grasp_search import _CPP_SEARCH_UNAVAILABLE, GraspSearch
from .grasp_target_writer import GraspTargetWriter


class GraspTargetComputer(
    GraspContext,
    GraspSampling,
    GraspSearch,
    GraspObjectStore,
    GraspTargetWriter,
):
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
        self.odom_topic = str(params.get("odom_topic", self.chassis_frame)).strip()
        self.odom_history_duration_sec = float(params.get("odom_history_duration_sec", 10.0))
        self._odom_transformer = None
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
        self.torso_sample_z_min_key = str(
            params.get("torso_sample_z_min_key", "grasp_object_torso_sample_z_min_m")
        ).strip()
        self.torso_sample_z_max_key = str(
            params.get("torso_sample_z_max_key", "grasp_object_torso_sample_z_max_m")
        ).strip()
        self.torso_sample_step_m = float(params.get("torso_sample_step_m", 0.05))
        self.torso_pitch_abs_max_rad = float(
            params.get("torso_pitch_abs_max_rad", math.radians(15.0))
        )
        self.torso_pitch_sample_min_deg = float(params.get("torso_pitch_sample_min_deg", 0.0))
        self.torso_pitch_sample_max_deg = float(params.get("torso_pitch_sample_max_deg", 15.0))
        self.torso_pitch_sample_step_deg = float(params.get("torso_pitch_sample_step_deg", 3.0))
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
        self._logged_cpp_search_module_name = None
        self.preserve_motion_flags = self._to_bool(
            params.get("preserve_motion_flags", False)
        )
        self.allowed_arm_side = str(params.get("allowed_arm_side", "both")).strip().lower()
        self.allowed_arm_side_key = str(params.get("allowed_arm_side_key", "")).strip()
        self.search_remaining_objects = self._to_bool(
            params.get("search_remaining_objects", False)
        )
        self.search_remaining_objects_key = str(
            params.get("search_remaining_objects_key", "")
        ).strip()
        self.sorted_grasp_objects_key = str(
            params.get("sorted_grasp_objects_key", "grasp_object_sorted_grasp_objects")
        ).strip()
        self.next_grasp_object_index_key = str(
            params.get("next_grasp_object_index_key", "grasp_object_next_grasp_object_index")
        ).strip()
        self.selected_grasp_object_info_key = str(
            params.get(
                "selected_grasp_object_info_key",
                "grasp_object_selected_grasp_object_info",
            )
        ).strip()
        self.selected_map_grasp_pose_key = str(
            params.get("selected_map_grasp_pose_key", "grasp_object_selected_map_grasp_pose")
        ).strip()
        self.left_grasp_pose_key = str(
            params.get("left_grasp_pose_key", "grasp_object_left_grasp_pose")
        ).strip()
        self.right_grasp_pose_key = str(
            params.get("right_grasp_pose_key", "grasp_object_right_grasp_pose")
        ).strip()
        self.left_pregrasp_pose_key = str(
            params.get("left_pregrasp_pose_key", "grasp_object_left_pregrasp_pose")
        ).strip()
        self.right_pregrasp_pose_key = str(
            params.get("right_pregrasp_pose_key", "grasp_object_right_pregrasp_pose")
        ).strip()
        self.left_selected_grasp_pose_key = str(
            params.get("left_selected_grasp_pose_key", "grasp_object_left_selected_grasp_pose")
        ).strip()
        self.right_selected_grasp_pose_key = str(
            params.get("right_selected_grasp_pose_key", "grasp_object_right_selected_grasp_pose")
        ).strip()
        self.left_selected_base_grasp_pose_key = str(
            params.get(
                "left_selected_base_grasp_pose_key",
                "grasp_object_left_selected_base_grasp_pose",
            )
        ).strip()
        self.right_selected_base_grasp_pose_key = str(
            params.get(
                "right_selected_base_grasp_pose_key",
                "grasp_object_right_selected_base_grasp_pose",
            )
        ).strip()
        self.left_selected_map_grasp_pose_key = str(
            params.get(
                "left_selected_map_grasp_pose_key",
                "grasp_object_left_selected_map_grasp_pose",
            )
        ).strip()
        self.right_selected_map_grasp_pose_key = str(
            params.get(
                "right_selected_map_grasp_pose_key",
                "grasp_object_right_selected_map_grasp_pose",
            )
        ).strip()
        self.left_selected_rotate_z_180_key = str(
            params.get(
                "left_selected_rotate_z_180_key",
                "grasp_object_left_selected_rotate_z_180",
            )
        ).strip()
        self.right_selected_rotate_z_180_key = str(
            params.get(
                "right_selected_rotate_z_180_key",
                "grasp_object_right_selected_rotate_z_180",
            )
        ).strip()
        self.dual_selected_waist_from_base_key = str(
            params.get(
                "dual_selected_waist_from_base_key",
                "grasp_object_dual_selected_waist_from_base",
            )
        ).strip()
        self.dual_selected_pair_info_key = str(
            params.get(
                "dual_selected_pair_info_key",
                "grasp_object_dual_selected_pair_info",
            )
        ).strip()
        self.knee_origin_x = float(params.get("knee_origin_x", 0.098))
        self.knee_origin_z = float(params.get("knee_origin_z", 0.376))
        register_blackboard_keys(
            self.blackboard,
            (self.services_key, self.model_type_key),
            py_trees.common.Access.READ,
        )
        read_keys = []
        if self.torso_sample_z_min_key:
            read_keys.append(self.torso_sample_z_min_key)
        if self.torso_sample_z_max_key:
            read_keys.append(self.torso_sample_z_max_key)
        if read_keys:
            register_blackboard_keys(
                self.blackboard,
                read_keys,
                py_trees.common.Access.READ,
            )
        if self.allowed_arm_side not in ("both", "left", "right"):
            raise ValueError("allowed_arm_side 仅支持 both、left 或 right")
        if self.allowed_arm_side_key:
            register_blackboard_keys(
                self.blackboard,
                (self.allowed_arm_side_key,),
                py_trees.common.Access.READ,
            )
        if self.search_remaining_objects_key:
            register_blackboard_keys(
                self.blackboard,
                (self.search_remaining_objects_key,),
                py_trees.common.Access.READ,
            )
        register_blackboard_read_write_keys(
            self.blackboard,
            (
                self.sorted_grasp_objects_key,
                self.next_grasp_object_index_key,
                self.selected_grasp_object_info_key,
            ),
            py_trees.common.Access.READ,
            py_trees.common.Access.WRITE,
        )
        require_non_empty(self.base_grasp_poses_key, "base_grasp_poses_key 不能为空")
        require_non_empty(self.camera_grasp_poses_key, "camera_grasp_poses_key 不能为空")
        require_non_empty(self.map_grasp_poses_key, "map_grasp_poses_key 不能为空")
        require_non_empty(self.odom_topic, "odom_topic 不能为空")
        self.blackboard.register_key(
            key=ODOM_POSE_TRANSFORMER_KEY,
            access=py_trees.common.Access.READ,
        )
        register_blackboard_read_write_keys(
            self.blackboard,
            (
                self.base_grasp_poses_key,
                self.camera_grasp_poses_key,
                self.map_grasp_poses_key,
            ),
            py_trees.common.Access.READ,
            py_trees.common.Access.WRITE,
        )
        write_keys = [
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
            self.left_grasp_pose_key,
            self.right_grasp_pose_key,
            self.left_pregrasp_pose_key,
            self.right_pregrasp_pose_key,
            self.left_selected_grasp_pose_key,
            self.right_selected_grasp_pose_key,
            self.left_selected_base_grasp_pose_key,
            self.right_selected_base_grasp_pose_key,
            self.left_selected_map_grasp_pose_key,
            self.right_selected_map_grasp_pose_key,
            self.left_selected_rotate_z_180_key,
            self.right_selected_rotate_z_180_key,
            self.dual_selected_waist_from_base_key,
            self.dual_selected_pair_info_key,
        ]
        for key in write_keys:
            if not key:
                raise ValueError("blackboard pose key 不能为空")
        register_blackboard_keys(
            self.blackboard,
            write_keys,
            py_trees.common.Access.WRITE,
        )
        require_non_empty(self.torso_enabled_key, "torso_enabled_key 不能为空")
        require_non_empty(
            self.chassis_lateral_enabled_key,
            "chassis_lateral_enabled_key 不能为空",
        )
        register_blackboard_keys(
            self.blackboard,
            (self.torso_enabled_key, self.chassis_lateral_enabled_key),
            py_trees.common.Access.WRITE,
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
        if self.torso_pitch_sample_step_deg <= 0.0:
            raise ValueError("torso_pitch_sample_step_deg 必须大于 0")
        if self.torso_pitch_sample_max_deg < self.torso_pitch_sample_min_deg:
            raise ValueError("torso_pitch_sample_max_deg 必须大于等于 torso_pitch_sample_min_deg")
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

    def compute_current_torso_target(self):
        """仅使用当前腰部位姿计算可达抓取目标。"""
        fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map = self.prepare_context()
        current_sample = self._current_torso_sample(current_fixed_knee_from_waist)
        current_sample["base_from_map"] = base_from_map
        if self._should_search_remaining_objects():
            selected = self._find_target_from_remaining_objects(
                [current_sample], fixed_knee_from_base, arm_targets, "当前腰部"
            )
            if selected is None:
                raise RuntimeError("剩余物体在当前腰部下均无可达抓取目标")
            self._write_grasp_targets(selected)
            return selected

        grasp_poses, poses_frame = self._read_grasp_poses()
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
        fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map = self.prepare_context()
        current_torso_pose = self._get_current_torso_pose()
        check_sample_count = 0
        check_total_sec = 0.0
        selected = None
        torso_samples = [self._current_torso_sample(current_fixed_knee_from_waist)]
        torso_samples.extend(self._torso_ik_samples(current_fixed_knee_from_waist, current_torso_pose))
        poses_frame = MAP_FRAME if self._should_search_remaining_objects() else self._read_grasp_poses()[1]
        samples = self._chassis_lateral_torso_samples(torso_samples, base_from_map, poses_frame)
        if self._should_search_remaining_objects():
            selected = self._find_target_from_remaining_objects(
                samples, fixed_knee_from_base, arm_targets, "腰部采样"
            )
            if selected is None:
                raise RuntimeError("剩余物体在所有底盘/腰部采样下均无可达目标")
            self._write_grasp_targets(selected)
            return selected

        grasp_poses, poses_frame = self._read_grasp_poses()
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

    def compute_current_torso_dual_targets(self):
        """仅使用当前腰部位姿，优先寻找左右手同步抓取两个不同物体的目标。"""
        fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map = self.prepare_context()
        current_sample = self._current_torso_sample(current_fixed_knee_from_waist)
        current_sample["base_from_map"] = base_from_map
        selected = self._find_dual_targets_from_remaining_objects(
            [current_sample],
            fixed_knee_from_base,
            arm_targets,
            "当前腰部",
        )
        if selected is None:
            raise RuntimeError("当前腰部下未找到左右手同步抓取两个不同物体的目标")
        self._write_dual_grasp_targets(selected)
        return selected

    def compute_torso_sample_dual_targets(self):
        """通过底盘横移和腰部采样寻找左右手同步抓取两个不同物体的目标。"""
        fixed_knee_from_base, current_fixed_knee_from_waist, arm_targets, base_from_map = self.prepare_context()
        current_torso_pose = self._get_current_torso_pose()
        torso_samples = [self._current_torso_sample(current_fixed_knee_from_waist)]
        torso_samples.extend(self._torso_ik_samples(current_fixed_knee_from_waist, current_torso_pose))
        samples = self._chassis_lateral_torso_samples(torso_samples, base_from_map, MAP_FRAME)
        selected = self._find_dual_targets_from_remaining_objects(
            samples,
            fixed_knee_from_base,
            arm_targets,
            "腰部采样",
        )
        if selected is None:
            raise RuntimeError("所有底盘/腰部采样下均未找到左右手同步抓取两个不同物体的目标")
        self._write_dual_grasp_targets(selected)
        return selected

    @staticmethod
    def _to_bool(value):
        """解析 JSON 参数里的布尔值。"""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    def refresh_selected_grasp_target(self):
        """底盘/腰部运动后按真实 TF 复核抓取手和最终目标。"""
        self._ensure_runtime_context()
        if not self.blackboard.exists(self.selected_rotate_z_180_key):
            raise RuntimeError(f"选中的抓取翻转标记不存在: key={self.selected_rotate_z_180_key}")
        if not self.blackboard.exists(self.selected_arm_side_key):
            raise RuntimeError(f"选中的手臂侧别不存在: key={self.selected_arm_side_key}")

        if self.blackboard.exists(self.selected_map_grasp_pose_key) and self.blackboard.get(self.selected_map_grasp_pose_key) is not None:
            map_grasp_pose = self.blackboard.get(self.selected_map_grasp_pose_key)
            base_from_map = self._base_from_map_matrix_via_melon_odom()
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

    def refresh_dual_selected_grasp_targets(self):
        """底盘/腰部运动后按真实 TF 复核并刷新同步双抓目标。"""
        self._ensure_runtime_context()
        fixed_knee_from_base = self._lookup_fixed_knee_from_base()
        base_from_waist = self._lookup_transform_matrix(BASE_LINK_FRAME, WAIST_YAW_LINK_FRAME)
        base_from_map = self._base_from_map_matrix_via_melon_odom()
        current_fixed_knee_from_waist = fixed_knee_from_base @ base_from_waist
        current_sample = self._current_torso_sample(current_fixed_knee_from_waist)
        current_sample["base_from_map"] = base_from_map
        waist_from_base = self._lookup_transform_matrix(WAIST_YAW_LINK_FRAME, BASE_LINK_FRAME)
        arm_targets = self._read_current_arm_targets()

        try:
            selected = {
                "sample": current_sample,
                "sample_order": 0,
                "left": self._refresh_single_dual_side_target(
                    "left", waist_from_base, base_from_map, arm_targets
                ),
                "right": self._refresh_single_dual_side_target(
                    "right", waist_from_base, base_from_map, arm_targets
                ),
            }
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原候选复核异常，尝试当前位置重选: {exc}"
            )
            selected = None

        if selected is not None:
            left_ok = self._dual_side_target_still_valid("left", selected["left"])
            right_ok = self._dual_side_target_still_valid("right", selected["right"])
            if left_ok and right_ok:
                self._write_dual_grasp_targets(selected, preserve_motion_flags=True)
                return selected
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原候选复核失败，尝试原物体对重选: "
                f"left_ok={left_ok}, right_ok={right_ok}"
            )
            reselected = self._reselect_original_dual_object_pair_after_torso_move(
                current_sample,
                fixed_knee_from_base,
                arm_targets,
                selected,
                left_ok,
                right_ok,
            )
            if reselected is not None:
                self._write_dual_grasp_targets(reselected, preserve_motion_flags=True)
                return reselected

        self.ros_node.get_logger().warning(
            f"[{self.config_label}] 同步双抓原物体对重选失败，尝试当前位置全局重选"
        )
        selected = self._find_dual_targets_from_remaining_objects(
            [current_sample],
            fixed_knee_from_base,
            arm_targets,
            "移动后当前腰部",
        )
        if selected is None:
            raise RuntimeError("移动后当前位置无法重新找到同步双抓目标")
        self._write_dual_grasp_targets(selected, preserve_motion_flags=True)
        return selected

    def compute_dual_pregrasp_targets(self):
        """根据当前同步双抓最终目标重新计算左右预抓取目标。"""
        self._ensure_runtime_context()
        for arm_side, selected_pose_key, pregrasp_pose_key in (
            ("left", self.left_selected_grasp_pose_key, self.left_pregrasp_pose_key),
            ("right", self.right_selected_grasp_pose_key, self.right_pregrasp_pose_key),
        ):
            if not self.blackboard.exists(selected_pose_key):
                raise RuntimeError(f"{arm_side} 手最终抓取矩阵不存在: key={selected_pose_key}")
            grasp_target_pose = self.blackboard.get(selected_pose_key)
            result = self._find_valid_pregrasp(arm_side, grasp_target_pose)
            if result is None:
                raise RuntimeError(f"{arm_side} 手所有预抓取 offset 均未通过 IK-FK 检查")
            pregrasp_pose, pregrasp_target, pregrasp_offset_m = result
            self.blackboard.set(pregrasp_pose_key, list(pregrasp_target), overwrite=True)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已刷新同步双抓{arm_side}手预抓取位姿: "
                f"offset={pregrasp_offset_m:.3f}m, key={pregrasp_pose_key}"
            )

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
