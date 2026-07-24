"""grasp_object 物体缓存、游标和重选辅助。"""

from tree.constants import BASE_LINK_FRAME, MAP_FRAME
from .grasp_search import _CPP_SEARCH_UNAVAILABLE
from .grasp_math import build_grasp_target


class GraspObjectStore:
    """封装 sorted objects 和 selected object 的 blackboard 读写。"""

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

    def _resolve_allowed_arm_side(self):
        """解析当前阶段允许搜索的手臂侧别。"""
        allowed_arm_side = self.allowed_arm_side
        if self.allowed_arm_side_key:
            if not self.blackboard.exists(self.allowed_arm_side_key):
                # 关键步骤：未启用分阶段抓取的旧行为树不会写该 key，保持 both 兼容。
                return allowed_arm_side
            allowed_arm_side = str(self.blackboard.get(self.allowed_arm_side_key)).strip().lower()
        if allowed_arm_side not in ("both", "left", "right"):
            raise ValueError(f"搜索手臂侧别非法: {allowed_arm_side!r}")
        return allowed_arm_side

    def _should_search_remaining_objects(self):
        """解析是否跨所有剩余物体搜索本阶段目标。"""
        if not self.search_remaining_objects_key:
            return self.search_remaining_objects
        if not self.blackboard.exists(self.search_remaining_objects_key):
            return self.search_remaining_objects
        return self._to_bool(self.blackboard.get(self.search_remaining_objects_key))

    def _find_dual_targets_from_remaining_objects(
        self, samples, fixed_knee_from_base, arm_targets, fallback_label
    ):
        """从剩余物体中选择一对左右手可同步抓取的不同物体。"""
        if not self.blackboard.exists(self.sorted_grasp_objects_key):
            raise RuntimeError(f"物体缓存不存在: key={self.sorted_grasp_objects_key}")
        objects = self.blackboard.get(self.sorted_grasp_objects_key)
        if not isinstance(objects, list) or not objects:
            raise RuntimeError("物体缓存为空")
        next_index = 0
        if self.blackboard.exists(self.next_grasp_object_index_key):
            next_index = int(self.blackboard.get(self.next_grasp_object_index_key))
        if next_index < 0 or next_index >= len(objects):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓搜索跳过: "
                f"next_index={next_index}, object_count={len(objects)}, fallback={fallback_label}"
            )
            return None
        if len(objects) - next_index < 2:
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 同步双抓搜索跳过: 剩余物体不足两个, "
                f"next_index={next_index}, object_count={len(objects)}"
            )
            return None

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 同步双抓搜索开始: fallback={fallback_label}, "
            f"next_index={next_index}, object_count={len(objects)}, "
            f"remaining_count={len(objects) - next_index}, samples={len(samples)}"
        )
        for sample_index, sample in enumerate(samples):
            selected = self._find_dual_targets_for_sample(
                sample,
                sample_index,
                objects,
                next_index,
                fixed_knee_from_base,
                arm_targets,
                fallback_label,
            )
            if selected is not None:
                self._promote_dual_objects_to_cursor(
                    objects,
                    next_index,
                    selected["right"]["object_index"],
                    selected["left"]["object_index"],
                )
                return selected
        return None

    def _find_dual_targets_for_sample(
        self,
        sample,
        sample_index,
        objects,
        next_index,
        fixed_knee_from_base,
        arm_targets,
        fallback_label,
    ):
        """在单个 sample 下按确定顺序尝试不同物体的左右手组合。"""
        candidate_cache = {}
        for first_index in range(next_index, len(objects) - 1):
            for second_index in range(first_index + 1, len(objects)):
                assignments = (
                    ("right", first_index, "left", second_index),
                    ("left", first_index, "right", second_index),
                )
                for first_arm, first_object_index, second_arm, second_object_index in assignments:
                    first_target = self._find_object_target_for_arm(
                        sample,
                        sample_index,
                        objects,
                        first_object_index,
                        first_arm,
                        fixed_knee_from_base,
                        arm_targets,
                        candidate_cache,
                        fallback_label,
                    )
                    if first_target is None:
                        continue
                    second_target = self._find_object_target_for_arm(
                        sample,
                        sample_index,
                        objects,
                        second_object_index,
                        second_arm,
                        fixed_knee_from_base,
                        arm_targets,
                        candidate_cache,
                        fallback_label,
                    )
                    if second_target is None:
                        continue
                    selected = {
                        "sample": sample,
                        "sample_order": sample_index,
                        "left": first_target if first_arm == "left" else second_target,
                        "right": first_target if first_arm == "right" else second_target,
                    }
                    self.ros_node.get_logger().info(
                        f"[{self.config_label}] 同步双抓找到候选: "
                        f"sample_order={sample_index}, sample={sample.get('label', '')}, "
                        f"left_object={selected['left']['object_index']}, "
                        f"right_object={selected['right']['object_index']}"
                    )
                    return selected
        return None

    def _find_object_target_for_arm(
        self,
        sample,
        sample_index,
        objects,
        object_index,
        arm_side,
        fixed_knee_from_base,
        arm_targets,
        candidate_cache,
        fallback_label,
    ):
        """复用单臂抓取搜索，为指定物体和指定手臂找第一个可达候选。"""
        cache_key = (sample_index, object_index, arm_side)
        if cache_key in candidate_cache:
            return candidate_cache[cache_key]

        grasp_object = objects[object_index]
        grasp_poses = grasp_object.get("map_grasp_poses", [])
        if not isinstance(grasp_poses, list) or not grasp_poses:
            candidate_cache[cache_key] = None
            return None

        selected = self._find_valid_grasp_target_cpp(
            [sample],
            grasp_poses,
            MAP_FRAME,
            fixed_knee_from_base,
            arm_targets,
            fallback_label=f"{fallback_label}/dual/object={object_index}/{arm_side}",
            allowed_arm_side=arm_side,
        )
        if selected is _CPP_SEARCH_UNAVAILABLE:
            selected = self._find_valid_grasp_target_for_torso_sample(
                sample,
                grasp_poses,
                MAP_FRAME,
                fixed_knee_from_base,
                arm_targets,
                allowed_arm_side=arm_side,
            )
        if selected is not None:
            selected["object_index"] = object_index
            selected["object_id"] = grasp_object.get("object_id")
            selected["object_info"] = grasp_object
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 同步双抓单臂候选可达: "
                f"sample_order={sample_index}, object_index={object_index}, "
                f"object_id={grasp_object.get('object_id')}, arm_side={arm_side}, "
                f"grasp_index={int(selected.get('index', -1)) + 1}, "
                f"pose_label={selected.get('pose_label')}"
            )
        candidate_cache[cache_key] = selected
        return selected

    def _promote_dual_objects_to_cursor(self, objects, next_index, right_object_index, left_object_index):
        """把同步双抓选中的两个物体移动到游标前两位，便于闭爪成功后一次消费两个。"""
        selected_indices = {right_object_index, left_object_index}
        promoted = [
            objects[right_object_index],
            objects[left_object_index],
        ]
        remaining = [
            grasp_object
            for index, grasp_object in enumerate(objects[next_index:], start=next_index)
            if index not in selected_indices
        ]
        reordered = list(objects[:next_index]) + promoted + remaining
        self.blackboard.set(self.sorted_grasp_objects_key, reordered, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 同步双抓已调整物体缓存顺序: "
            f"cursor={next_index}, right_object_index={right_object_index}, "
            f"left_object_index={left_object_index}"
        )

    def _find_target_from_remaining_objects(
        self, samples, fixed_knee_from_base, arm_targets, fallback_label
    ):
        """遍历未消费物体，选择采样移动成本最小的可达目标。"""
        if not self.blackboard.exists(self.sorted_grasp_objects_key):
            raise RuntimeError(f"物体缓存不存在: key={self.sorted_grasp_objects_key}")
        objects = self.blackboard.get(self.sorted_grasp_objects_key)
        if not isinstance(objects, list) or not objects:
            raise RuntimeError("物体缓存为空")
        next_index = 0
        if self.blackboard.exists(self.next_grasp_object_index_key):
            next_index = int(self.blackboard.get(self.next_grasp_object_index_key))
        if next_index < 0 or next_index >= len(objects):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 跨剩余物体搜索跳过: "
                f"next_index={next_index}, object_count={len(objects)}, "
                f"fallback={fallback_label}"
            )
            return None

        allowed_arm_side = self._resolve_allowed_arm_side()
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 跨剩余物体搜索开始: "
            f"fallback={fallback_label}, allowed_arm_side={allowed_arm_side}, "
            f"next_index={next_index}, object_count={len(objects)}, "
            f"remaining_count={len(objects) - next_index}, samples={len(samples)}"
        )

        best = None
        best_key = None
        checked_count = 0
        for object_index in range(next_index, len(objects)):
            grasp_object = objects[object_index]
            grasp_poses = grasp_object.get("map_grasp_poses", [])
            if not isinstance(grasp_poses, list) or not grasp_poses:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 跨剩余物体搜索跳过空候选: "
                    f"object_index={object_index}, object_id={grasp_object.get('object_id')}, "
                    f"pose_count=0"
                )
                continue
            checked_count += 1
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 跨剩余物体搜索检查物体: "
                f"object_index={object_index}, object_id={grasp_object.get('object_id')}, "
                f"pose_count={len(grasp_poses)}, distance={float(grasp_object.get('distance_m', 0.0)):.4f}m, "
                f"allowed_arm_side={allowed_arm_side}, samples={len(samples)}"
            )
            selected = self._find_valid_grasp_target_cpp(
                samples,
                grasp_poses,
                MAP_FRAME,
                fixed_knee_from_base,
                arm_targets,
                fallback_label=f"{fallback_label}/object={object_index}",
            )
            if selected is _CPP_SEARCH_UNAVAILABLE:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 跨剩余物体搜索 C++ 不可用，回退 Python: "
                    f"object_index={object_index}, object_id={grasp_object.get('object_id')}"
                )
                selected = None
                for sample_index, sample in enumerate(samples):
                    candidate = self._find_valid_grasp_target_for_torso_sample(
                        sample,
                        grasp_poses,
                        MAP_FRAME,
                        fixed_knee_from_base,
                        arm_targets,
                    )
                    if candidate is not None:
                        selected = candidate
                        selected["sample_order"] = sample_index
                        break
            if selected is None:
                self.ros_node.get_logger().info(
                    f"[{self.config_label}] 跨剩余物体搜索未找到可达目标: "
                    f"object_index={object_index}, object_id={grasp_object.get('object_id')}, "
                    f"pose_count={len(grasp_poses)}, allowed_arm_side={allowed_arm_side}"
                )
                continue
            sample_order = int(selected.get("sample_order", 0))
            candidate_key = (sample_order, object_index)
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 跨剩余物体搜索找到候选: "
                f"object_index={object_index}, object_id={grasp_object.get('object_id')}, "
                f"sample_order={sample_order}, arm_side={selected.get('arm_side')}, "
                f"grasp_index={int(selected.get('index', -1)) + 1}, "
                f"pose_label={selected.get('pose_label')}"
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best = (object_index, grasp_object, selected)

        if best is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 跨剩余物体搜索失败: "
                f"fallback={fallback_label}, allowed_arm_side={allowed_arm_side}, "
                f"next_index={next_index}, object_count={len(objects)}, "
                f"checked_count={checked_count}, samples={len(samples)}"
            )
            return None

        object_index, grasp_object, selected = best
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 跨剩余物体搜索选中目标: "
            f"from_object_index={object_index}, target_cursor={next_index}, "
            f"object_id={grasp_object.get('object_id')}, arm_side={selected.get('arm_side')}, "
            f"sample_order={int(selected.get('sample_order', 0))}, "
            f"grasp_index={int(selected.get('index', -1)) + 1}"
        )
        # 关键步骤：把跨物体搜索选中项换到当前游标，不会跳过中间未抓物体。
        if object_index != next_index:
            objects = list(objects)
            objects[next_index], objects[object_index] = objects[object_index], objects[next_index]
            self.blackboard.set(self.sorted_grasp_objects_key, objects, overwrite=True)
            grasp_object = objects[next_index]
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 跨剩余物体搜索交换缓存顺序: "
                f"from_object_index={object_index}, target_cursor={next_index}, "
                f"object_id={grasp_object.get('object_id')}"
            )
        # 关键步骤：规划阶段不消费物体，只有夹爪成功闭合后才由行为树推进游标。
        self.blackboard.set(
            self.base_grasp_poses_key,
            grasp_object.get("base_grasp_poses", []),
            overwrite=True,
        )
        self.blackboard.set(
            self.camera_grasp_poses_key,
            grasp_object.get("camera_grasp_poses", []),
            overwrite=True,
        )
        self.blackboard.set(
            self.map_grasp_poses_key,
            grasp_object.get("map_grasp_poses", []),
            overwrite=True,
        )
        self.blackboard.set(
            self.selected_grasp_object_info_key,
            {
                "rank_index": next_index,
                "object_count": len(objects),
                "object_id": grasp_object.get("object_id"),
                "bbox": grasp_object.get("bbox"),
                "distance_m": float(grasp_object.get("distance_m", 0.0)),
                "pose_count": len(grasp_object.get("base_grasp_poses", [])),
            },
            overwrite=True,
        )
        return selected

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

    def _reselect_original_dual_object_pair_after_torso_move(
        self,
        current_sample,
        fixed_knee_from_base,
        arm_targets,
        refreshed_selected,
        left_ok,
        right_ok,
    ):
        """腰部到位后优先围绕原同步双抓物体对穷举重选。"""
        pair_info = self._read_dual_selected_pair_info()
        if pair_info is None:
            return None
        objects = self._read_sorted_grasp_objects_for_dual_refresh()
        if objects is None:
            return None

        left_object_index = self._find_original_dual_object_index(objects, pair_info, "left")
        right_object_index = self._find_original_dual_object_index(objects, pair_info, "right")
        if left_object_index is None or right_object_index is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原物体对重选跳过: "
                f"left_index={left_object_index}, right_index={right_object_index}, pair_info={pair_info}"
            )
            return None
        if left_object_index == right_object_index:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原物体对重选跳过: 左右物体索引相同 {left_object_index}"
            )
            return None

        left_object = objects[left_object_index]
        right_object = objects[right_object_index]
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 同步双抓原物体对重选开始: "
            f"left_object_index={left_object_index}, left_object_id={left_object.get('object_id')}, "
            f"right_object_index={right_object_index}, right_object_id={right_object.get('object_id')}"
        )

        left_targets = self._original_dual_side_targets(
            "left",
            current_sample,
            fixed_knee_from_base,
            arm_targets,
            refreshed_selected,
            left_ok,
            left_object,
            left_object_index,
            pair_info,
        )
        right_targets = self._original_dual_side_targets(
            "right",
            current_sample,
            fixed_knee_from_base,
            arm_targets,
            refreshed_selected,
            right_ok,
            right_object,
            right_object_index,
            pair_info,
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 同步双抓原物体对候选统计: "
            f"left_valid={len(left_targets)}, right_valid={len(right_targets)}"
        )
        if not left_targets or not right_targets:
            return None

        selected = {
            "sample": current_sample,
            "sample_order": 0,
            "left": left_targets[0],
            "right": right_targets[0],
        }
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 同步双抓原物体对重选成功: "
            f"left_grasp_index={int(selected['left'].get('index', -1)) + 1}, "
            f"left_pose_label={selected['left'].get('pose_label')}, "
            f"right_grasp_index={int(selected['right'].get('index', -1)) + 1}, "
            f"right_pose_label={selected['right'].get('pose_label')}"
        )
        return selected

    def _read_dual_selected_pair_info(self):
        """读取同步双抓原始物体对信息，缺失时返回 None。"""
        if not self.blackboard.exists(self.dual_selected_pair_info_key):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原物体对信息不存在: key={self.dual_selected_pair_info_key}"
            )
            return None
        pair_info = self.blackboard.get(self.dual_selected_pair_info_key)
        if not isinstance(pair_info, dict):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原物体对信息格式异常: {pair_info!r}"
            )
            return None
        return pair_info

    def _read_sorted_grasp_objects_for_dual_refresh(self):
        """读取当前物体缓存，供移动后原物体对重选使用。"""
        if not self.blackboard.exists(self.sorted_grasp_objects_key):
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓物体缓存不存在: key={self.sorted_grasp_objects_key}"
            )
            return None
        objects = self.blackboard.get(self.sorted_grasp_objects_key)
        if not isinstance(objects, list) or not objects:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓物体缓存为空或格式异常"
            )
            return None
        return objects

    def _find_original_dual_object_index(self, objects, pair_info, arm_side):
        """按 object_id 优先、原索引兜底，定位复核前选中的同步双抓物体。"""
        object_id = pair_info.get(f"{arm_side}_object_id")
        if object_id is not None:
            for object_index, grasp_object in enumerate(objects):
                if grasp_object.get("object_id") == object_id:
                    return object_index

        try:
            object_index = int(pair_info.get(f"{arm_side}_object_index", -1))
        except (TypeError, ValueError):
            return None
        if 0 <= object_index < len(objects):
            return object_index
        return None

    def _original_dual_side_targets(
        self,
        arm_side,
        current_sample,
        fixed_knee_from_base,
        arm_targets,
        refreshed_selected,
        side_ok,
        grasp_object,
        object_index,
        pair_info,
    ):
        """返回原物体在当前腰部下可用于指定手臂的候选列表。"""
        if side_ok:
            target = dict(refreshed_selected[arm_side])
            # 关键步骤：原候选复核通过时直接保留，但补齐物体和 grasp index 元信息。
            self._attach_dual_object_info_to_target(
                target,
                grasp_object,
                object_index,
                self._read_dual_grasp_index(pair_info, arm_side, target.get("index", -1)),
            )
            return [target]

        grasp_poses = grasp_object.get("map_grasp_poses", [])
        if not isinstance(grasp_poses, list) or not grasp_poses:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓原物体{arm_side}手无 grasp pose: "
                f"object_index={object_index}, object_id={grasp_object.get('object_id')}"
            )
            return []

        targets = self._find_all_valid_grasp_targets_for_torso_sample(
            current_sample,
            grasp_poses,
            MAP_FRAME,
            fixed_knee_from_base,
            arm_targets,
            allowed_arm_side=arm_side,
        )
        for target in targets:
            self._attach_dual_object_info_to_target(
                target,
                grasp_object,
                object_index,
                int(target.get("index", -1)),
            )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 同步双抓原物体{arm_side}手重选候选: "
            f"object_index={object_index}, object_id={grasp_object.get('object_id')}, "
            f"pose_count={len(grasp_poses)}, valid_count={len(targets)}"
        )
        return targets

    def _attach_dual_object_info_to_target(self, target, grasp_object, object_index, grasp_index):
        """给同步双抓单侧目标补齐物体缓存信息。"""
        target["object_index"] = object_index
        target["object_id"] = grasp_object.get("object_id")
        target["object_info"] = grasp_object
        target["index"] = grasp_index

    def _read_dual_grasp_index(self, pair_info, arm_side, default_index):
        """读取同步双抓记录里的 grasp index，异常时使用默认值。"""
        try:
            return int(pair_info.get(f"{arm_side}_grasp_index", default_index))
        except (TypeError, ValueError):
            try:
                return int(default_index)
            except (TypeError, ValueError):
                return -1

    def _refresh_single_dual_side_target(self, arm_side, waist_from_base, base_from_map, arm_targets):
        """用当前真实 TF 重投影同步双抓中单侧手的抓取矩阵。"""
        map_key = (
            self.left_selected_map_grasp_pose_key
            if arm_side == "left"
            else self.right_selected_map_grasp_pose_key
        )
        base_key = (
            self.left_selected_base_grasp_pose_key
            if arm_side == "left"
            else self.right_selected_base_grasp_pose_key
        )
        rotate_key = (
            self.left_selected_rotate_z_180_key
            if arm_side == "left"
            else self.right_selected_rotate_z_180_key
        )
        if not self.blackboard.exists(rotate_key):
            raise RuntimeError(f"{arm_side} 手抓取翻转标记不存在: key={rotate_key}")
        if self.blackboard.exists(map_key) and self.blackboard.get(map_key) is not None:
            source_grasp_pose = self.blackboard.get(map_key)
            grasp_pose = base_from_map @ source_grasp_pose
            poses_frame = MAP_FRAME
        elif self.blackboard.exists(base_key):
            source_grasp_pose = self.blackboard.get(base_key)
            grasp_pose = source_grasp_pose
            poses_frame = BASE_LINK_FRAME
        else:
            raise RuntimeError(f"{arm_side} 手抓取矩阵不存在: map_key={map_key}, base_key={base_key}")

        rotate_z_180 = bool(self.blackboard.get(rotate_key))
        grasp_target, grasp_target_pose = build_grasp_target(
            grasp_pose,
            waist_from_base,
            offset_m=self.grasp_offset_m,
            rotate_z_180=rotate_z_180,
        )
        return {
            "sample": None,
            "arm_side": arm_side,
            "index": -1,
            "pose_label": "刷新原候选",
            "rotate_z_180": rotate_z_180,
            "poses_frame": poses_frame,
            "arm_targets": {side: list(target) for side, target in arm_targets.items()},
            "source_grasp_pose": source_grasp_pose,
            "grasp_pose": grasp_pose,
            "grasp_target_pose": grasp_target_pose,
            "grasp_target": grasp_target,
            "pregrasp_target_pose": None,
            "pregrasp_target": None,
            "pregrasp_offset_m": None,
            "waist_from_base": waist_from_base,
        }

    def _dual_side_target_still_valid(self, arm_side, selected):
        """检查同步双抓中单侧手的最终抓取和预抓取是否仍可达。"""
        try:
            grasp_valid, _, pos_error, angle_error = self._check_arm_pose(
                arm_side,
                selected["grasp_target_pose"],
                pos_threshold=0.01,
                angle_threshold=0.05,
            )
            pregrasp_result = self._find_valid_pregrasp(arm_side, selected["grasp_target_pose"])
        except Exception as exc:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓{arm_side}手复核异常: {exc}"
            )
            return False
        if not grasp_valid or pregrasp_result is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 同步双抓{arm_side}手复核失败: "
                f"pos={pos_error:.4f}m/0.0100m, angle={angle_error:.4f}rad/0.0500rad, "
                f"pregrasp_valid={pregrasp_result is not None}"
            )
            return False
        pregrasp_pose, pregrasp_target, pregrasp_offset_m = pregrasp_result
        selected["pregrasp_target_pose"] = pregrasp_pose
        selected["pregrasp_target"] = pregrasp_target
        selected["pregrasp_offset_m"] = pregrasp_offset_m
        return True

