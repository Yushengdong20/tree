"""启动时快照 YOLO 多个箱体目标，并按距离写入 blackboard 队列。"""

import math
import time

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_CHASSIS_URL,
    build_chassis_config,
    get_chassis_current_pose,
    transform_base_point_to_global,
)


class MoveBoxSnapshotYoloTargets(TimedMockAction):
    """读取启动时 YOLO 多目标，将箱体中心转换到 global 后按近到远排序。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.target_queue_key = str(params.get("target_queue_key", "move_box_yolo_target_queue")).strip()
        self.max_targets = max(1, int(params.get("max_targets", 3)))
        self.min_targets = max(0, int(params.get("min_targets", 1)))
        self.dedupe_distance_m = max(0.0, float(params.get("dedupe_distance_m", 0.25)))
        self.detect_timeout_sec = max(0.0, float(params.get("detect_timeout_sec", 3.0)))
        self.poll_interval_sec = max(0.01, float(params.get("poll_interval_sec", 0.2)))
        self.front_only = self._to_bool(params.get("front_only", True))
        self.chassis_config = build_chassis_config(
            base_url=str(params.get("chassis_url", DEFAULT_CHASSIS_URL)).strip(),
        )

        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.target_queue_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key="flow_result", access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            self.ros_node.set_live_runtime(self.config_label, "YOLO_SNAPSHOT", "读取启动时底盘位姿")
            current_pose = get_chassis_current_pose(self.chassis_config)
            services = self._get_services()
            target_poses = self._wait_yolo_target_poses(services)
            targets = self._build_sorted_targets(current_pose, target_poses)
            if len(targets) < self.min_targets:
                raise RuntimeError(
                    f"YOLO 目标数量不足: got={len(targets)}, min_targets={self.min_targets}"
                )

            self.blackboard.set(self.target_queue_key, targets, overwrite=True)
            self._store_result(current_pose, targets)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已记录 YOLO 箱体队列: count={len(targets)}, "
                f"distances={[round(item['distanceFromStartM'], 3) for item in targets]}"
            )
            return Status.SUCCESS
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] YOLO 多目标快照失败: {exc}")
            return Status.FAILURE

    def _get_services(self):
        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            raise RuntimeError(f"move_box services missing on blackboard: key={self.services_key}")
        if not hasattr(services, "yolo_detector"):
            raise RuntimeError("move_box services 缺少 yolo_detector")
        return services

    def _wait_yolo_target_poses(self, services):
        """等待 YOLO 输出完整多目标列表。"""
        deadline = time.monotonic() + self.detect_timeout_sec
        latest_target_poses = []
        while time.monotonic() <= deadline:
            self.ros_node.set_live_runtime(
                self.config_label,
                "YOLO_SNAPSHOT",
                "等待 YOLO 多目标检测结果",
            )
            updated = services.yolo_detector.update_latest_target_poses()
            latest_target_poses = services.yolo_detector.get_latest_target_poses()
            # 关键步骤：只接受本轮等待期间的新 YOLO 消息，避免复用上一次任务的旧缓存。
            if updated and latest_target_poses:
                return latest_target_poses
            time.sleep(self.poll_interval_sec)

        raise RuntimeError(
            f"{self.detect_timeout_sec:.1f}s 内未获得有效 YOLO 多目标检测结果: "
            f"latest_count={len(latest_target_poses)}"
        )

    def _build_sorted_targets(self, current_pose, target_poses):
        """将 base_link 下的箱体点转换到 global，并按启动距离排序。"""
        targets = []
        for pose_stamped in target_poses:
            position = pose_stamped.pose.position
            base_position = {
                "x": float(position.x),
                "y": float(position.y),
                "z": float(position.z),
            }
            if self.front_only and base_position["x"] <= 0.0:
                continue

            global_position = transform_base_point_to_global(
                current_pose,
                base_position["x"],
                base_position["y"],
            )
            global_position["z"] = base_position["z"]
            distance_m = math.hypot(base_position["x"], base_position["y"])
            if self._is_duplicate_target(targets, global_position):
                continue

            targets.append(
                {
                    "id": f"box_{len(targets)}",
                    "basePositionAtSnapshot": base_position,
                    "globalPosition": global_position,
                    "distanceFromStartM": distance_m,
                }
            )

        targets.sort(key=self._get_target_distance)
        return targets[: self.max_targets]

    def _is_duplicate_target(self, targets, global_position):
        """按 global 平面距离去掉重复检测框。"""
        if self.dedupe_distance_m <= 0.0:
            return False

        for target in targets:
            existing = target["globalPosition"]
            distance_m = math.hypot(
                global_position["x"] - existing["x"],
                global_position["y"] - existing["y"],
            )
            if distance_m < self.dedupe_distance_m:
                return True
        return False

    @staticmethod
    def _get_target_distance(target):
        """返回启动时 base_link 到箱体中心的平面距离。"""
        return target["distanceFromStartM"]

    def _store_result(self, current_pose, targets):
        try:
            flow_result = self.blackboard.flow_result
        except (AttributeError, KeyError):
            flow_result = None
        if flow_result is None:
            flow_result = {}

        flow_result["yoloTargetSnapshot"] = {
            "startPose": {
                "x": current_pose.x,
                "y": current_pose.y,
                "yaw": current_pose.yaw,
            },
            "targets": targets,
            "maxTargets": self.max_targets,
            "dedupeDistanceM": self.dedupe_distance_m,
            "frontOnly": self.front_only,
        }
        self.blackboard.flow_result = flow_result

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxSnapshotYoloTargets start"
