"""从 multi 抓取结果里按距离顺序选择下一个物体。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class SelectNextGraspObject(TimedMockAction):
    """把排序后的单个物体抓取候选写回旧 pose key，复用后续抓取计算节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.sorted_grasp_objects_key = str(
            params.get("sorted_grasp_objects_key", "grasp_object_sorted_grasp_objects")
        ).strip()
        self.next_grasp_object_index_key = str(
            params.get("next_grasp_object_index_key", "grasp_object_next_grasp_object_index")
        ).strip()
        self.selected_grasp_object_info_key = str(
            params.get("selected_grasp_object_info_key", "grasp_object_selected_grasp_object_info")
        ).strip()
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
        self.advance_cursor = self._to_bool(params.get("advance_cursor", True))

        if not self.sorted_grasp_objects_key:
            raise ValueError("sorted_grasp_objects_key 不能为空")
        if not self.next_grasp_object_index_key:
            raise ValueError("next_grasp_object_index_key 不能为空")
        if not self.selected_grasp_object_info_key:
            raise ValueError("selected_grasp_object_info_key 不能为空")
        if not self.base_grasp_poses_key:
            raise ValueError("base_grasp_poses_key 不能为空")
        if not self.camera_grasp_poses_key:
            raise ValueError("camera_grasp_poses_key 不能为空")
        if not self.map_grasp_poses_key:
            raise ValueError("map_grasp_poses_key 不能为空")

        self.blackboard.register_key(
            key=self.sorted_grasp_objects_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.next_grasp_object_index_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.next_grasp_object_index_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.selected_grasp_object_info_key,
            access=py_trees.common.Access.READ,
        )
        self.blackboard.register_key(
            key=self.selected_grasp_object_info_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.base_grasp_poses_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.camera_grasp_poses_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=self.map_grasp_poses_key,
            access=py_trees.common.Access.WRITE,
        )

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            objects = self._read_objects()
            selected_index = self._get_selected_index(objects)
            selected_object = objects[selected_index]
            base_grasp_poses = selected_object.get("base_grasp_poses", [])
            camera_grasp_poses = selected_object.get("camera_grasp_poses", [])
            map_grasp_poses = selected_object.get("map_grasp_poses", [])
            if not isinstance(base_grasp_poses, list) or not base_grasp_poses:
                raise RuntimeError(f"第 {selected_index + 1} 个物体没有 base_link 抓取位姿")
            if not isinstance(camera_grasp_poses, list):
                raise RuntimeError(f"第 {selected_index + 1} 个物体 camera 抓取位姿格式无效")
            if not isinstance(map_grasp_poses, list) or not map_grasp_poses:
                raise RuntimeError(f"第 {selected_index + 1} 个物体没有 map 抓取位姿")

            # 关键步骤：后续 ComputeGraspTarget 仍读取旧 key，这里只切换当前物体候选集合。
            self.blackboard.set(self.base_grasp_poses_key, base_grasp_poses, overwrite=True)
            self.blackboard.set(self.camera_grasp_poses_key, camera_grasp_poses, overwrite=True)
            self.blackboard.set(self.map_grasp_poses_key, map_grasp_poses, overwrite=True)

            selected_info = self._build_selected_info(selected_index, len(objects), selected_object)
            self.blackboard.set(
                self.selected_grasp_object_info_key,
                selected_info,
                overwrite=True,
            )
            if self.advance_cursor:
                self.blackboard.set(
                    self.next_grasp_object_index_key,
                    selected_index + 1,
                    overwrite=True,
                )

            self.ros_node.get_logger().info(
                f"[{self.config_label}] 已选择第 {selected_index + 1}/{len(objects)} 个抓取物体: "
                f"object_id={selected_info['object_id']}, "
                f"distance={selected_info['distance_m']:.4f}m, "
                f"pose_count={selected_info['pose_count']}, "
                f"advance_cursor={self.advance_cursor}"
            )
            return Status.SUCCESS
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.get_logger().error(f"[{self.config_label}] 选择 multi 抓取物体失败: {exc}")
            return Status.FAILURE

    def _read_objects(self):
        """读取并检查按距离排序后的物体列表。"""
        if not self.blackboard.exists(self.sorted_grasp_objects_key):
            raise RuntimeError(f"multi 抓取物体列表不存在: key={self.sorted_grasp_objects_key}")
        objects = self.blackboard.get(self.sorted_grasp_objects_key)
        if not isinstance(objects, list) or not objects:
            raise RuntimeError(f"multi 抓取物体列表为空: key={self.sorted_grasp_objects_key}")
        return objects

    def _get_selected_index(self, objects):
        """advance_cursor=True 时取下一个物体，否则复用当前已选物体序号。"""
        if self.advance_cursor:
            if not self.blackboard.exists(self.next_grasp_object_index_key):
                return 0
            selected_index = int(self.blackboard.get(self.next_grasp_object_index_key))
        else:
            selected_index = self._read_current_selected_index()
        if selected_index < 0 or selected_index >= len(objects):
            raise RuntimeError(
                f"multi 抓取物体序号越界: index={selected_index}, count={len(objects)}"
            )
        return selected_index

    def _read_current_selected_index(self):
        """回退路径里复用当前物体，避免跳到下一个距离更远的物体。"""
        if self.blackboard.exists(self.selected_grasp_object_info_key):
            selected_info = self.blackboard.get(self.selected_grasp_object_info_key)
            if isinstance(selected_info, dict) and "rank_index" in selected_info:
                return int(selected_info["rank_index"])
        if self.blackboard.exists(self.next_grasp_object_index_key):
            return int(self.blackboard.get(self.next_grasp_object_index_key)) - 1
        raise RuntimeError(f"当前选中物体信息不存在: key={self.selected_grasp_object_info_key}")

    @staticmethod
    def _build_selected_info(selected_index, object_count, selected_object):
        """生成轻量调试信息，避免把完整矩阵重复写入 info key。"""
        return {
            "rank_index": selected_index,
            "object_count": object_count,
            "object_id": selected_object.get("object_id"),
            "bbox": selected_object.get("bbox"),
            "distance_m": float(selected_object.get("distance_m", 0.0)),
            "pose_count": len(selected_object.get("base_grasp_poses", [])),
            "scores": list(selected_object.get("scores", [])),
            "openings": list(selected_object.get("openings", [])),
        }

    def describe_start(self):
        return (
            f"[{self.config_label}] SelectNextGraspObject start: "
            f"objects_key={self.sorted_grasp_objects_key}, "
            f"next_index_key={self.next_grasp_object_index_key}, "
            f"advance_cursor={self.advance_cursor}"
        )
