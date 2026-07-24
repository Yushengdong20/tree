"""YOLO 记忆版直接抓箱放箱 HTTP 任务适配器。"""

import py_trees

from tree.runtime.http_service.models import DirectGraspPlaceMemoryRequestModel


TASK_INPUT_KEYS = {
    "task_id": "move_box_direct_memory_active_task_id",
    "done_count": "move_box_direct_memory_done_count",
    "target_count": "move_box_direct_memory_target_count",
    "find_navigation_target": "move_box_navi_pose_find_box",
    "valid_polygon": "move_box_valid_polygon",
    "place_navigation_target": "move_box_navi_pose_place_box",
    "place_plane_height": "move_box_height_place_plane",
    "box_memory": "move_box_box_memory",
    "current_box_target": "move_box_current_box_target",
    "finished_box_targets": "move_box_finished_box_targets",
}


class DirectGraspPlaceMemoryTaskAdapter:
    """将 HTTP 入参写入直接抓箱放箱 service tree 所需黑板键。"""

    def __init__(self):
        self.blackboard = py_trees.blackboard.Client(
            name="direct_grasp_place_memory_http_task_adapter"
        )
        for key in TASK_INPUT_KEYS.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def parse_payload(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return DirectGraspPlaceMemoryRequestModel.model_validate(payload).model_dump()

    def write_blackboard(self, task_id, request):
        """重置本任务状态并写入外部给定的 A/B 点、区域和放置高度。"""
        self.blackboard.set(TASK_INPUT_KEYS["task_id"], task_id, overwrite=True)
        self.blackboard.set(TASK_INPUT_KEYS["done_count"], 0, overwrite=True)
        self.blackboard.set(
            TASK_INPUT_KEYS["target_count"], int(request["targetCount"]), overwrite=True
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["find_navigation_target"],
            request["naviPoseFindBox"],
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["valid_polygon"], request["validPolygon"], overwrite=True
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["place_navigation_target"],
            request["naviPosePlaceBox"],
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["place_plane_height"],
            float(request["heightPlacePlane"]),
            overwrite=True,
        )
        # 每个 HTTP 任务独占一次记忆周期，避免使用上次任务已经抓完的目标。
        self.blackboard.set(TASK_INPUT_KEYS["box_memory"], [], overwrite=True)
        self.blackboard.set(TASK_INPUT_KEYS["current_box_target"], None, overwrite=True)
        self.blackboard.set(TASK_INPUT_KEYS["finished_box_targets"], [], overwrite=True)
