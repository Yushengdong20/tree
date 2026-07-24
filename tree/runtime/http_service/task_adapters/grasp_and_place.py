"""grasp_and_place HTTP service tree 任务适配器。"""

import py_trees

from tree.runtime.http_service.models import GraspAndPlaceRequestModel


FIXED_KNEE_Z_FROM_BASE_M = 0.37614321
TORSO_SAMPLE_GRASP_HEIGHT_WINDOW_M = 0.4


TASK_INPUT_KEYS = {
    "task_id": "grasp_and_place_active_task_id",
    "done_count": "grasp_and_place_done_count",
    "target_count": "grasp_and_place_target_count",
    "navi_pose_grasp": "grasp_object_pick_navigation_target",
    "navi_pose_place": "grasp_object_place_navigation_target",
    "height_grasp_plane": "grasp_object_height_grasp_plane",
    "height_place_plane": "grasp_object_height_place_plane",
    "sorted_grasp_objects": "grasp_object_sorted_grasp_objects",
    "next_grasp_object_index": "grasp_object_next_grasp_object_index",
    "grasp_mode": "grasp_object_grasp_mode",
    "torso_sample_z_min": "grasp_object_torso_sample_z_min_m",
    "torso_sample_z_max": "grasp_object_torso_sample_z_max_m",
}


def _model_to_dict(model):
    """
    转换 Pydantic model 为普通 dict。
    :param model: Pydantic model 实例。
    :return: 可 JSON 序列化 dict。
    """
    return model.model_dump()


class GraspAndPlaceTaskAdapter:
    """负责 grasp_and_place 请求校验和 blackboard 写入。"""

    def __init__(self):
        self.blackboard = py_trees.blackboard.Client(
            name="grasp_and_place_http_task_adapter"
        )
        for key in TASK_INPUT_KEYS.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def parse_payload(self, payload):
        """
        校验并归一化 start_grasp_and_place 请求。
        :param payload: HTTP JSON 请求体。
        :return: 归一化后的任务请求。
        """
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        # 关键步骤：HTTP 协议集中交给 Pydantic model 做类型校验和数值归一化。
        return _model_to_dict(GraspAndPlaceRequestModel.model_validate(payload))

    def write_blackboard(self, task_id, request):
        """
        将任务输入写入 blackboard。
        :param task_id: 本次任务 ID。
        :param request: parse_payload 返回的归一化请求。
        """
        # 关键步骤：每个 HTTP 任务重新设置计数和抓取缓存，避免复用上一次任务残留。
        self.blackboard.set(TASK_INPUT_KEYS["done_count"], 0, overwrite=True)
        self.blackboard.set(
            TASK_INPUT_KEYS["target_count"],
            int(request["targetCount"]),
            overwrite=True,
        )
        self.blackboard.set(TASK_INPUT_KEYS["sorted_grasp_objects"], [], overwrite=True)
        self.blackboard.set(TASK_INPUT_KEYS["next_grasp_object_index"], 0, overwrite=True)
        self.blackboard.set(TASK_INPUT_KEYS["grasp_mode"], "single", overwrite=True)

        self.blackboard.set(
            TASK_INPUT_KEYS["navi_pose_grasp"],
            request["naviPoseGrasp"],
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["navi_pose_place"],
            request["naviPosePlace"],
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["height_grasp_plane"],
            request["heightGraspPlane"],
            overwrite=True,
        )
        # 关键步骤：server 模式按本次抓取平面动态收缩 torso 采样高度范围。
        torso_sample_z_min = (
            float(request["heightGraspPlane"]) - FIXED_KNEE_Z_FROM_BASE_M
        )
        torso_sample_z_max = (
            torso_sample_z_min + TORSO_SAMPLE_GRASP_HEIGHT_WINDOW_M
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["torso_sample_z_min"],
            torso_sample_z_min,
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["torso_sample_z_max"],
            torso_sample_z_max,
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["height_place_plane"],
            request["heightPlacePlane"],
            overwrite=True,
        )
        self.blackboard.set(TASK_INPUT_KEYS["task_id"], task_id, overwrite=True)
