"""move_box HTTP service tree 任务适配器。"""

import py_trees

from tree.runtime.http_service.models import MoveBoxRequestModel


TASK_INPUT_KEYS = {
    "task_id": "move_box_active_task_id",
    "navi_pose_find_box": "move_box_navi_pose_find_box",
    "valid_polygon": "move_box_valid_polygon",
    "navi_pose_place_box": "move_box_navi_pose_place_box",
    "box_pose_place_center": "move_box_box_pose_place_center",
    "height_place_plane": "move_box_height_place_plane",
}


def _model_to_dict(model):
    """
    转换 Pydantic model 为普通 dict。
    :param model: Pydantic model 实例。
    :return: 可 JSON 序列化 dict。
    """
    return model.model_dump()


class MoveBoxTaskAdapter:
    """负责 move_box 请求校验和 blackboard 写入。"""

    def __init__(self):
        self.blackboard = py_trees.blackboard.Client(name="move_box_http_task_adapter")
        for key in TASK_INPUT_KEYS.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def parse_payload(self, payload):
        """
        校验并归一化 start_move_box 请求。
        :param payload: HTTP JSON 请求体。
        :return: 归一化后的任务请求。
        """
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        # 关键步骤：请求协议集中交给 Pydantic model 校验和归一化。
        return _model_to_dict(MoveBoxRequestModel.model_validate(payload))

    def write_blackboard(self, task_id, request):
        """
        将任务输入写入 blackboard。
        :param task_id: 本次任务 ID。
        :param request: parse_payload 返回的归一化请求。
        """
        # 关键步骤：所有外部请求字段统一写入 blackboard，行为树只读取这些 key。
        self.blackboard.set(
            TASK_INPUT_KEYS["navi_pose_find_box"],
            request["naviPoseFindBox"],
            overwrite=True,
        )
        self.blackboard.set(TASK_INPUT_KEYS["valid_polygon"], request["validPolygon"], overwrite=True)
        self.blackboard.set(
            TASK_INPUT_KEYS["navi_pose_place_box"],
            request["naviPosePlaceBox"],
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["box_pose_place_center"],
            request["boxPosePlaceCenter"],
            overwrite=True,
        )
        self.blackboard.set(
            TASK_INPUT_KEYS["height_place_plane"],
            request["heightPlacePlane"],
            overwrite=True,
        )
        self.blackboard.set(TASK_INPUT_KEYS["task_id"], task_id, overwrite=True)
