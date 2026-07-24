"""navigation HTTP service tree 任务适配器。"""

import py_trees

from tree.runtime.http_service.models import NavigationRequestModel


TASK_INPUT_KEYS = {
    "task_id": "navigation_active_task_id",
    "target_pose": "navigation_target_pose",
}


def _model_to_dict(model):
    """
    转换 Pydantic model 为普通 dict。
    :param model: Pydantic model 实例。
    :return: 可 JSON 序列化 dict。
    """
    return model.model_dump()


class NavigationTaskAdapter:
    """负责 navigation 请求校验和 blackboard 写入。"""

    def __init__(self):
        self.blackboard = py_trees.blackboard.Client(name="navigation_http_task_adapter")
        for key in TASK_INPUT_KEYS.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def parse_payload(self, payload):
        """
        校验并归一化 start_navigation 请求。
        :param payload: HTTP JSON 请求体。
        :return: 归一化后的任务请求。
        """
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        # 关键步骤：导航请求只接受目标点，数值校验复用 Pose2DModel。
        return _model_to_dict(NavigationRequestModel.model_validate(payload))

    def write_blackboard(self, task_id, request):
        """
        将任务输入写入 blackboard。
        :param task_id: 本次任务 ID。
        :param request: parse_payload 返回的归一化请求。
        """
        # 关键步骤：行为树只从 blackboard 读取目标点，HTTP 字段不直接传给叶子节点。
        self.blackboard.set(
            TASK_INPUT_KEYS["target_pose"],
            request["targetPose"],
            overwrite=True,
        )
        self.blackboard.set(TASK_INPUT_KEYS["task_id"], task_id, overwrite=True)
