"""YOLO 记忆版直接抓箱放箱 HTTP 服务的请求/适配器基础测试。"""

import py_trees

from tree.runtime.http_service.models import DirectGraspPlaceMemoryRequestModel
from tree.runtime.http_service.task_adapters.direct_grasp_place_memory import (
    DirectGraspPlaceMemoryTaskAdapter,
    TASK_INPUT_KEYS,
)


def test_direct_grasp_place_memory_request_normalizes_payload():
    model = DirectGraspPlaceMemoryRequestModel.model_validate(
        {
            "naviPoseFindBox": {"x": "0.84", "y": 4.26, "yaw": -105.02},
            "validPolygon": [[-1.5, 3.0], [-1.5, -1.3], [3.0, -1.3]],
            "naviPosePlaceBox": {"x": 1.84, "y": "4.26", "yaw": 6.3},
            "heightPlacePlane": "0.7",
            "targetCount": "2",
        }
    )

    assert model.targetCount == 2
    assert model.heightPlacePlane == 0.7
    assert model.validPolygon[0].x == -1.5


def test_direct_grasp_place_memory_adapter_writes_task_inputs():
    adapter = DirectGraspPlaceMemoryTaskAdapter()
    request = adapter.parse_payload(
        {
            "naviPoseFindBox": {"x": 0.84, "y": 4.26, "yaw": -105.02},
            "validPolygon": [[-1.5, 3.0], [-1.5, -1.3], [3.0, -1.3]],
            "naviPosePlaceBox": {"x": 1.84, "y": 4.26, "yaw": 6.3},
            "heightPlacePlane": 0.7,
        }
    )
    adapter.write_blackboard("task-direct-memory", request)

    blackboard = py_trees.blackboard.Blackboard
    assert blackboard.get(TASK_INPUT_KEYS["task_id"]) == "task-direct-memory"
    assert blackboard.get(TASK_INPUT_KEYS["target_count"]) == 1
    assert blackboard.get(TASK_INPUT_KEYS["done_count"]) == 0
    assert blackboard.get(TASK_INPUT_KEYS["valid_polygon"])[0]["x"] == -1.5
    assert blackboard.get(TASK_INPUT_KEYS["box_memory"]) == []
