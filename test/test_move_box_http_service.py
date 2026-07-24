"""move_box HTTP 服务的请求、适配器与 service tree 静态测试。"""

import json
from pathlib import Path

import pytest

from tree.runtime.http_service.models import MoveBoxRequestModel
from tree.runtime.http_service.task_adapters.move_box import (
    MoveBoxTaskAdapter,
    TASK_INPUT_KEYS,
)


def test_move_box_request_model_normalizes_numbers_and_polygon_points():
    """HTTP 请求应归一化数字字符串与 [x, y] 多边形点。"""
    model = MoveBoxRequestModel.model_validate(
        {
            "naviPoseFindBox": {"x": "0.84", "y": 4.26, "yaw": "-105.02"},
            "validPolygon": [[-1.5, 3.0], {"x": -1.5, "y": -1.3}, [3.0, -1.3]],
            "naviPosePlaceBox": {"x": 1.84, "y": "4.26", "yaw": 6.3},
            "boxPosePlaceCenter": {"x": 2.10, "y": 4.26, "yaw": 6.3},
            "heightPlacePlane": "0.7",
        }
    )

    assert model.naviPoseFindBox.x == pytest.approx(0.84)
    assert model.validPolygon[0].x == pytest.approx(-1.5)
    assert model.heightPlacePlane == pytest.approx(0.7)


def test_move_box_request_model_rejects_invalid_polygon():
    """少于三个点的有效区域必须在请求层拒绝。"""
    with pytest.raises(ValueError):
        MoveBoxRequestModel.model_validate(
            {
                "naviPoseFindBox": {"x": 0.84, "y": 4.26, "yaw": -105.02},
                "validPolygon": [[0.0, 0.0], [1.0, 1.0]],
                "naviPosePlaceBox": {"x": 1.84, "y": 4.26, "yaw": 6.3},
                "boxPosePlaceCenter": {"x": 2.10, "y": 4.26, "yaw": 6.3},
                "heightPlacePlane": 0.7,
            }
        )


def test_move_box_adapter_writes_all_service_inputs():
    """adapter 应把 A/B/C/H 与有效区域写到 service tree 读取的键。"""
    adapter = MoveBoxTaskAdapter()
    request = adapter.parse_payload(
        {
            "naviPoseFindBox": {"x": 0.84, "y": 4.26, "yaw": -105.02},
            "validPolygon": [[-1.5, 3.0], [-1.5, -1.3], [3.0, -1.3]],
            "naviPosePlaceBox": {"x": 1.84, "y": 4.26, "yaw": 6.3},
            "boxPosePlaceCenter": {"x": 2.10, "y": 4.26, "yaw": 6.3},
            "heightPlacePlane": 0.7,
        }
    )
    adapter.write_blackboard("task-move-box-1", request)

    assert adapter.blackboard.get(TASK_INPUT_KEYS["task_id"]) == "task-move-box-1"
    assert adapter.blackboard.get(TASK_INPUT_KEYS["navi_pose_find_box"])["x"] == 0.84
    assert adapter.blackboard.get(TASK_INPUT_KEYS["valid_polygon"])[0]["x"] == -1.5
    assert adapter.blackboard.get(TASK_INPUT_KEYS["navi_pose_place_box"])["x"] == 1.84
    assert adapter.blackboard.get(TASK_INPUT_KEYS["box_pose_place_center"])["x"] == 2.10
    assert adapter.blackboard.get(TASK_INPUT_KEYS["height_place_plane"]) == 0.7


def test_move_box_service_tree_and_subtree_paths_are_valid_json_files():
    """至少保证 service tree JSON 与全部相对引用子树在提交时存在。"""
    root = (
        Path(__file__).parents[1]
        / "config"
        / "tree"
        / "service"
        / "move_box"
        / "start_move_box.json"
    ).resolve()
    visited = set()

    def check_tree(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "tree" in data

        def visit(node):
            if node.get("name") == "SubTree":
                raw_path = node["params"]["file"]["value"]
                child = (path.parent / raw_path).resolve()
                assert child.is_file(), f"missing subtree: {child}"
                check_tree(child)
            for child in node.get("childs", []):
                visit(child)

        visit(data["tree"])

    check_tree(root)
