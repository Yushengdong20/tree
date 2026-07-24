"""grasp_object 重构后的公共执行模板和解析器测试。"""

import numpy as np
import py_trees
from py_trees.common import Status

from tree.node.grasp_object.helper import grasp_compute_base
from tree.node.grasp_object.helper.grasp_compute_base import (
    AsyncGraspComputeBase,
    GraspComputeBase,
)
from tree.node.grasp_object.helper.grasp_request_parser import GraspObjectPayloadParser


class _Logger:
    """测试用日志对象。"""

    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class _RosNode:
    """测试用最小 ros_node。"""

    def __init__(self):
        self.logger = _Logger()
        self.live_runtime = None
        self.clear_count = 0

    def get_logger(self):
        return self.logger

    def set_live_runtime(self, *values):
        self.live_runtime = values

    def clear_live_runtime(self):
        self.clear_count += 1
        self.live_runtime = None


class _Computer:
    """测试用抓取计算器替身。"""

    def __init__(self, *_, **__):
        self.base_grasp_poses_key = "base_key"
        self.sorted_grasp_objects_key = "objects_key"
        self.pregrasp_pose_key = "pregrasp_key"


class _SyncAction(GraspComputeBase):
    """同步 compute 模板测试节点。"""

    runtime_code = "SYNC_TEST"
    runtime_message = "sync test"

    def run_grasp_compute(self):
        return {"ok": True}


class _FailingSyncAction(_SyncAction):
    """同步失败路径测试节点。"""

    def run_grasp_compute(self):
        raise RuntimeError("boom")


class _AsyncAction(AsyncGraspComputeBase):
    """异步 compute 模板测试节点。"""

    runtime_code = "ASYNC_TEST"
    runtime_message = "async test"

    def run_grasp_compute(self):
        return {"ok": True}


class _Parser(GraspObjectPayloadParser):
    """payload 解析器测试宿主。"""

    def __init__(self):
        self.config_label = "解析测试"
        self.ros_node = _RosNode()
        self.filter_downward_grasp_poses = True
        self.downward_grasp_max_angle_deg = 45.0
        self.distance_metric = "horizontal"


def test_sync_grasp_compute_base_returns_success(monkeypatch):
    """同步模板应统一设置 runtime、执行任务并清理 runtime。"""
    monkeypatch.setattr(grasp_compute_base, "GraspTargetComputer", _Computer)
    node = _SyncAction("Sync", "同步测试", _RosNode(), {})

    assert node.update() == Status.SUCCESS
    assert node.ros_node.clear_count == 1


def test_sync_grasp_compute_base_converts_exception_to_failure(monkeypatch):
    """同步模板应把计算异常转成 FAILURE。"""
    monkeypatch.setattr(grasp_compute_base, "GraspTargetComputer", _Computer)
    node = _FailingSyncAction("SyncFail", "同步失败测试", _RosNode(), {})

    assert node.update() == Status.FAILURE
    assert node.feedback_message == "boom"


def test_async_grasp_compute_base_returns_running_then_success(monkeypatch):
    """异步模板第一次 tick 提交任务，任务完成后返回 SUCCESS。"""
    monkeypatch.setattr(grasp_compute_base, "GraspTargetComputer", _Computer)
    node = _AsyncAction("Async", "异步测试", _RosNode(), {})

    assert node.update() == Status.RUNNING
    node._sample_future.result(timeout=1.0)
    assert node.update() == Status.SUCCESS


def test_payload_parser_rejects_invalid_matrix():
    """解析器应拒绝非齐次矩阵。"""
    parser = _Parser()
    invalid_pose = np.eye(4)
    invalid_pose[3, 3] = 2.0

    try:
        parser._validate_grasp_pose(invalid_pose, "测试位姿")
    except RuntimeError as exc:
        assert "齐次矩阵" in str(exc)
    else:
        raise AssertionError("invalid matrix was accepted")


def test_payload_parser_filters_downward_grasp_objects():
    """向下抓取过滤应保留局部 z 轴朝下的候选。"""
    parser = _Parser()
    downward_pose = np.eye(4)
    downward_pose[:3, 2] = [0.0, 0.0, -1.0]
    side_pose = np.eye(4)
    side_pose[:3, 2] = [1.0, 0.0, 0.0]

    objects = [
        {
            "object_id": 1,
            "camera_grasp_poses": [downward_pose, side_pose],
            "base_grasp_poses": [downward_pose, side_pose],
            "map_grasp_poses": [downward_pose, side_pose],
            "scores": [0.8, 0.1],
            "openings": [0.02, 0.03],
        }
    ]

    filtered = parser._filter_downward_grasp_objects(objects)

    assert len(filtered) == 1
    assert len(filtered[0]["base_grasp_poses"]) == 1
    assert filtered[0]["scores"] == [0.8]


def test_payload_parser_sorts_objects_by_horizontal_distance():
    """multi 物体排序默认使用 base_link 下水平距离。"""
    parser = _Parser()
    near_pose = np.eye(4)
    near_pose[:3, 3] = [0.1, 0.0, 5.0]
    far_pose = np.eye(4)
    far_pose[:3, 3] = [1.0, 0.0, 0.0]

    sorted_objects = parser._sort_grasp_objects(
        [
            {"object_id": 2, "base_grasp_poses": [far_pose]},
            {"object_id": 1, "base_grasp_poses": [near_pose]},
        ]
    )

    assert [grasp_object["object_id"] for grasp_object in sorted_objects] == [1, 2]
