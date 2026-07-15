"""双手 C++ 抓取搜索的确定性与预抓取约束测试。"""

import json
from pathlib import Path

import numpy as np
import pytest


grasp_search_cpp = pytest.importorskip(
    "tree.node.grasp_object._grasp_search_cpp",
    reason="需要先构建 grasp_search_cpp 扩展",
)


def _grasp_pose(x, y, z):
    """构造 base_link 下无旋转的抓取候选。"""
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def _search(grasp_pose, **overrides):
    """用单个当前腰部 sample 执行双手搜索。"""
    params = {
        "sample_fixed_knee_from_waist": np.eye(4, dtype=float)[None, ...],
        "sample_base_from_map": np.eye(4, dtype=float)[None, ...],
        "sample_enabled": np.array([False], dtype=bool),
        "sample_chassis_lateral_enabled": np.array([False], dtype=bool),
        "grasp_poses": np.asarray([grasp_pose], dtype=float),
        "poses_frame": "base_link",
        "fixed_knee_from_base": np.eye(4, dtype=float),
        "grasp_offset_m": 0.06,
        "model_type": "grasp",
        "pos_threshold": 0.01,
        "angle_threshold": 0.05,
        "pregrasp_offset_min_m": 0.12,
        "pregrasp_offset_max_m": 0.18,
        "pregrasp_offset_samples": 5,
        "cpp_search_workers": 2,
        "cpp_search_batch_size": 8,
    }
    params.update(overrides)
    return grasp_search_cpp.find_grasp_target(**params)


def test_equal_cost_prefers_right_arm():
    """两手同成本可达时稳定选择右手。"""
    result = _search(_grasp_pose(0.35, 0.0, 0.30))
    assert result is not None
    assert result["arm_side"] == "right"


def test_left_arm_is_selected_when_right_arm_is_unreachable():
    """物体偏左且右手不可达时选择左手。"""
    result = _search(_grasp_pose(0.25, 0.10, 0.30))
    assert result is not None
    assert result["arm_side"] == "left"


def test_candidate_requires_reachable_pregrasp():
    """最终抓取可达但预抓取距离过大时不得返回候选。"""
    result = _search(
        _grasp_pose(0.35, 0.0, 0.30),
        pregrasp_offset_min_m=1.0,
        pregrasp_offset_max_m=1.0,
        pregrasp_offset_samples=1,
    )
    assert result is None


def test_sample_cost_has_priority_over_arm_preference_with_multiple_workers():
    """更小移动 sample 的左手解应优先于后续 sample 的右手解。"""
    sample_transforms = np.repeat(np.eye(4, dtype=float)[None, ...], 2, axis=0)
    base_from_map = sample_transforms.copy()
    base_from_map[0, 1, 3] = 0.10
    base_from_map[1, 1, 3] = -0.10
    for _ in range(5):
        result = _search(
            _grasp_pose(0.25, 0.0, 0.30),
            sample_fixed_knee_from_waist=sample_transforms,
            sample_base_from_map=base_from_map,
            sample_enabled=np.array([False, False], dtype=bool),
            sample_chassis_lateral_enabled=np.array([False, True], dtype=bool),
            poses_frame="map",
            cpp_search_batch_size=1,
        )
        assert result is not None
        assert result["sample_index"] == 0
        assert result["arm_side"] == "left"


def test_all_grasp_object_json_files_are_valid():
    """确保所有抓取入口和子树都能解析为 JSON。"""
    config_root = Path(__file__).parents[1] / "config" / "tree" / "grasp_object"
    files = sorted(config_root.rglob("*.json"))
    assert files
    for config_file in files:
        with config_file.open("r", encoding="utf-8") as handle:
            json.load(handle)
