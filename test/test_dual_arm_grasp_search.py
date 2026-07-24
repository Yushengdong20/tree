"""双手 C++ 抓取搜索的确定性与预抓取约束测试。"""

import json
from pathlib import Path

import numpy as np
import pytest


grasp_search_cpp = pytest.importorskip(
    "tree.node.grasp_object._grasp_search_cpp",
    reason="需要先构建 grasp_search_cpp 扩展",
)
grasp_search_new_ik_cpp = pytest.importorskip(
    "tree.node.grasp_object._grasp_search_new_ik_cpp",
    reason="需要先构建 grasp_search_new_ik_cpp 扩展",
)

GRASP_SEARCH_MODULES = (grasp_search_cpp, grasp_search_new_ik_cpp)


def _grasp_pose(x, y, z):
    """构造 base_link 下无旋转的抓取候选。"""
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def _reachable_grasp_pose(grasp_search_module, arm_side="both"):
    """按 IK 内核选择一个已知可达的基础抓取位姿。"""
    if grasp_search_module is grasp_search_new_ik_cpp:
        if arm_side == "left":
            return _grasp_pose(0.20, 0.0, 0.25)
        return _grasp_pose(0.20, -0.30, 0.30)
    return _grasp_pose(0.35, 0.0, 0.30)


def _search(grasp_pose, grasp_search_module=grasp_search_cpp, **overrides):
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
    return grasp_search_module.find_grasp_target(**params)


@pytest.mark.parametrize("grasp_search_module", GRASP_SEARCH_MODULES)
def test_equal_cost_prefers_right_arm(grasp_search_module):
    """两手同成本可达时稳定选择右手。"""
    result = _search(_reachable_grasp_pose(grasp_search_module), grasp_search_module)
    assert result is not None
    assert result["arm_side"] == "right"


@pytest.mark.parametrize("grasp_search_module", GRASP_SEARCH_MODULES)
def test_left_arm_is_selected_when_right_arm_is_unreachable(grasp_search_module):
    """物体偏左且右手不可达时选择左手。"""
    result = _search(_grasp_pose(0.25, 0.10, 0.30), grasp_search_module)
    assert result is not None
    assert result["arm_side"] == "left"


@pytest.mark.parametrize("grasp_search_module", GRASP_SEARCH_MODULES)
def test_allowed_arm_side_can_force_left_arm(grasp_search_module):
    """分阶段第二抓时可强制只搜索左手。"""
    result = _search(
        _reachable_grasp_pose(grasp_search_module, arm_side="left"),
        grasp_search_module,
        allowed_arm_side="left",
    )
    assert result is not None
    assert result["arm_side"] == "left"


@pytest.mark.parametrize("grasp_search_module", GRASP_SEARCH_MODULES)
def test_allowed_arm_side_can_force_right_arm(grasp_search_module):
    """第一抓需要时可强制只搜索右手。"""
    result = _search(
        _reachable_grasp_pose(grasp_search_module, arm_side="right"),
        grasp_search_module,
        allowed_arm_side="right",
    )
    assert result is not None
    assert result["arm_side"] == "right"


@pytest.mark.parametrize("grasp_search_module", GRASP_SEARCH_MODULES)
def test_candidate_requires_reachable_pregrasp(grasp_search_module):
    """最终抓取可达但预抓取距离过大时不得返回候选。"""
    result = _search(
        _reachable_grasp_pose(grasp_search_module),
        grasp_search_module,
        pregrasp_offset_min_m=1.0,
        pregrasp_offset_max_m=1.0,
        pregrasp_offset_samples=1,
    )
    assert result is None


@pytest.mark.parametrize("grasp_search_module", GRASP_SEARCH_MODULES)
def test_sample_cost_has_priority_over_arm_preference_with_multiple_workers(grasp_search_module):
    """更小移动 sample 的左手解应优先于后续 sample 的右手解。"""
    sample_transforms = np.repeat(np.eye(4, dtype=float)[None, ...], 2, axis=0)
    base_from_map = sample_transforms.copy()
    base_from_map[0, 1, 3] = 0.10
    base_from_map[1, 1, 3] = -0.10
    for _ in range(5):
        result = _search(
            _grasp_pose(0.25, 0.0, 0.30),
            grasp_search_module,
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
