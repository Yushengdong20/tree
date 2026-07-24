"""grasp_and_place HTTP service 任务测试。"""

from pathlib import Path
import sys
import types

import py_trees
import pytest
from py_trees.common import Status

from tree.core.tree_factory import BehaviorTreeFactory, ParamsWrapper
from tree.node.common.check_blackboard_number import CheckBlackboardNumber
from tree.node.common.repeat_until_blackboard_number import RepeatUntilBlackboardNumber
from tree.runtime.http_service.models import GraspAndPlaceRequestModel
from tree.runtime.http_service.task_adapters.grasp_and_place import (
    FIXED_KNEE_Z_FROM_BASE_M,
    GraspAndPlaceTaskAdapter,
    TASK_INPUT_KEYS,
    TORSO_SAMPLE_GRASP_HEIGHT_WINDOW_M,
)


class _Logger:
    """测试用日志对象。"""

    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _RosNode:
    """测试用最小 ros_node。"""

    manual_result_mode = True

    def __init__(self):
        self.config = type("Config", (), {"manual_result_mode": True})()
        self.logger = _Logger()

    def get_logger(self):
        return self.logger

    def get_param(self, _key, default=None):
        return default

    def create_message_subscription(self, *_args, **_kwargs):
        return None


def _params(**values):
    """构造行为树节点使用的只读参数对象。"""
    return ParamsWrapper(values)


def _seed_blackboard(values):
    """向 blackboard 写入测试数据。"""
    blackboard = py_trees.blackboard.Client(name="test_grasp_and_place_seed")
    for key, value in values.items():
        blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
        blackboard.set(key, value, overwrite=True)
    return blackboard


def _install_tree_load_stubs():
    """安装行为树加载测试所需的外部依赖最小替身。"""
    module_names = [
        "kuavo_humanoid_sdk",
        "kuavo_humanoid_sdk.kuavo_strategy_v2",
        "kuavo_humanoid_sdk.kuavo_strategy_v2.common",
        "kuavo_humanoid_sdk.kuavo_strategy_v2.common.events",
        "kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate",
    ]
    for module_name in module_names:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    ik_module_name = (
        "kuavo_humanoid_sdk.kuavo_strategy_v2.common.events."
        "mobile_manipulate.ik_library"
    )
    ik_module = sys.modules.setdefault(ik_module_name, types.ModuleType(ik_module_name))
    ik_module.IK_MODEL_GRASP = "grasp"

    rospy_module = sys.modules.setdefault("rospy", types.ModuleType("rospy"))
    rospy_module.Time = object

    tf_module = sys.modules.setdefault("tf", types.ModuleType("tf"))
    transformations_module = sys.modules.setdefault(
        "tf.transformations",
        types.ModuleType("tf.transformations"),
    )
    transformations_module.euler_from_quaternion = lambda _value: (0.0, 0.0, 0.0)
    tf_module.transformations = transformations_module

    sys.modules.setdefault("tf2_ros", types.ModuleType("tf2_ros"))

    geometry_msgs_module = sys.modules.setdefault(
        "geometry_msgs",
        types.ModuleType("geometry_msgs"),
    )
    geometry_msgs_msg_module = sys.modules.setdefault(
        "geometry_msgs.msg",
        types.ModuleType("geometry_msgs.msg"),
    )
    geometry_msgs_msg_module.PointStamped = type("PointStamped", (), {})
    geometry_msgs_msg_module.PoseStamped = type("PoseStamped", (), {})
    geometry_msgs_module.msg = geometry_msgs_msg_module

    nav_msgs_module = sys.modules.setdefault("nav_msgs", types.ModuleType("nav_msgs"))
    nav_msgs_msg_module = sys.modules.setdefault(
        "nav_msgs.msg",
        types.ModuleType("nav_msgs.msg"),
    )
    nav_msgs_msg_module.Odometry = type("Odometry", (), {})
    nav_msgs_module.msg = nav_msgs_msg_module

    std_msgs_module = sys.modules.setdefault("std_msgs", types.ModuleType("std_msgs"))
    std_msgs_msg_module = sys.modules.setdefault(
        "std_msgs.msg",
        types.ModuleType("std_msgs.msg"),
    )
    std_msgs_msg_module.Header = type("Header", (), {})
    std_msgs_module.msg = std_msgs_msg_module


def test_grasp_and_place_request_model_normalizes_numbers():
    """HTTP 请求模型应接受数字字符串并归一化。"""
    model = GraspAndPlaceRequestModel.model_validate(
        {
            "naviPoseGrasp": {"x": "1.0", "y": "2.0", "yaw": "0.0"},
            "naviPosePlace": {"x": 3.0, "y": 4.0, "yaw": 1.57},
            "heightGraspPlane": "0.8",
            "heightPlacePlane": "0.9",
            "targetCount": "3",
        }
    )

    assert model.naviPoseGrasp.x == 1.0
    assert model.heightGraspPlane == 0.8
    assert model.heightPlacePlane == 0.9
    assert model.targetCount == 3


def test_grasp_and_place_request_model_rejects_invalid_values():
    """HTTP 请求模型应拒绝无效数量和 bool 数值字段。"""
    payload = {
        "naviPoseGrasp": {"x": 1.0, "y": 2.0, "yaw": 0.0},
        "naviPosePlace": {"x": 3.0, "y": 4.0, "yaw": 1.57},
        "heightGraspPlane": 0.8,
        "heightPlacePlane": 0.9,
        "targetCount": 0,
    }
    with pytest.raises(ValueError):
        GraspAndPlaceRequestModel.model_validate(payload)

    payload["targetCount"] = 1
    payload["heightPlacePlane"] = True
    with pytest.raises(ValueError):
        GraspAndPlaceRequestModel.model_validate(payload)


def test_grasp_and_place_adapter_writes_blackboard():
    """adapter 应把 HTTP 请求写入 service tree 使用的 blackboard key。"""
    adapter = GraspAndPlaceTaskAdapter()
    request = adapter.parse_payload(
        {
            "naviPoseGrasp": {"x": 1.0, "y": 2.0, "yaw": 0.0},
            "naviPosePlace": {"x": 3.0, "y": 4.0, "yaw": 1.57},
            "heightGraspPlane": 0.8,
            "heightPlacePlane": 0.9,
            "targetCount": 5,
        }
    )

    adapter.write_blackboard("task-1", request)

    assert adapter.blackboard.get(TASK_INPUT_KEYS["task_id"]) == "task-1"
    assert adapter.blackboard.get(TASK_INPUT_KEYS["done_count"]) == 0
    assert adapter.blackboard.get(TASK_INPUT_KEYS["target_count"]) == 5
    assert adapter.blackboard.get(TASK_INPUT_KEYS["height_grasp_plane"]) == 0.8
    assert adapter.blackboard.get(TASK_INPUT_KEYS["height_place_plane"]) == 0.9
    assert adapter.blackboard.get(TASK_INPUT_KEYS["torso_sample_z_min"]) == pytest.approx(
        0.8 - FIXED_KNEE_Z_FROM_BASE_M
    )
    assert adapter.blackboard.get(TASK_INPUT_KEYS["torso_sample_z_max"]) == pytest.approx(
        0.8 - FIXED_KNEE_Z_FROM_BASE_M + TORSO_SAMPLE_GRASP_HEIGHT_WINDOW_M
    )
    assert adapter.blackboard.get(TASK_INPUT_KEYS["sorted_grasp_objects"]) == []
    assert adapter.blackboard.get(TASK_INPUT_KEYS["next_grasp_object_index"]) == 0


def test_check_blackboard_number_supports_required_operators():
    """数值比较节点应支持 >= 和 < 两类 service tree 分支判断。"""
    _seed_blackboard({"test_done_count": 3, "test_target_count": 3})
    ge_node = CheckBlackboardNumber(
        name="CheckBlackboardNumber",
        config_label="测试数量达到",
        ros_node=_RosNode(),
        params=_params(
            left_key="test_done_count",
            operator=">=",
            right_key="test_target_count",
        ),
    )
    assert ge_node.update() == Status.SUCCESS

    _seed_blackboard({"test_done_count_lt": 2, "test_target_count_lt": 3})
    lt_node = CheckBlackboardNumber(
        name="CheckBlackboardNumber",
        config_label="测试数量未达到",
        ros_node=_RosNode(),
        params=_params(
            left_key="test_done_count_lt",
            operator="<",
            right_key="test_target_count_lt",
        ),
    )
    assert lt_node.update() == Status.SUCCESS


def test_check_blackboard_number_rejects_missing_or_non_numeric_values():
    """数值比较节点遇到缺 key 或非数字值时返回 FAILURE。"""
    missing_node = CheckBlackboardNumber(
        name="CheckBlackboardNumber",
        config_label="测试缺少数值",
        ros_node=_RosNode(),
        params=_params(
            left_key="test_missing_left",
            operator=">=",
            right_value=1,
        ),
    )
    assert missing_node.update() == Status.FAILURE

    _seed_blackboard({"test_non_number": "abc"})
    non_number_node = CheckBlackboardNumber(
        name="CheckBlackboardNumber",
        config_label="测试非数字值",
        ros_node=_RosNode(),
        params=_params(
            left_key="test_non_number",
            operator=">=",
            right_value=1,
        ),
    )
    assert non_number_node.update() == Status.FAILURE


class _SetNumberBehaviour(py_trees.behaviour.Behaviour):
    """测试用：第一次 tick 后写入目标数值。"""

    def __init__(self, key, value):
        super().__init__(name="SetNumberBehaviour")
        self.key = key
        self.value = value
        self.blackboard = py_trees.blackboard.Client(name="test_repeat_setter")
        self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    def update(self):
        self.blackboard.set(self.key, self.value, overwrite=True)
        return Status.SUCCESS


def test_repeat_until_blackboard_number_stops_when_target_reached():
    """动态循环节点应在子树把完成数更新到目标后返回 SUCCESS。"""
    _seed_blackboard({"test_repeat_done": 0, "test_repeat_target": 2})
    child = _SetNumberBehaviour("test_repeat_done", 2)
    node = RepeatUntilBlackboardNumber(
        name="RepeatUntilBlackboardNumber",
        child=child,
        config_label="测试动态循环",
        ros_node=_RosNode(),
        params=_params(
            left_key="test_repeat_done",
            operator=">=",
            right_key="test_repeat_target",
        ),
    )

    list(node.tick())

    assert node.status == Status.SUCCESS


def test_service_grasp_and_place_json_loads():
    """service 版 grasp_and_place 行为树应能被工厂加载。"""
    _install_tree_load_stubs()
    config_file = (
        Path(__file__).parents[1]
        / "config"
        / "tree"
        / "service"
        / "grasp_object"
        / "start_grasp_and_place.json"
    )

    tree = BehaviorTreeFactory(_RosNode()).load_tree_from_json(str(config_file))

    assert tree.root.name == "HTTP grasp_and_place 单次任务流程"
