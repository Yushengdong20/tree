"""navigation HTTP service 任务测试。"""

from pathlib import Path
import sys
import types

import pytest

from tree.core.tree_factory import BehaviorTreeFactory
from tree.runtime.http_service.models import NavigationRequestModel
from tree.runtime.http_service.task_adapters.navigation import (
    NavigationTaskAdapter,
    TASK_INPUT_KEYS,
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

    def create_publisher(self, *_args, **_kwargs):
        return None


def _install_tree_load_stubs():
    """安装行为树加载测试所需的外部依赖最小替身。"""
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
    geometry_msgs_msg_module.Point = type("Point", (), {})
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

    visualization_msgs_module = sys.modules.setdefault(
        "visualization_msgs",
        types.ModuleType("visualization_msgs"),
    )
    visualization_msgs_msg_module = sys.modules.setdefault(
        "visualization_msgs.msg",
        types.ModuleType("visualization_msgs.msg"),
    )
    visualization_msgs_msg_module.Marker = type(
        "Marker",
        (),
        {"ADD": 0, "DELETEALL": 3, "ARROW": 0, "SPHERE": 2, "LINE_LIST": 5, "TEXT_VIEW_FACING": 9},
    )
    visualization_msgs_msg_module.MarkerArray = type("MarkerArray", (), {})
    visualization_msgs_module.msg = visualization_msgs_msg_module


def test_navigation_request_model_normalizes_numbers():
    """HTTP 请求模型应接受 targetPose 数字字符串并归一化。"""
    model = NavigationRequestModel.model_validate(
        {
            "targetPose": {
                "x": "1.0",
                "y": "2.0",
                "yaw": "0.5",
            }
        }
    )

    assert model.targetPose.x == 1.0
    assert model.targetPose.y == 2.0
    assert model.targetPose.yaw == 0.5


def test_navigation_request_model_rejects_invalid_pose_values():
    """HTTP 请求模型应拒绝 bool 和非数字位姿字段。"""
    with pytest.raises(ValueError):
        NavigationRequestModel.model_validate(
            {"targetPose": {"x": True, "y": 2.0, "yaw": 0.5}}
        )

    with pytest.raises(ValueError):
        NavigationRequestModel.model_validate(
            {"targetPose": {"x": 1.0, "y": "bad", "yaw": 0.5}}
        )


def test_navigation_adapter_writes_blackboard():
    """adapter 应把 HTTP 请求写入 service tree 使用的 blackboard key。"""
    adapter = NavigationTaskAdapter()
    request = adapter.parse_payload(
        {
            "targetPose": {
                "x": 1.0,
                "y": 2.0,
                "yaw": 0.5,
            }
        }
    )

    adapter.write_blackboard("task-nav-1", request)

    assert adapter.blackboard.get(TASK_INPUT_KEYS["task_id"]) == "task-nav-1"
    assert adapter.blackboard.get(TASK_INPUT_KEYS["target_pose"]) == {
        "x": 1.0,
        "y": 2.0,
        "yaw": 0.5,
    }


def test_service_navigation_json_loads():
    """service 版 navigation 行为树应能被工厂加载。"""
    _install_tree_load_stubs()
    config_file = (
        Path(__file__).parents[1]
        / "config"
        / "tree"
        / "service"
        / "navigation"
        / "start_navigation.json"
    )

    tree = BehaviorTreeFactory(_RosNode()).load_tree_from_json(str(config_file))

    assert tree.root.name == "HTTP navigation 单次任务流程"
