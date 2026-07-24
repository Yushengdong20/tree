"""同步双抓流程的轻量配置和游标测试。"""

import json
from pathlib import Path

import py_trees
from py_trees.common import Status

from tree.core.tree_factory import ParamsWrapper
from tree.node.grasp_object.advance_grasp_object_cursor import AdvanceGraspObjectCursor


class _Logger:
    """测试用日志对象，兼容节点里调用的 info/error 接口。"""

    def info(self, _message):
        pass

    def error(self, _message):
        pass


class _RosNode:
    """测试用最小 ros_node。"""

    def get_logger(self):
        return _Logger()


def _params(**values):
    """构造行为树节点使用的只读参数对象。"""
    return ParamsWrapper(values)


def test_all_grasp_object_json_files_are_valid_without_cpp_extension():
    """同步双抓新增子树不应引入 JSON 语法错误。"""
    config_root = Path(__file__).parents[1] / "config" / "tree" / "grasp_object"
    files = sorted(config_root.rglob("*.json"))
    assert files
    for config_file in files:
        with config_file.open("r", encoding="utf-8") as handle:
            json.load(handle)


def test_advance_grasp_object_cursor_can_consume_two_objects():
    """同步双抓闭爪成功后一次消费两个物体。"""
    objects_key = "test_dual_cursor_objects"
    cursor_key = "test_dual_cursor_index"
    blackboard = py_trees.blackboard.Client(name="test_dual_cursor_seed")
    blackboard.register_key(key=objects_key, access=py_trees.common.Access.WRITE)
    blackboard.register_key(key=cursor_key, access=py_trees.common.Access.WRITE)
    blackboard.set(objects_key, [{"id": 1}, {"id": 2}, {"id": 3}], overwrite=True)
    blackboard.set(cursor_key, 0, overwrite=True)

    node = AdvanceGraspObjectCursor(
        name="AdvanceGraspObjectCursor",
        config_label="测试同步双抓游标",
        ros_node=_RosNode(),
        params=_params(objects_key=objects_key, cursor_key=cursor_key, count=2),
    )

    assert node.update() == Status.SUCCESS
    assert blackboard.get(cursor_key) == 2


def test_advance_grasp_object_cursor_rejects_count_overflow():
    """剩余物体不足两个时不能误推进游标。"""
    objects_key = "test_dual_cursor_overflow_objects"
    cursor_key = "test_dual_cursor_overflow_index"
    blackboard = py_trees.blackboard.Client(name="test_dual_cursor_overflow_seed")
    blackboard.register_key(key=objects_key, access=py_trees.common.Access.WRITE)
    blackboard.register_key(key=cursor_key, access=py_trees.common.Access.WRITE)
    blackboard.set(objects_key, [{"id": 1}], overwrite=True)
    blackboard.set(cursor_key, 0, overwrite=True)

    node = AdvanceGraspObjectCursor(
        name="AdvanceGraspObjectCursor",
        config_label="测试同步双抓游标越界",
        ros_node=_RosNode(),
        params=_params(objects_key=objects_key, cursor_key=cursor_key, count=2),
    )

    assert node.update() == Status.FAILURE
    assert blackboard.get(cursor_key) == 0
