"""Blackboard 启动初始化辅助函数。"""

import json
import os

import py_trees


def load_blackboard_from_json(ros, blackboard_json_file):
    """启动时从 blackboard.json 读取共享参数，并写入全局 py_trees blackboard。"""
    if not os.path.exists(blackboard_json_file):
        ros.get_logger().info(
            f"Blackboard bootstrap skipped, file not found: {blackboard_json_file}"
        )
        return

    with open(blackboard_json_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("blackboard.json 顶层必须是 object/dict")

    loaded_keys = []
    for key, value in payload.items():
        py_trees.blackboard.Blackboard.set(key, value)
        loaded_keys.append(key)

    ros.get_logger().info(
        f"Loaded blackboard bootstrap: {blackboard_json_file}, keys={loaded_keys}"
    )
