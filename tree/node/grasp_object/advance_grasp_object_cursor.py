"""在夹爪成功闭合后消费当前抓取物体。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class AdvanceGraspObjectCursor(TimedMockAction):
    """将 multi 物体游标推进一位。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.objects_key = str(
            params.get("objects_key", "grasp_object_sorted_grasp_objects")
        ).strip()
        self.cursor_key = str(
            params.get("cursor_key", "grasp_object_next_grasp_object_index")
        ).strip()
        self.count = int(params.get("count", 1))
        if self.count < 1:
            raise ValueError("count 必须大于等于 1")
        self.blackboard.register_key(
            key=self.objects_key, access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=self.cursor_key, access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key=self.cursor_key, access=py_trees.common.Access.WRITE
        )

    def update(self):
        if not self.blackboard.exists(self.objects_key):
            self.feedback_message = f"物体缓存不存在: key={self.objects_key}"
            return Status.FAILURE
        objects = self.blackboard.get(self.objects_key)
        if not isinstance(objects, list) or not objects:
            self.feedback_message = "物体缓存为空"
            return Status.FAILURE
        cursor = 0
        if self.blackboard.exists(self.cursor_key):
            cursor = int(self.blackboard.get(self.cursor_key))
        if cursor < 0 or cursor >= len(objects):
            self.feedback_message = f"物体游标越界: index={cursor}, count={len(objects)}"
            return Status.FAILURE
        if cursor + self.count > len(objects):
            self.feedback_message = (
                f"物体游标推进越界: index={cursor}, count={self.count}, "
                f"object_count={len(objects)}"
            )
            return Status.FAILURE

        # 关键步骤：该节点放在 CloseClaw 成功之后，避免规划或运动失败时丢失物体。
        self.blackboard.set(self.cursor_key, cursor + self.count, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已消费 {self.count} 个缓存物体: "
            f"index={cursor}->{cursor + self.count}, object_count={len(objects)}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] AdvanceGraspObjectCursor start: "
            f"objects_key={self.objects_key}, cursor_key={self.cursor_key}, "
            f"count={self.count}"
        )
