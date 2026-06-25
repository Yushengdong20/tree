"""从 YOLO 目标队列中取出下一个待处理箱体。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class MoveBoxPopNextYoloTarget(TimedMockAction):
    """从 blackboard 队列弹出下一个箱体目标。"""

    allow_manual_result_override = False

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.target_queue_key = str(params.get("target_queue_key", "move_box_yolo_target_queue")).strip()
        self.current_target_key = str(params.get("current_target_key", "move_box_current_yolo_target")).strip()
        self.blackboard.register_key(key=self.target_queue_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.target_queue_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.current_target_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if not self.target_queue_key or not self.current_target_key:
            self.ros_node.get_logger().error(f"[{self.config_label}] 目标队列 key 配置为空")
            return Status.FAILURE

        targets = self.blackboard.get(self.target_queue_key) if self.blackboard.exists(self.target_queue_key) else []
        if not targets:
            self.ros_node.get_logger().info(f"[{self.config_label}] YOLO 目标队列已空")
            return Status.FAILURE

        # 关键步骤：每轮只消费队首目标，剩余目标留给下一轮 Repeat。
        current_target = targets[0]
        remaining_targets = list(targets[1:])
        self.blackboard.set(self.current_target_key, current_target, overwrite=True)
        self.blackboard.set(self.target_queue_key, remaining_targets, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 取出下一个 YOLO 箱体目标: "
            f"id={current_target.get('id')}, remaining={len(remaining_targets)}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return f"[{self.config_label}] MoveBoxPopNextYoloTarget start"
