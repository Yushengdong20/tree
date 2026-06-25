"""导航示例节点。

这个节点不是 HTTP 版真实导航，而是一个简化的 mock 示例：
- 只保存目标位姿参数
- 依赖基类统一处理手动结果或固定结果
"""

from ..base import TimedMockAction


class MoveToTarget(TimedMockAction):
    """示例导航目标节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 示例里把目标位姿直接从 JSON 读进来，真实项目里通常会在 update 中发导航指令。
        self.x = float(params.get("x", 0.0))
        self.y = float(params.get("y", 0.0))
        self.yaw = float(params.get("yaw", 0.0))

    def describe_start(self):
        # 这里只打印目标位姿，真实版本通常还会带导航任务 id、速度限制等信息。
        return (
            f"[{self.config_label}] MoveToTarget start: "
            f"x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f}"
        )
