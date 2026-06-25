"""头部示例节点。

这个节点是最轻量的 mock 叶子节点之一，适合用来理解：
- JSON 参数读取
- describe_start/finish 日志
- base.py 里的手动结果控制
"""

from ..base import TimedMockAction


class MoveHead(TimedMockAction):
    """示例头部运动节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 头部偏航/俯仰角同样只是示例参数，当前实现主要用于演示行为树调度。
        self.yaw = float(params.get("yaw", 0.0))
        self.pitch = float(params.get("pitch", 0.0))

    def describe_start(self):
        # describe_start/finish 只负责生成日志文本，不参与状态机逻辑。
        return (
            f"[{self.config_label}] MoveHead start: "
            f"yaw={self.yaw:.3f}, pitch={self.pitch:.3f}"
        )
