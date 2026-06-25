"""夹爪示例节点。

这个节点当前主要用于演示：
- JSON 参数如何映射到节点
- mock/manual 模式下树如何推进

它本身不驱动真实硬件。
"""

from ..base import TimedMockAction


class MoveClaw(TimedMockAction):
    """示例夹爪动作节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 这里保留字符串格式，便于和原始 JSON 配置保持一致，日志里也更直观。
        self.target_positions = str(params.get("target_positions", "0.0,0.0"))
        self.velocity = str(params.get("velocity", "50,50"))
        self.torque = str(params.get("torque", "1.0,1.0"))

    def describe_start(self):
        # 夹爪示例节点同样只模拟“准备执行”的日志，不直接驱动真实硬件。
        return (
            f"[{self.config_label}] MoveClaw start: "
            f"target_positions={self.target_positions}, "
            f"velocity={self.velocity}, torque={self.torque}"
        )
