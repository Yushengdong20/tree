"""外部手动结果发送工具。

这个文件提供一个独立的小 ROS 节点，用来给正在运行的行为树发送：
- s / f / r
- 或带节点名的 `<node_label> s/f/r`

它常用于：
- 主程序不直接开终端输入时
- 希望把“跑树”和“喂结果”分到两个终端里
"""

import argparse

from tree.ros_interface import create_ros_interface


class ManualResultSender:
    """Publish manual result commands for a running behaviour tree node.

    这个工具节点的定位不是“跑树”，而是单纯充当一个外部测试控制台。
    """

    def __init__(self, ros, topic: str, waiting_nodes_topic: str):
        self.ros = ros
        self.topic = topic
        self.waiting_nodes_topic = waiting_nodes_topic
        self.publisher = self.ros.create_string_publisher(topic, queue_size=10)
        self.waiting_nodes = []
        self.waiting_nodes_subscriber = self.ros.create_string_subscription(
            waiting_nodes_topic,
            self._on_waiting_nodes_message,
            latch=True,
        )

    def send(self, command: str):
        """Publish one command using the same syntax as the runner's terminal prompt."""
        self.publisher.publish(command)

    def _on_waiting_nodes_message(self, data: str):
        # 收到 runner 发布的等待列表后，直接在当前终端打印，方便只盯着一个窗口操作。
        waiting = [item for item in data.split(",") if item]
        if waiting == self.waiting_nodes:
            return
        self.waiting_nodes = waiting
        text = ", ".join(waiting) if waiting else "none"
        print(f"\nwaiting nodes [{self.waiting_nodes_topic}]: {text}", flush=True)

    def prompt_text(self) -> str:
        """Build a prompt that reflects how specific the next input needs to be."""
        # 一个等待节点时，用户直接输 s/f/r 就够了；
        # 多个等待节点时，提示里显式提醒需要带 node label。
        if len(self.waiting_nodes) == 1:
            return f"send result for [{self.waiting_nodes[0]}] "
        if len(self.waiting_nodes) > 1:
            return "send result with node label "
        return f"send manual result [{self.topic}] "

    def start_background_spin(self):
        """Keep ROS callbacks flowing even while the terminal blocks on input()."""
        # input() 会阻塞主线程，所以 ROS 回调要放到后台线程持续处理。
        self.ros.start_background_spin()


def build_argument_parser():
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Send manual behaviour-tree results to mercurytree"
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="One-shot command, e.g. 's' or 'approach_zone_a f'",
    )
    parser.add_argument(
        "--topic",
        default="/pytrees_ros2_runner/manual_result",
        help="Target topic used by the runner",
    )
    parser.add_argument(
        "--waiting-topic",
        default="/pytrees_ros2_runner/waiting_nodes",
        help="Topic that publishes the current waiting node list",
    )
    return parser


def main(args=None):
    """Run once with a command argument or start an interactive publish loop."""
    # 这是一个“树外部的小工具”：
    # 不负责建树，不负责 tick，只负责把用户输入转成 topic 消息发给 runner。
    parser = build_argument_parser()
    parsed = parser.parse_args(args=args)

    ros = create_ros_interface(node_name="pytrees_ros2_manual_result_sender")
    ros.init(args=args)
    node = ManualResultSender(
        ros=ros,
        topic=parsed.topic,
        waiting_nodes_topic=parsed.waiting_topic,
    )
    try:
        node.start_background_spin()
        if parsed.command:
            node.send(" ".join(parsed.command).strip())
            return

        print(
            "Interactive manual-result sender started. "
            "Input: s/f/r or '<node_label> s/f/r'. Ctrl-D to quit.",
            flush=True,
        )
        while ros.ok():
            try:
                raw = input(f"{node.prompt_text()}> ").strip()
            except EOFError:
                print()
                break
            if not raw:
                continue
            node.send(raw)
    finally:
        ros.destroy_node()
        ros.shutdown()


if __name__ == "__main__":
    main()
