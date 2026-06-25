# -*- coding: utf-8 -*-
"""MoveClaw 节点。"""

import time

from py_trees.common import Status

from kuavo_msgs.msg import lejuClawCommand

from ..base import TimedMockAction


class MoveClaw(TimedMockAction):
    """真实夹爪控制节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super(MoveClaw, self).__init__(name, config_label, ros_node, params)
        self.target_positions = self._parse_number_pair(
            params.get("target_positions", "0.0,0.0"),
            default=(0.0, 0.0),
        )
        self.velocity = self._parse_number_pair(
            params.get("velocity", "80,80"),
            default=(80.0, 80.0),
        )
        self.torque = self._parse_number_pair(
            params.get("torque", "1.0,1.0"),
            default=(1.0, 1.0),
        )
        self.topic_name = str(params.get("topic_name", "/leju_claw_command"))
        self.publish_count = int(params.get("publish_count", 20))
        self.publish_interval_sec = float(params.get("publish_interval_sec", 0.1))
        self.success = False
        self.publisher = None

    def describe_start(self):
        return (
            f"[{self.config_label}] MoveClaw start: "
            f"target_positions={self.target_positions}, "
            f"velocity={self.velocity}, torque={self.torque}, "
            f"topic={self.topic_name}"
        )

    def initialise(self):
        super(MoveClaw, self).initialise()
        self.success = False

        if self.should_use_mock_execution():
            return

        if self.should_skip_claw_motion():
            self.log_skip_claw_motion()
            self.success = True
            return

        try:
            if self.publisher is None:
                self.publisher = self.ros_node.create_publisher(
                    self.topic_name,
                    lejuClawCommand,
                    queue_size=10,
                )

            self._publish_claw_command(
                pos=self.target_positions,
                vel=self.velocity,
                effort=self.torque,
            )
            self.success = True
        except Exception as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 移动夹爪失败: {exc}"
            )
            self.success = False

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if self.success:
            self.ros_node.get_logger().info(self.describe_finish(Status.SUCCESS))
            return Status.SUCCESS

        self.ros_node.get_logger().info(self.describe_finish(Status.FAILURE))
        return Status.FAILURE

    def _publish_claw_command(self, pos, vel, effort):
        msg = lejuClawCommand()
        msg.data.name = ["left_claw", "right_claw"]
        msg.data.position = list(pos)
        msg.data.velocity = list(vel)
        msg.data.effort = list(effort)

        for _ in range(self.publish_count):
            self.publisher.publish(msg)
            time.sleep(self.publish_interval_sec)

    @staticmethod
    def _parse_number_pair(raw_value, default):
        if isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
            return [float(raw_value[0]), float(raw_value[1])]

        text = str(raw_value).strip()
        if not text:
            return [float(default[0]), float(default[1])]

        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) != 2:
            raise ValueError(f"需要两个逗号分隔的数值，当前是: {raw_value}")
        return [float(parts[0]), float(parts[1])]
