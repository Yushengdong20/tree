"""Generic fake leaf node for ROS1/ROS2 smoke tests."""

from py_trees import common

from ..base import TimedMockAction


class FakeStep(TimedMockAction):
    """A configurable fake node that can stay RUNNING for a few ticks before finishing."""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.message = str(params.get("message", "fake step")).strip()
        self.running_ticks = max(0, int(params.get("running_ticks", 0)))
        self._ticks = 0

    def initialise(self):
        super().initialise()
        self._ticks = 0

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        if self._ticks < self.running_ticks:
            self._ticks += 1
            self.ros_node.get_logger().info(
                f"[{self.config_label}] FakeStep running "
                f"{self._ticks}/{self.running_ticks}: {self.message}"
            )
            return common.Status.RUNNING

        return self.update_mock_result()

    def describe_start(self):
        return (
            f"[{self.config_label}] FakeStep start: "
            f"message={self.message}, running_ticks={self.running_ticks}, result={self.result}"
        )
