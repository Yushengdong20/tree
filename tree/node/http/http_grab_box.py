"""抓箱 HTTP 节点。

这个节点在真实模式下会调用 `/grab_box`，
在手动模式下则退化成一个等待 s/f/r 的可控节点。

当前会经历的分解动作如下：

1. REQUEST
   先进入一个显式的 RUNNING 帧，告诉外部“即将发起抓箱请求”。

2. EXECUTE
   真正调用 grab_box，并把返回结果写入 blackboard。
"""

from py_trees import common

from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_GRAB_URL,
    build_grab_config,
    post_grab_box,
)


class HttpGrabBox(TimedMockAction):
    """抓箱异步节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 抓箱节点会在已有流程结果上追加 grab 结果。
        self.blackboard.register_key(key="flow_result", access=common.Access.READ)
        self.blackboard.register_key(key="flow_result", access=common.Access.WRITE)
        self.grab_config = build_grab_config(
            base_url=str(params.get("grab_url", DEFAULT_GRAB_URL)).strip(),
            arm=params.get("arm"),
        )
        self._phase = "IDLE"

    def initialise(self):
        super().initialise()
        # 先切到 REQUEST，让树至少经历一个可观测的 RUNNING 帧。
        self._phase = "REQUEST"

    def update(self):
        if self.should_use_mock_execution():
            status = self.update_mock_result()
            if status == common.Status.SUCCESS:
                self.blackboard.flow_result["grab"] = {
                    "simulated": True,
                    "arm": self.grab_config.arm,
                }
            return status

        try:
            if self._phase == "REQUEST":
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "GRAB",
                    f"Ready to call grab_box for arm={self.grab_config.arm or 'default'}",
                )
                self._phase = "EXECUTE"
                return common.Status.RUNNING

            if self._phase == "EXECUTE":
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "GRAB",
                    f"Calling grab_box for arm={self.grab_config.arm or 'default'}",
                )
                result = post_grab_box(self.grab_config)
                self.blackboard.flow_result["grab"] = result
                self.ros_node.clear_live_runtime()
                return common.Status.SUCCESS

            raise RuntimeError(f"unsupported grab phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] grab failed: {exc}")
            return common.Status.FAILURE

    def describe_start(self):
        return f"[{self.config_label}] HttpGrabBox start"
