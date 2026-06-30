"""放箱 HTTP 节点。

这个节点在真实模式下调用 `/put_down`，
在手动模式下则继续走统一的 mock/manual 结果逻辑。

当前会经历的分解动作如下：

1. REQUEST
   先进入一个显式的 RUNNING 帧，告诉外部“即将发起放箱请求”。

2. EXECUTE
   真正调用 put_down，并把返回结果写入 blackboard。
"""

from py_trees import common

from tree.constants import FLOW_RESULT_KEY
from ..base import TimedMockAction
from tree.runtime.http.move_and_grab_flow import (
    DEFAULT_GRAB_URL,
    build_grab_config,
    post_put_down,
)


class HttpPutDownBox(TimedMockAction):
    """放箱异步节点。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        # 放箱节点会在已有流程结果上追加 putDown 结果。
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=common.Access.READ)
        self.blackboard.register_key(key=FLOW_RESULT_KEY, access=common.Access.WRITE)
        self.grab_config = build_grab_config(
            base_url=str(params.get("grab_url", DEFAULT_GRAB_URL)).strip(),
            arm=params.get("arm"),
        )
        self._phase = "IDLE"

    def initialise(self):
        super().initialise()
        # 和 grab_box 一样，先进入一个显式的 RUNNING 阶段再真正访问接口。
        self._phase = "REQUEST"

    def update(self):
        if self.should_use_mock_execution():
            status = self.update_mock_result()
            if status == common.Status.SUCCESS:
                self.blackboard.flow_result["putDown"] = {
                    "simulated": True,
                    "arm": self.grab_config.arm,
                }
            return status

        try:
            if self._phase == "REQUEST":
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "PUT_DOWN",
                    f"Ready to call put_down for arm={self.grab_config.arm or 'default'}",
                )
                self._phase = "EXECUTE"
                return common.Status.RUNNING

            if self._phase == "EXECUTE":
                self.ros_node.set_live_runtime(
                    self.config_label,
                    "PUT_DOWN",
                    f"Calling put_down for arm={self.grab_config.arm or 'default'}",
                )
                result = post_put_down(self.grab_config)
                self.blackboard.flow_result["putDown"] = result
                self.ros_node.clear_live_runtime()
                return common.Status.SUCCESS

            raise RuntimeError(f"unsupported put_down phase: {self._phase}")
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.clear_live_runtime()
            self.ros_node.get_logger().error(f"[{self.config_label}] put_down failed: {exc}")
            return common.Status.FAILURE

    def describe_start(self):
        return f"[{self.config_label}] HttpPutDownBox start"
