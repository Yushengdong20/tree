"""切换 FoundationPose 当前识别模型。"""

import py_trees
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY
from tree.utils.chassis_navigation import post_json

from ..base import TimedMockAction


DEFAULT_FOUNDATIONPOSE_BASE_URL = "http://192.168.26.12:13604"


class SwitchFoundationPoseTarget(TimedMockAction):
    """通过 HTTP 切换 FP 的 mesh，并记录当前业务目标类型。

    例如拆垛前切到 ``tuopan`` 估计托盘；完成托盘区域标定后再切回
    ``large_green``，供正常箱体检测与抓取使用。该节点只切换模型，是否 reset
    和等待新帧由后续 ``EnsureMoveBoxDetectionReady`` 明确负责，避免隐式重启。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.base_url = str(
            params.get("foundationpose_base_url", DEFAULT_FOUNDATIONPOSE_BASE_URL)
        ).rstrip("/")
        self.mesh_name = str(params.get("mesh_name", "")).strip()
        self.timeout_sec = float(params.get("timeout_sec", 5.0))
        self.target_key = str(
            params.get("target_key", "move_box_foundationpose_active_target")
        ).strip()
        if not self.mesh_name:
            raise ValueError("SwitchFoundationPoseTarget 必须配置 mesh_name")
        self.blackboard.register_key(key=ROBOT_SERVICES_KEY, access=py_trees.common.Access.READ)
        if self.target_key:
            self.blackboard.register_key(key=self.target_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            if self.target_key:
                self.blackboard.set(self.target_key, self.mesh_name, overwrite=True)
            return self.update_mock_result()
        url = f"{self.base_url}/foundationpose/switch_target"
        try:
            result = post_json(
                url,
                {"mesh_name": self.mesh_name},
                self.timeout_sec,
                "foundationpose_switch_target",
            )
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.get_logger().error(
                f"[{self.config_label}] FP 切换识别目标失败: mesh={self.mesh_name}, "
                f"url={url}, error={exc}"
            )
            return Status.FAILURE
        if self.target_key:
            self.blackboard.set(self.target_key, self.mesh_name, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] FP 已切换识别目标: mesh={self.mesh_name}, "
            f"url={url}, response={result}"
        )
        return Status.SUCCESS

    def describe_start(self):
        return (
            f"[{self.config_label}] SwitchFoundationPoseTarget start: "
            f"mesh={self.mesh_name}, base_url={self.base_url}"
        )
