"""基于 blackboard 中的末端位姿计算相对偏移后的末端位姿。"""

import ast

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class ComputeRelativeEndEffectorPose(TimedMockAction):
    """读取 6 维末端位姿，叠加同坐标系下的相对偏移后写回 blackboard。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.input_pose_key = str(params.get("input_pose_key", "")).strip()
        self.output_pose_key = str(params.get("output_pose_key", "")).strip()
        self.dx = float(params.get("dx", 0.0))
        self.dy = float(params.get("dy", 0.0))
        self.dz = float(params.get("dz", 0.0))
        self.dyaw = float(params.get("dyaw", 0.0))
        self.dpitch = float(params.get("dpitch", 0.0))
        self.droll = float(params.get("droll", 0.0))
        if not self.input_pose_key:
            raise ValueError("input_pose_key 不能为空")
        if not self.output_pose_key:
            raise ValueError("output_pose_key 不能为空")

        self.blackboard = py_trees.blackboard.Client(name=name)
        self.blackboard.register_key(key=self.input_pose_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.output_pose_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        try:
            if not self.blackboard.exists(self.input_pose_key):
                raise RuntimeError(f"输入末端位姿不存在: key={self.input_pose_key}")
            input_pose = self._parse_pose(self.blackboard.get(self.input_pose_key))
            # 关键步骤：这里按 ArmsToPose 使用的 [x, y, z, yaw, pitch, roll] 语义做同坐标系数值偏移。
            output_pose = [
                input_pose[0] + self.dx,
                input_pose[1] + self.dy,
                input_pose[2] + self.dz,
                input_pose[3] + self.dyaw,
                input_pose[4] + self.dpitch,
                input_pose[5] + self.droll,
            ]
            self.blackboard.set(self.output_pose_key, output_pose, overwrite=True)
        except Exception as exc:
            self.feedback_message = str(exc)
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 计算相对末端位姿失败: {exc}"
            )
            return Status.FAILURE

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算相对末端位姿: "
            f"input_key={self.input_pose_key}, input={input_pose}, "
            f"offset=({self.dx:.3f}, {self.dy:.3f}, {self.dz:.3f}, "
            f"{self.dyaw:.3f}, {self.dpitch:.3f}, {self.droll:.3f}), "
            f"output_key={self.output_pose_key}, output={output_pose}"
        )
        return Status.SUCCESS

    @staticmethod
    def _parse_pose(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError("末端位姿必须是长度为 6 的列表: [x, y, z, yaw, pitch, roll]")
        return [float(item) for item in value]

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeRelativeEndEffectorPose start: "
            f"input_key={self.input_pose_key}, output_key={self.output_pose_key}, "
            f"offset=({self.dx:.3f}, {self.dy:.3f}, {self.dz:.3f}, "
            f"{self.dyaw:.3f}, {self.dpitch:.3f}, {self.droll:.3f})"
        )
