"""根据箱体中心计算躯干目标位姿，并写入 blackboard。

本节点只负责计算，不直接控制腰部；后续由 MoveTorsoPose 读取 pose_key 执行。
计算逻辑保留原抓箱流程的躯干对准约束。
"""

import math

import py_trees
from kuavo_humanoid_sdk.common.three_link_torso_ik import ThreeLinkTorsoIk
from py_trees.common import Status

from tree.constants import ROBOT_SERVICES_KEY

from ..base import TimedMockAction


class ComputeMoveBoxTorsoToBoxCenterPose(TimedMockAction):
    """根据最新箱体中心生成抓取前躯干对准目标。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = ROBOT_SERVICES_KEY
        self.box_center_key = str(
            params.get("box_center_key", "move_box_latest_box_center")
        ).strip()
        self.preferred_box_center_key = str(
            params.get("preferred_box_center_key", "")
        ).strip()
        self.target_pose_key = str(
            params.get("target_pose_key", "move_box_torso_box_center_pose")
        ).strip()
        self.knee_origin_z = float(params.get("knee_origin_z", 0.1055))
        self.ik_angle_step_rad = math.radians(
            float(params.get("ik_angle_step_deg", 1.0))
        )
        self.torso_ik = ThreeLinkTorsoIk()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)
        if self.preferred_box_center_key:
            self.blackboard.register_key(
                key=self.preferred_box_center_key,
                access=py_trees.common.Access.READ,
            )
        self.blackboard.register_key(key=self.target_pose_key, access=py_trees.common.Access.WRITE)

    def update(self):
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        box_center, box_center_source = self._get_box_center()
        if services is None or box_center is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] robot services 或箱体中心缺失: "
                f"services={services is not None}, box_center={box_center is not None}"
            )
            return Status.FAILURE

        target_pose = list(services.torso_controller.current_pose)
        if len(target_pose) != 6:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 当前躯干位姿长度异常: {target_pose}"
            )
            return Status.FAILURE

        target_pose[0] = min(max(float(box_center[0]) - 0.5, 0.0), 0.5)
        target_pose[2] = min(max(float(box_center[2]) + 0.2, 0.6), 1.4)

        # 关键步骤：三连杆模型以 knee_joint 为原点，控制目标 z 需要扣除该关节高度。
        ik_target_x = target_pose[0]
        ik_target_z = target_pose[2] - self.knee_origin_z
        ik_pose = self.torso_ik.solve_exact(
            ik_target_x,
            ik_target_z,
            angle_step=self.ik_angle_step_rad,
        )
        if ik_pose is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 三连杆 IK 无精确解: "
                f"torso_target=({target_pose[0]:.3f}, {target_pose[2]:.3f}), "
                f"ik_target=({ik_target_x:.3f}, {ik_target_z:.3f})"
            )
            return Status.FAILURE

        # 关键步骤：torso pitch 定义为 l3 相对竖直方向的有符号夹角，前倾为正。
        target_pose[4] = ik_pose["torso_pitch"]
        self.blackboard.set(self.target_pose_key, target_pose, overwrite=True)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算抓取前躯干目标: "
            f"source={box_center_source}, "
            f"box=({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}), "
            f"l3_angle={ik_pose['l3_angle']:.3f}, pitch={target_pose[4]:.3f}, "
            f"pose={target_pose}, key={self.target_pose_key}"
        )
        return Status.SUCCESS

    def _get_box_center(self):
        """优先读取到点时保存的箱体中心，缺失时回退到最新检测结果。"""
        if self.preferred_box_center_key and self.blackboard.exists(self.preferred_box_center_key):
            box_center = self.blackboard.get(self.preferred_box_center_key)
            if box_center is not None:
                return box_center, self.preferred_box_center_key
        if self.blackboard.exists(self.box_center_key):
            return self.blackboard.get(self.box_center_key), self.box_center_key
        return None, self.box_center_key

    def describe_start(self):
        return (
            f"[{self.config_label}] ComputeMoveBoxTorsoToBoxCenterPose start: "
            f"box_center_key={self.box_center_key}, target_pose_key={self.target_pose_key}"
        )
