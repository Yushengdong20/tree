"""从持续运行的检测器中刷新最新抓取数据。"""

import py_trees
from py_trees.common import Status

from ..base import TimedMockAction


class RefreshMoveBoxGraspData(TimedMockAction):
    """把当前最新的抓取点和箱体方向缓存同步到 blackboard。"""

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.services_key = str(params.get("services_key", "move_box_services")).strip()
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(
            params.get("box_center_key", "move_box_latest_box_center")
        ).strip()
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.WRITE)

    def update(self):
        """使用共享检测器刷新最新抓取结果，并写入 blackboard。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        services = self.blackboard.get(self.services_key) if self.blackboard.exists(self.services_key) else None
        if services is None:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] move_box services missing on blackboard: key={self.services_key}"
            )
            return Status.FAILURE

        services.box_detector.update_latest_grasp_pose(
            services.arm_controller.get_initial_left_ypr(),
            services.arm_controller.get_initial_right_ypr(),
        )
        grasp_pair = services.box_detector.get_latest_grasp_pair()
        box_axes = services.box_detector.get_latest_box_axes()
        box_center = services.box_detector.get_latest_box_center()
        self.blackboard.set(self.grasp_pair_key, grasp_pair, overwrite=True)
        self.blackboard.set(self.box_axes_key, box_axes, overwrite=True)
        self.blackboard.set(self.box_center_key, box_center, overwrite=True)
        return Status.SUCCESS
