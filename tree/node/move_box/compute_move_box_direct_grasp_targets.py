"""计算 move_box 直接夹取流程的双手同步目标点。"""

import numpy as np
import py_trees
from py_trees.common import Status
from visualization_msgs.msg import MarkerArray

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, ROBOT_SERVICES_KEY
from ..base import TimedMockAction
from .fp_grasp_visualization import publish_fp_box_and_targets


class ComputeMoveBoxDirectGraspTargets(TimedMockAction):
    """计算直接抓取的双爪目标点，兼容旧夹爪与新夹爪两种动作模式。

    ``top_down_legacy``（默认）保留历史行为：从箱体上方到边缘下方，适用于
    旧款“下探/插入式”夹爪。

    ``side_inward_top_clamp`` 适用于新款上下合拢夹爪：双臂先在箱体两侧外侧
    预张开，再沿箱体左右方向向内移动到侧边夹持位，最后由 ``CloseClaw`` 执行
    上下合拢。夹持高度由箱顶向下的距离决定，默认是箱顶下 3 cm，即夹持槽覆盖
    箱体顶部及其下方约 6 cm 的中部位置。
    """

    def __init__(self, name, config_label, ros_node, params):
        super().__init__(name=name, config_label=config_label, ros_node=ros_node, params=params)
        self.grasp_pair_key = str(params.get("grasp_pair_key", "move_box_latest_grasp_pair")).strip()
        self.box_axes_key = str(params.get("box_axes_key", "move_box_latest_box_axes")).strip()
        self.box_center_key = str(params.get("box_center_key", "move_box_latest_box_center")).strip()
        self.services_key = str(params.get("services_key", ROBOT_SERVICES_KEY)).strip()
        self.grasp_mode = str(params.get("grasp_mode", "top_down_legacy")).strip().lower()
        self.fp_grasp_visualization_enabled = self._to_bool(
            params.get("fp_grasp_visualization_enabled", True)
        )
        self.fp_grasp_visualization_topic = str(
            params.get("fp_grasp_visualization_topic", "/move_box/fp_grasp_markers")
        ).strip()
        self.odom_topic = str(params.get("odom_topic", "melon_odom")).strip()
        self.odom_transformer = self.get_odom_pose_transformer(
            self.odom_topic,
            target_frame=MAP_FRAME,
            base_frame=BASE_LINK_FRAME,
        )
        self.fp_grasp_visualization_publisher = None
        if self.fp_grasp_visualization_enabled and self.fp_grasp_visualization_topic:
            self.fp_grasp_visualization_publisher = self.ros_node.create_publisher(
                self.fp_grasp_visualization_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
        self.target_keys = {
            "left_above": str(params.get("left_above_key", "move_box_direct_left_above_edge")).strip(),
            "right_above": str(params.get("right_above_key", "move_box_direct_right_above_edge")).strip(),
            "left_below": str(params.get("left_below_key", "move_box_direct_left_below_edge")).strip(),
            "right_below": str(params.get("right_below_key", "move_box_direct_right_below_edge")).strip(),
            "left_lift": str(params.get("left_lift_key", "move_box_direct_left_lift_target")).strip(),
            "right_lift": str(params.get("right_lift_key", "move_box_direct_right_lift_target")).strip(),
        }
        self.blackboard.register_key(key=self.services_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.grasp_pair_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_axes_key, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=self.box_center_key, access=py_trees.common.Access.READ)
        for key in self.target_keys.values():
            self.blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    def update(self):
        """根据当前夹爪模式计算双手同步目标。"""
        if self.should_use_mock_execution():
            return self.update_mock_result()

        grasp_pair = self.blackboard.get(self.grasp_pair_key) if self.blackboard.exists(self.grasp_pair_key) else None
        box_axes = self.blackboard.get(self.box_axes_key) if self.blackboard.exists(self.box_axes_key) else None
        if grasp_pair is None or box_axes is None:
            self.ros_node.get_logger().error(f"[{self.config_label}] 缺少直接抓取目标计算所需数据")
            return Status.FAILURE

        left_edge_point, right_edge_point = [
            np.asarray(point, dtype=float) for point in grasp_pair
        ]
        try:
            up_axis = self._normalize_axis(box_axes.get("up"), "up")
        except (AttributeError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 读取箱体上方向失败: {exc}"
            )
            return Status.FAILURE
        approach_offset = float(
            self.params.get(
                "direct_approach_offset",
                self.ros_node.get_param(
                    "direct_approach_offset",
                    self.ros_node.get_param("left_approach_offset", 0.05),
                ),
            )
        )
        descend_below_offset = float(
            self.params.get(
                "direct_descend_below_offset",
                self.ros_node.get_param(
                    "direct_descend_below_offset",
                    self.ros_node.get_param("left_descend_below_offset", 0.06),
                ),
            )
        )
        lift_offset = float(
            self.params.get(
                "direct_lift_offset",
                self.ros_node.get_param(
                    "direct_lift_offset",
                    self.ros_node.get_param("left_lift_offset", 0.2),
                ),
            )
        )

        try:
            if self.grasp_mode == "top_down_legacy":
                targets, log_detail = self._compute_top_down_legacy_targets(
                    left_edge_point,
                    right_edge_point,
                    up_axis,
                    approach_offset,
                    descend_below_offset,
                    lift_offset,
                )
            elif self.grasp_mode == "side_inward_top_clamp":
                targets, log_detail = self._compute_side_inward_top_clamp_targets(
                    left_edge_point,
                    right_edge_point,
                    up_axis,
                    lift_offset,
                )
            else:
                raise ValueError(
                    f"不支持的 direct grasp_mode: {self.grasp_mode}; "
                    "可选 top_down_legacy / side_inward_top_clamp"
                )
        except (KeyError, TypeError, ValueError) as exc:
            self.ros_node.get_logger().error(
                f"[{self.config_label}] 计算直接抓取目标失败: {exc}"
            )
            return Status.FAILURE

        self.blackboard.set(self.target_keys["left_above"], targets["left_above"], overwrite=True)
        self.blackboard.set(self.target_keys["right_above"], targets["right_above"], overwrite=True)
        self.blackboard.set(self.target_keys["left_below"], targets["left_below"], overwrite=True)
        self.blackboard.set(self.target_keys["right_below"], targets["right_below"], overwrite=True)
        self.blackboard.set(self.target_keys["left_lift"], targets["left_lift"], overwrite=True)
        self.blackboard.set(self.target_keys["right_lift"], targets["right_lift"], overwrite=True)
        self._publish_visualization(
            grasp_pair,
            box_axes,
            targets,
        )
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已计算直接抓取目标: mode={self.grasp_mode}, {log_detail}"
        )
        return Status.SUCCESS

    @staticmethod
    def _normalize_axis(raw_axis, name):
        axis = np.asarray(raw_axis, dtype=float)
        if axis.shape != (3,):
            raise ValueError(f"box_axes[{name!r}] 不是长度为 3 的向量: {raw_axis}")
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"box_axes[{name!r}] 非法: {raw_axis}")
        return axis / norm

    @staticmethod
    def _target_dictionary(left_first, right_first, left_second, right_second, up_axis, lift_offset):
        """统一输出两阶段目标；第二阶段是实际夹持位，供后续上提复用。"""
        return {
            "left_above": left_first,
            "right_above": right_first,
            "left_below": left_second,
            "right_below": right_second,
            "left_lift": left_second + up_axis * lift_offset,
            "right_lift": right_second + up_axis * lift_offset,
        }

    def _compute_top_down_legacy_targets(
        self,
        left_edge_point,
        right_edge_point,
        up_axis,
        approach_offset,
        descend_below_offset,
        lift_offset,
    ):
        """历史下探式夹爪动作点，禁止改动其几何含义以避免影响旧树。"""
        above_left_edge = left_edge_point + up_axis * approach_offset
        above_right_edge = right_edge_point + up_axis * approach_offset
        below_left_edge = left_edge_point - up_axis * descend_below_offset
        below_right_edge = right_edge_point - up_axis * descend_below_offset
        return (
            self._target_dictionary(
                above_left_edge,
                above_right_edge,
                below_left_edge,
                below_right_edge,
                up_axis,
                lift_offset,
            ),
            f"approach={approach_offset:.3f}, descend={descend_below_offset:.3f}, "
            f"lift={lift_offset:.3f}",
        )

    def _compute_side_inward_top_clamp_targets(
        self,
        left_edge_point,
        right_edge_point,
        up_axis,
        lift_offset,
    ):
        """新夹爪：两侧外侧预抓位 -> 两侧边缘夹持位 -> 上下闭爪。"""
        side_axis = self._normalize_axis(left_edge_point - right_edge_point, "left-right")
        half_span = float(np.linalg.norm(left_edge_point - right_edge_point)) * 0.5
        if half_span < 1e-4:
            raise ValueError("左右 FoundationPose 抓取点重合，无法规划新夹爪侧向合拢")

        # FP 历史抓取点位于箱顶边缘；新夹爪需要把工具中心降至顶部夹持槽中心。
        clamp_from_top = float(self.params.get("new_claw_grasp_from_top_offset_m", 0.03))
        outward_offset = float(self.params.get("new_claw_side_pregrasp_offset_m", 0.06))
        if clamp_from_top < 0.0 or outward_offset < 0.0:
            raise ValueError("new_claw_grasp_from_top_offset_m 与 new_claw_side_pregrasp_offset_m 必须非负")

        left_clamp = left_edge_point - up_axis * clamp_from_top
        right_clamp = right_edge_point - up_axis * clamp_from_top
        left_pregrasp = left_clamp + side_axis * outward_offset
        right_pregrasp = right_clamp - side_axis * outward_offset
        return (
            self._target_dictionary(
                left_pregrasp,
                right_pregrasp,
                left_clamp,
                right_clamp,
                up_axis,
                lift_offset,
            ),
            "侧向预张开={:.3f}, 顶部向下夹持={:.3f}, lift={:.3f}, "
            "两爪边缘间距={:.3f}".format(
                outward_offset,
                clamp_from_top,
                lift_offset,
                half_span * 2.0,
            ),
        )

    def _publish_visualization(self, grasp_pair, box_axes, target_points):
        box_center = (
            self.blackboard.get(self.box_center_key)
            if self.blackboard.exists(self.box_center_key)
            else None
        )
        services = (
            self.blackboard.get(self.services_key)
            if self.blackboard.exists(self.services_key)
            else None
        )
        publish_fp_box_and_targets(
            ros_node=self.ros_node,
            publisher=self.fp_grasp_visualization_publisher,
            topic=self.fp_grasp_visualization_topic,
            config_label=self.config_label,
            odom_transformer=self.odom_transformer,
            services=services,
            box_center=box_center,
            box_axes=box_axes,
            strategy="direct",
            grasp_pair=grasp_pair,
            target_points=target_points,
            include_grasp_targets=True,
        )
