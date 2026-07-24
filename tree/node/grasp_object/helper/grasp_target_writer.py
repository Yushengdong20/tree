"""grasp_object 目标写入和调试 pose 发布。"""

import numpy as np
from geometry_msgs.msg import PoseStamped

from tree.constants import BASE_LINK_FRAME, MAP_FRAME, WAIST_YAW_LINK_FRAME
from tree.utils.geometry import format_xyz_ypr, matrix_to_xyz_ypr


class GraspTargetWriter:
    """封装单抓/双抓目标写入 blackboard 和 ROS 调试发布。"""

    def _write_grasp_targets(self, selected, preserve_motion_flags=None):
        """写入选中抓取方案，到位后复核时可保留原底盘/torso 执行标记。"""
        sample = selected["sample"]
        preserve_flags = (
            self.preserve_motion_flags
            if preserve_motion_flags is None
            else bool(preserve_motion_flags)
        )
        self._publish_grasp_pose(selected["grasp_target_pose"], selected["waist_from_base"])
        self._publish_raw_grasp_pose(selected["grasp_pose"])

        self.blackboard.set(
            self.left_pose_key,
            selected["arm_targets"]["left"],
            overwrite=True,
        )
        self.blackboard.set(self.grasp_pose_key, list(selected["grasp_target"]), overwrite=True)
        self.blackboard.set(
            self.pregrasp_pose_key,
            list(selected["pregrasp_target"]),
            overwrite=True,
        )
        self.blackboard.set(
            self.selected_arm_side_key,
            selected["arm_side"],
            overwrite=True,
        )
        self.blackboard.set(self.selected_grasp_pose_key, selected["grasp_target_pose"], overwrite=True)
        self.blackboard.set(self.selected_base_grasp_pose_key, selected["grasp_pose"], overwrite=True)
        selected_map_grasp_pose = (
            selected["source_grasp_pose"]
            if selected.get("poses_frame") == MAP_FRAME
            else None
        )
        self.blackboard.set(self.selected_map_grasp_pose_key, selected_map_grasp_pose, overwrite=True)
        self.blackboard.set(self.selected_rotate_z_180_key, bool(selected["rotate_z_180"]), overwrite=True)
        self.blackboard.set(self.selected_waist_from_base_key, selected["waist_from_base"], overwrite=True)
        lateral_offset = float(sample.get("chassis_lateral_offset_m", 0.0))
        if not preserve_flags:
            self.blackboard.set(self.torso_enabled_key, bool(sample["enabled"]), overwrite=True)
            self.blackboard.set(self.chassis_lateral_offset_key, lateral_offset, overwrite=True)
            self.blackboard.set(
                self.chassis_lateral_enabled_key,
                bool(sample.get("chassis_lateral_enabled", False)),
                overwrite=True,
            )
            if sample["enabled"]:
                self.blackboard.set(self.torso_pose_key, list(sample["torso_pose"]), overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选择{sample['label']}下第 "
            f"{selected['index'] + 1} 个{selected['pose_label']}{selected['arm_side']}臂抓取位姿，"
            f"已写入 blackboard: side={self.selected_arm_side_key}, "
            f"grasp={self.grasp_pose_key}, "
            f"use_torso={self.torso_enabled_key}, "
            f"chassis_lateral={lateral_offset:.3f}m"
        )
        self._log_selected_grasp_source_poses(selected)

    def _write_dual_grasp_targets(self, selected, preserve_motion_flags=None):
        """写入同步双抓的左右手目标和复核元信息。"""
        sample = selected["sample"]
        preserve_flags = (
            self.preserve_motion_flags
            if preserve_motion_flags is None
            else bool(preserve_motion_flags)
        )
        left = selected["left"]
        right = selected["right"]

        # 关键步骤：同步双抓执行节点会按左右独立 key 一次性下发双臂目标。
        self.blackboard.set(self.left_grasp_pose_key, list(left["grasp_target"]), overwrite=True)
        self.blackboard.set(self.right_grasp_pose_key, list(right["grasp_target"]), overwrite=True)
        self.blackboard.set(self.left_pregrasp_pose_key, list(left["pregrasp_target"]), overwrite=True)
        self.blackboard.set(self.right_pregrasp_pose_key, list(right["pregrasp_target"]), overwrite=True)

        self.blackboard.set(self.left_selected_grasp_pose_key, left["grasp_target_pose"], overwrite=True)
        self.blackboard.set(self.right_selected_grasp_pose_key, right["grasp_target_pose"], overwrite=True)
        self.blackboard.set(self.left_selected_base_grasp_pose_key, left["grasp_pose"], overwrite=True)
        self.blackboard.set(self.right_selected_base_grasp_pose_key, right["grasp_pose"], overwrite=True)
        self.blackboard.set(
            self.left_selected_map_grasp_pose_key,
            left["source_grasp_pose"] if left.get("poses_frame") == MAP_FRAME else None,
            overwrite=True,
        )
        self.blackboard.set(
            self.right_selected_map_grasp_pose_key,
            right["source_grasp_pose"] if right.get("poses_frame") == MAP_FRAME else None,
            overwrite=True,
        )
        self.blackboard.set(
            self.left_selected_rotate_z_180_key,
            bool(left["rotate_z_180"]),
            overwrite=True,
        )
        self.blackboard.set(
            self.right_selected_rotate_z_180_key,
            bool(right["rotate_z_180"]),
            overwrite=True,
        )
        self.blackboard.set(self.dual_selected_waist_from_base_key, left["waist_from_base"], overwrite=True)
        self.blackboard.set(
            self.dual_selected_pair_info_key,
            {
                "sample_order": int(selected.get("sample_order", 0)),
                "left_object_index": int(left.get("object_index", -1)),
                "right_object_index": int(right.get("object_index", -1)),
                "left_object_id": left.get("object_id"),
                "right_object_id": right.get("object_id"),
                "left_grasp_index": int(left.get("index", -1)),
                "right_grasp_index": int(right.get("index", -1)),
            },
            overwrite=True,
        )

        lateral_offset = float(sample.get("chassis_lateral_offset_m", 0.0))
        if not preserve_flags:
            self.blackboard.set(self.torso_enabled_key, bool(sample["enabled"]), overwrite=True)
            self.blackboard.set(self.chassis_lateral_offset_key, lateral_offset, overwrite=True)
            self.blackboard.set(
                self.chassis_lateral_enabled_key,
                bool(sample.get("chassis_lateral_enabled", False)),
                overwrite=True,
            )
            if sample["enabled"]:
                self.blackboard.set(self.torso_pose_key, list(sample["torso_pose"]), overwrite=True)

        self.ros_node.get_logger().info(
            f"[{self.config_label}] 已写入同步双抓目标: "
            f"sample={sample['label']}, "
            f"left_object={left.get('object_index')}, left_grasp={left.get('grasp_target')}, "
            f"right_object={right.get('object_index')}, right_grasp={right.get('grasp_target')}, "
            f"use_torso={bool(sample['enabled'])}, chassis_lateral={lateral_offset:.3f}m"
        )

    def _log_selected_grasp_source_poses(self, selected):
        """打印选中候选在 camera、base_link 和 waist_yaw_link 下的位姿，便于核对坐标转换。"""
        camera_grasp_pose = self._read_camera_grasp_pose_by_index(int(selected["index"]))
        base_grasp_target = matrix_to_xyz_ypr(selected["grasp_pose"])
        waist_grasp_target = matrix_to_xyz_ypr(selected["grasp_target_pose"])
        if camera_grasp_pose is None:
            self.ros_node.get_logger().warning(
                f"[{self.config_label}] 选中候选缺少 camera 原始抓取位姿: "
                f"key={self.camera_grasp_poses_key}, index={selected['index'] + 1}"
            )
            self.ros_node.get_logger().info(
                f"[{self.config_label}] 选中候选抓取位姿坐标: "
                f"camera=<missing>, "
                f"base_link={format_xyz_ypr(base_grasp_target)}, "
                f"waist_yaw_link={format_xyz_ypr(waist_grasp_target)}"
            )
            return

        camera_grasp_target = matrix_to_xyz_ypr(camera_grasp_pose)
        self.ros_node.get_logger().info(
            f"[{self.config_label}] 选中候选抓取位姿坐标: "
            f"camera={format_xyz_ypr(camera_grasp_target)}, "
            f"base_link={format_xyz_ypr(base_grasp_target)}, "
            f"waist_yaw_link={format_xyz_ypr(waist_grasp_target)}"
        )

    def _publish_target_pose(self, target_pose):
        quaternion = self._tf.transformations.quaternion_from_matrix(target_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = WAIST_YAW_LINK_FRAME
        message.pose.position.x = float(target_pose[0, 3])
        message.pose.position.y = float(target_pose[1, 3])
        message.pose.position.z = float(target_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.pose_pub.publish(message)

    def _publish_grasp_pose(self, target_pose, target_from_base):
        source_pose = np.linalg.inv(target_from_base) @ target_pose
        quaternion = self._tf.transformations.quaternion_from_matrix(source_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = BASE_LINK_FRAME
        message.pose.position.x = float(source_pose[0, 3])
        message.pose.position.y = float(source_pose[1, 3])
        message.pose.position.z = float(source_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.grasp_pose_pub.publish(message)

    def _publish_raw_grasp_pose(self, grasp_pose):
        quaternion = self._tf.transformations.quaternion_from_matrix(grasp_pose)
        message = PoseStamped()
        message.header.stamp = self.ros_node.now()
        message.header.frame_id = BASE_LINK_FRAME
        message.pose.position.x = float(grasp_pose[0, 3])
        message.pose.position.y = float(grasp_pose[1, 3])
        message.pose.position.z = float(grasp_pose[2, 3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.raw_grasp_pose_pub.publish(message)
