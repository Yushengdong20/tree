"""move_box 真实机器人业务的共享服务封装。"""

from dataclasses import dataclass

import rospy
import tf

from kuavo_humanoid_sdk.common.arm_controller import ArmController
from kuavo_humanoid_sdk.common.box_detection import (
    BoxGraspDetector,
    FPBoxDetector,
    YoloBoxDetector,
)
from kuavo_humanoid_sdk.common.head_controller import HeadController
from kuavo_humanoid_sdk.common.lb_control import set_quick_mode
from kuavo_humanoid_sdk.common.static_tf_publisher import StaticTfPublisher
from kuavo_humanoid_sdk.common.torso_controller import TorsoController


@dataclass
class MoveBoxServices:
    """集中保存 move_box 流程要复用的真实控制器和检测器实例。"""

    target_frame: str
    detector_topic: str
    detector_type: str
    tf_listener: tf.TransformListener
    static_tf_publisher: StaticTfPublisher
    box_detector: BoxGraspDetector
    yolo_detector: YoloBoxDetector
    arm_controller: ArmController
    torso_controller: TorsoController
    head_controller: HeadController


def build_move_box_services() -> MoveBoxServices:
    """创建一套可被行为树节点复用的 move_box 真实服务对象。"""
    if bool(rospy.get_param("~auto_set_quick_mode", True)):
        set_quick_mode(int(rospy.get_param("~quick_mode", 2)))

    detector_type = str(rospy.get_param("~box_detector_type", "foundationpose")).lower()
    marker_topic = rospy.get_param("~marker_topic", "/aruco_single/poses")
    box_pose_topic = rospy.get_param("~box_pose_topic", "/foundationpose/pose")
    yolo_target_poses_topic = rospy.get_param("~yolo_target_poses_topic", "/yolo/target_poses")
    target_frame = rospy.get_param("~target_frame", "base_link")
    if target_frame != "base_link":
        rospy.logwarn(
            "当前手臂 IK 事件只支持 base_link 目标，已将 target_frame 从 %s 改为 base_link",
            target_frame,
        )
        target_frame = "base_link"

    static_tf_publisher = StaticTfPublisher()
    tf_listener = tf.TransformListener()
    if detector_type in ("aruco", "marker"):
        box_detector = BoxGraspDetector(target_frame, tf_listener, marker_topic)
        detector_topic = marker_topic
    else:
        detector_type = "foundationpose"
        box_detector = FPBoxDetector(target_frame, tf_listener, box_pose_topic)
        detector_topic = box_pose_topic
    yolo_detector = YoloBoxDetector(target_frame, tf_listener, yolo_target_poses_topic)
    arm_controller = ArmController(tf_listener, target_frame)
    torso_controller = TorsoController()
    head_controller = HeadController()
    return MoveBoxServices(
        target_frame=target_frame,
        detector_topic=detector_topic,
        detector_type=detector_type,
        tf_listener=tf_listener,
        static_tf_publisher=static_tf_publisher,
        box_detector=box_detector,
        yolo_detector=yolo_detector,
        arm_controller=arm_controller,
        torso_controller=torso_controller,
        head_controller=head_controller,
    )
