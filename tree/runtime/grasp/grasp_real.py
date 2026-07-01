"""grasp_object 真实机器人业务的共享服务封装。"""

from dataclasses import dataclass

import rospy
import tf

from tree.constants import WAIST_YAW_LINK_FRAME
from kuavo_humanoid_sdk.common.arm_controller import ArmController
from kuavo_humanoid_sdk.common.lb_control import set_quick_mode
from kuavo_humanoid_sdk.kuavo_strategy_v2.common.events.mobile_manipulate.ik_library import (
    IK_MODEL_GRASP,
)


@dataclass
class GraspObjectServices:
    """集中保存 grasp_object 流程要复用的真实控制实例。"""

    target_frame: str
    tf_listener: tf.TransformListener
    arm_controller: ArmController
    model_type: str


def build_grasp_object_services(model_type=IK_MODEL_GRASP) -> GraspObjectServices:
    model_type = str(model_type).strip() or IK_MODEL_GRASP
    if bool(rospy.get_param("~auto_set_quick_mode", True)):
        set_quick_mode(int(rospy.get_param("~quick_mode", 2)))

    """创建一套可被抓取行为树节点复用的真实服务对象。"""
    tf_listener = tf.TransformListener()
    arm_controller = ArmController(
        tf_listener,
        WAIST_YAW_LINK_FRAME,
        model_type=model_type,
    )
    return GraspObjectServices(
        target_frame=WAIST_YAW_LINK_FRAME,
        tf_listener=tf_listener,
        arm_controller=arm_controller,
        model_type=model_type,
    )
