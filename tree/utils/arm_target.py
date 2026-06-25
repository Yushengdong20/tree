"""机械臂目标结构辅助函数。"""


def looks_like_arm_pose_pair(value):
    """判断值是否长得像 [left_poses, right_poses] 这种双臂结构。"""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (list, tuple))
        and isinstance(value[1], (list, tuple))
    )


def is_pose_object_pair(value):
    """判断值是否已经是 SDK Pose 对象组成的双臂结构。"""
    if not looks_like_arm_pose_pair(value):
        return False
    left = value[0][0] if value[0] else None
    right = value[1][0] if value[1] else None
    return hasattr(left, "frame") or hasattr(right, "frame")


def normalize_wrench(target_wrench, left_count, right_count):
    """把力控输入规范成 (left_wrenches, right_wrenches) 结构。"""
    default_left = [[0.0] * 6 for _ in range(left_count)]
    default_right = [[0.0] * 6 for _ in range(right_count)]
    if target_wrench is None:
        return (default_left, default_right)

    if isinstance(target_wrench, dict):
        left = target_wrench.get("left", default_left)
        right = target_wrench.get("right", default_right)
        return (left, right)

    if (
        isinstance(target_wrench, (list, tuple))
        and len(target_wrench) == 2
        and isinstance(target_wrench[0], (list, tuple))
        and isinstance(target_wrench[1], (list, tuple))
    ):
        return (list(target_wrench[0]), list(target_wrench[1]))

    raise ValueError(
        "target_wrench 需要是 {'left': [...], 'right': [...]} 或 [left_wrenches, right_wrenches]"
    )
