"""grasp_object 参数校验和 blackboard key 注册辅助。"""


def require_non_empty(value, message):
    """校验字符串配置非空，并返回原值。"""
    if not value:
        raise ValueError(message)
    return value


def register_blackboard_keys(blackboard, keys, access):
    """批量注册 blackboard key，避免初始化代码重复展开。"""
    for key in keys:
        blackboard.register_key(key=key, access=access)


def register_blackboard_read_write_keys(blackboard, keys, read_access, write_access):
    """批量注册既读又写的 blackboard key。"""
    for key in keys:
        blackboard.register_key(key=key, access=read_access)
        blackboard.register_key(key=key, access=write_access)
