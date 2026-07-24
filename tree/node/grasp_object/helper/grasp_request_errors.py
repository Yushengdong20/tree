"""grasp_object 抓取请求异常类型。"""


class NoGraspObjectError(RuntimeError):
    """抓取服务明确表示当前没有可用抓取目标。"""
