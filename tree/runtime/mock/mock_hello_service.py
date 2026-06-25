"""用于验证 blackboard 复用实例行为的共享 mock 服务。"""


class MockHelloService:
    """一个带状态的小型服务，用于记录创建次数和调用次数。"""

    # 类变量：统计这个类型一共被实例化了多少次，用来验证“是否重复创建”。
    created_count = 0

    def __init__(self):
        # 每创建一个新实例，就递增全局创建计数，并把当时的序号记到实例上。
        type(self).created_count += 1
        self.creation_index = type(self).created_count
        # 实例变量：统计同一个实例被调用了多少次，用来验证“是否真的是同一个对象在复用”。
        self.call_count = 0

    def say_hello(self):
        """返回 hello world 文本，并递增当前实例的调用计数。"""
        # 每次调用都递增实例内计数；如果节点复用的是同一个实例，这个值会持续累加。
        self.call_count += 1
        return (
            f"hello world | instance_id={id(self)} | "
            f"creation_index={self.creation_index} | call_count={self.call_count}"
        )
