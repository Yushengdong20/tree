"""ROS1 rospy 适配实现。"""

import os
import threading
from datetime import datetime

from tree.ros_interface.base import RosInterface
from tree.ros_interface.ros1.publisher import Ros1StringPublisher
from tree.ros_interface.ros1.timer import Ros1Timer


class Ros1Interface(RosInterface):
    """基于 rospy 的运行时适配器。"""

    def __init__(self, node_name: str):
        import rospy
        from std_msgs.msg import String

        self._rospy = rospy
        self._string_type = String
        self._node_name = node_name
        self._spin_thread = None
        self._initialized = False
        self._log_file_path = None
        self._file_log_warning_reported = False

    def init(self, args=None):
        del args
        if self._initialized:
            return
        # ROS1 init_node 只能执行一次，这里由接口层统一管理。
        self._rospy.init_node(self._node_name, anonymous=False)
        # 关键步骤：每次启动生成独立日志文件，避免多轮任务日志互相覆盖。
        self._init_file_logger()
        self._initialized = True

    def shutdown(self):
        if self._rospy.core.is_initialized():
            self._rospy.signal_shutdown("shutdown requested")

    def ok(self) -> bool:
        return not self._rospy.is_shutdown()

    def spin(self):
        self._rospy.spin()

    def start_background_spin(self):
        if self._spin_thread is not None:
            return
        self._spin_thread = threading.Thread(target=self.spin, daemon=True)
        self._spin_thread.start()

    def destroy_node(self):
        # rospy 没有显式 node destroy，资源跟随进程和 signal_shutdown 释放。
        return

    def get_logger(self):
        return self

    def info(self, msg: str):
        self._rospy.loginfo(msg)
        self._write_file_log("INFO", msg)

    def warning(self, msg: str):
        self._rospy.logwarn(msg)
        self._write_file_log("WARN", msg)

    def error(self, msg: str):
        self._rospy.logerr(msg)
        self._write_file_log("ERROR", msg)

    def _init_file_logger(self):
        """初始化 MercuryTree 独立文件日志路径。"""
        log_dir = "/mnt/ssd/log"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        node_name = self._node_name.strip("/") or "mercury_tree"
        self._log_file_path = os.path.join(
            log_dir,
            f"mercury_tree_{node_name}_{timestamp}.log",
        )

    def _write_file_log(self, level: str, msg: str):
        """把通过 Ros1Interface 的日志同步追加到独立文件。"""
        if not self._log_file_path:
            return

        try:
            # 关键步骤：实机上 /mnt/ssd/log 可能不存在，这里按需创建。
            os.makedirs(os.path.dirname(self._log_file_path), exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            with open(self._log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp} [{level}] [{self._node_name}] {msg}\n")
        except Exception as exc:
            if not self._file_log_warning_reported:
                self._file_log_warning_reported = True
                self._rospy.logwarn(f"MercuryTree 文件日志写入失败: {exc}")

    def get_name(self) -> str:
        return self._rospy.get_name().strip("/") or self._node_name

    def get_namespace(self) -> str:
        return self._rospy.get_namespace()

    def get_native_node(self):
        return None

    def get_package_share_directory(self, package_name: str) -> str:
        import rospkg

        # ROS1 没有 ament share 概念，包根目录承担资源目录角色。
        return rospkg.RosPack().get_path(package_name)

    def get_param(self, name: str, default_value):
        return self._rospy.get_param(f"~{name}", default_value)

    def set_parameters(self, parameters):
        for name, value in parameters.items():
            self._rospy.set_param(f"~{name}", value)

    def sleep(self, duration_sec: float):
        self._rospy.sleep(duration_sec)

    def now(self):
        return self._rospy.Time.now()

    def zero_time(self):
        return self._rospy.Time(0)

    def duration(self, duration_sec: float):
        return self._rospy.Duration(duration_sec)

    def create_timer(self, period_sec, callback):
        def _on_timer(_event):
            callback()

        timer = self._rospy.Timer(self._rospy.Duration(period_sec), _on_timer)
        return Ros1Timer(timer)

    def create_publisher(self, topic_name, message_type, queue_size=10, latch=False):
        return self._rospy.Publisher(
            topic_name,
            message_type,
            queue_size=queue_size,
            latch=latch,
        )

    def create_subscription(self, topic_name, message_type, callback, queue_size=10, latch=False):
        del latch
        return self._rospy.Subscriber(
            topic_name,
            message_type,
            callback,
            queue_size=queue_size,
        )

    def create_string_publisher(self, topic_name, queue_size=10, latch=False):
        publisher = self._rospy.Publisher(
            topic_name,
            self._string_type,
            queue_size=queue_size,
            latch=latch,
        )
        return Ros1StringPublisher(publisher, self._string_type)

    def create_string_subscription(self, topic_name, callback, queue_size=10, latch=False):
        del latch

        def _on_message(message):
            callback(message.data.strip())

        return self._rospy.Subscriber(
            topic_name,
            self._string_type,
            _on_message,
            queue_size=queue_size,
        )

    def call_trigger_service(self, service_name: str, timeout_sec: float = 5.0):
        from std_srvs.srv import Trigger

        self._rospy.wait_for_service(service_name, timeout=timeout_sec)
        client = self._rospy.ServiceProxy(service_name, Trigger)
        return client()
