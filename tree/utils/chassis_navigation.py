"""底盘 HTTP 导航工具函数。

这个模块只保留底盘导航相关协议：
- 查询当前底盘 global 位姿
- 创建导航任务
- 查询导航任务状态

抓取、放置、视觉检测不放在这里，避免和旧 move_box HTTP 链路耦合。
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


# 使用坐标点导航，不从底盘点位数据库加载 POI。
DEFAULT_POI_ID = 0
# 普通目标点类型，保持和底盘调试客户端的默认值一致。
DEFAULT_POI_TYPE = 0
# 1 表示普通绕障到点导航，2 是循迹模式，这里固定使用普通导航。
DEFAULT_NAVIGATION_MODE = 1
# 底盘 m1d HTTP 服务地址。
DEFAULT_CHASSIS_URL = "http://192.168.168.125:8081"
# 底盘接口要求的客户端标识。
DEFAULT_CLIENT_ID = "android-app"
# 底盘导航和状态查询的单次 HTTP 超时时间。
DEFAULT_HTTP_TIMEOUT_SEC = 8.0
# 底盘导航任务自身的超时时间，同时作为等待任务成功的总超时。
DEFAULT_NAVIGATION_TIMEOUT_SEC = 600
# 底盘线速度上限。
DEFAULT_LINEAR_SPEED_LIMIT = 0.6
# 查询导航任务状态的轮询间隔。
DEFAULT_POLL_INTERVAL_SEC = 0.2


TASK_STATUS_INITIALIZED = 0
TASK_STATUS_RUNNING = 1
TASK_STATUS_SCHEDULED = 2
TASK_STATUS_SUCCEEDED = 3
TASK_STATUS_FAILED = 4
TASK_STATUS_ABORTED = 5
TASK_STATUS_CANCELLED = 8
TASK_STATUS_TIMEOUTED = 10
TASK_STATUS_CANCELLED_TEMPORARILY = 11

TASK_STATUS_LABELS = {
    TASK_STATUS_INITIALIZED: "已初始化",
    TASK_STATUS_RUNNING: "运行中",
    TASK_STATUS_SCHEDULED: "已调度",
    TASK_STATUS_SUCCEEDED: "已成功",
    TASK_STATUS_FAILED: "失败",
    TASK_STATUS_ABORTED: "已中止",
    TASK_STATUS_CANCELLED: "已取消",
    TASK_STATUS_TIMEOUTED: "已超时",
    TASK_STATUS_CANCELLED_TEMPORARILY: "临时取消",
}

TERMINAL_FAILED_TASK_STATUSES = {
    TASK_STATUS_FAILED,
    TASK_STATUS_ABORTED,
    TASK_STATUS_CANCELLED,
    TASK_STATUS_TIMEOUTED,
}


@dataclass
class ChassisConfig:
    """底盘 HTTP 服务配置。"""

    base_url: str
    client_id: str
    http_timeout_sec: float
    navigation_timeout_sec: int
    linear_speed_limit: float


@dataclass
class Pose2D:
    """统一的二维位姿表达。"""

    x: float
    y: float
    yaw: float


def build_chassis_config(
    base_url: str = DEFAULT_CHASSIS_URL,
    client_id: str = DEFAULT_CLIENT_ID,
    http_timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
    navigation_timeout_sec: int = DEFAULT_NAVIGATION_TIMEOUT_SEC,
    linear_speed_limit: float = DEFAULT_LINEAR_SPEED_LIMIT,
) -> ChassisConfig:
    """根据行为树参数构造底盘配置，并保留统一默认值。"""
    return ChassisConfig(
        base_url=base_url,
        client_id=client_id,
        http_timeout_sec=http_timeout_sec,
        navigation_timeout_sec=navigation_timeout_sec,
        linear_speed_limit=linear_speed_limit,
    )


def post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_sec: float,
    description: str,
) -> Dict[str, Any]:
    """统一发送 JSON POST，并把网络层异常转成更一致的调用体验。"""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("HTTP 运行时需要 requests，请在当前 Python 环境安装 requests") from exc

    # 关键步骤：所有 HTTP POST 都收口到这里，方便统一超时和统一错误处理。
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_sec,
    )
    response.raise_for_status()

    response_payload = response.json()
    if not isinstance(response_payload, dict):
        raise RuntimeError(f"{description} response is not JSON object: {response_payload!r}")
    return response_payload


def to_float(value: Any) -> Optional[float]:
    """尽量把接口字段转成 float，失败时返回 None 而不是直接抛异常。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_chassis_current_pose(payload: Dict[str, Any]) -> Pose2D:
    """从底盘 current pose 响应中提取 global 位姿。"""
    candidates = []
    pose_info = payload.get("poseInfo")
    if isinstance(pose_info, dict):
        poi = pose_info.get("poi")
        if isinstance(poi, dict):
            candidates.append(poi)
        candidates.append(pose_info)

    pose = payload.get("pose")
    if isinstance(pose, dict):
        candidates.append(pose)
    candidates.append(payload)

    for candidate in candidates:
        x = to_float(candidate.get("x"))
        y = to_float(candidate.get("y"))
        yaw = to_float(candidate.get("yaw", candidate.get("angle")))
        if x is None or y is None or yaw is None:
            continue
        return Pose2D(x=x, y=y, yaw=yaw)

    raise RuntimeError(f"unable to extract chassis current pose: {payload}")


def post_chassis_current_pose(config: ChassisConfig) -> Dict[str, Any]:
    """请求底盘当前位姿的原始 JSON。"""
    request_url = "/localization/get_current_pose"
    # 关键步骤：导航目标必须在 global 坐标系下，因此先读取底盘定位结果作为转换基准。
    payload = {
        "clientId": config.client_id,
        "requestId": str(uuid.uuid4()),
        "requestUrl": request_url,
    }
    url = f"{config.base_url.rstrip('/')}{request_url}"
    return post_json(url, payload, config.http_timeout_sec, "current_pose")


def get_chassis_current_pose(config: ChassisConfig) -> Pose2D:
    """请求并解析底盘当前位姿，返回统一的 Pose2D。"""
    response_payload = post_chassis_current_pose(config)
    pose = extract_chassis_current_pose(response_payload)
    print(
        f"chassis current pose x={pose.x:.3f} y={pose.y:.3f} yaw={pose.yaw:.3f}",
        flush=True,
    )
    return pose


def post_chassis_navigation(
    config: ChassisConfig,
    task_id: str,
    x: float,
    y: float,
    yaw: float,
    coarse: bool = True,
) -> Dict[str, Any]:
    """创建一次导航任务，返回底盘原始响应。"""
    # 关键步骤：这里只创建导航任务，等待任务结束由调用方按 tick 或循环查询状态完成。
    payload = {
        "clientId": config.client_id,
        "coarse": coarse,
        "forbiddenAreaList": [],
        "inverseTrajectory": False,
        "linearSpeedLimit": config.linear_speed_limit,
        "navigationMode": DEFAULT_NAVIGATION_MODE,
        "poi": {
            "id": DEFAULT_POI_ID,
            "type": DEFAULT_POI_TYPE,
            "x": x,
            "y": y,
            "yaw": yaw,
        },
        "requestId": task_id,
        "requestUrl": "",
        "timeout": config.navigation_timeout_sec,
        "useOfflineMapTrajectory": False,
    }
    url = f"{config.base_url.rstrip('/')}/navigation/create_navigation_task"
    print(
        "create navigation task "
        f"request_id={task_id} "
        f"x={x:.3f} y={y:.3f} yaw={yaw:.3f} "
        f"coarse={coarse} url={url}",
        flush=True,
    )
    return post_json(url, payload, config.http_timeout_sec, "navigation")


def extract_navigation_task_id(payload: Dict[str, Any]) -> str:
    """从创建导航任务的返回值里提取 taskInstanceId。"""
    if payload.get("succeeded") is not True:
        raise RuntimeError(f"create navigation task failed: {payload}")

    task_info = payload.get("navigationTaskInfo")
    if not isinstance(task_info, dict):
        raise RuntimeError(f"navigation response missing navigationTaskInfo: {payload}")

    task_instance_id = task_info.get("taskInstanceId")
    if not isinstance(task_instance_id, str) or not task_instance_id.strip():
        raise RuntimeError(f"navigation response missing taskInstanceId: {payload}")

    return task_instance_id.strip()


def post_navigation_task_status(config: ChassisConfig, task_instance_id: str) -> Dict[str, Any]:
    """查询某个导航任务当前状态。"""
    request_url = "/navigation/query_navigation_task_status"
    payload = {
        "clientId": config.client_id,
        "requestId": str(uuid.uuid4()),
        "requestUrl": request_url,
        "navigationTaskInfo": {
            "taskInstanceId": task_instance_id,
        },
    }
    url = f"{config.base_url.rstrip('/')}{request_url}"
    return post_json(url, payload, config.http_timeout_sec, "navigation_status")


def extract_task_status(payload: Dict[str, Any]) -> int:
    """从导航状态响应中提取整数 taskStatus。"""
    if payload.get("succeeded") is not True:
        raise RuntimeError(f"query navigation task status failed: {payload}")

    task_status_info = payload.get("taskStatusInfo")
    if not isinstance(task_status_info, dict):
        raise RuntimeError(f"navigation status response missing taskStatusInfo: {payload}")

    task_status = task_status_info.get("taskStatus")
    if not isinstance(task_status, int):
        raise RuntimeError(f"navigation status response missing taskStatus: {payload}")

    return task_status
