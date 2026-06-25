"""Reusable business helpers shared by the behaviour-tree HTTP nodes.

这个文件位于“业务函数层”：
- 行为树叶子节点会调用这里的函数
- 这里只保留树运行时真正需要的底层能力函数
"""

import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


# =========================
# 默认参数与协议常量
# =========================
# 这一段集中定义：
# 1. HTTP 服务默认地址
# 2. 导航协议里固定会用到的字段
# 3. 对正逻辑里的默认容差、增益和限幅
#
# 行为树 JSON 不显式覆盖时，节点就会回落到这里的默认值。

# 使用坐标点导航，不从底盘点位数据库加载 POI。
DEFAULT_POI_ID = 0
# 普通目标点类型，保持和底盘调试客户端的默认值一致。
DEFAULT_POI_TYPE = 0
# 1 表示普通绕障到点导航，2 是循迹模式，这里固定使用普通导航。
DEFAULT_NAVIGATION_MODE = 1
# 底盘 m1d HTTP 服务地址。
DEFAULT_CHASSIS_URL = "http://192.168.168.125:8081"
# 本项目抓箱 HTTP 服务地址。
DEFAULT_GRAB_URL = "http://127.0.0.1:8080"
# 底盘接口要求的客户端标识。
DEFAULT_CLIENT_ID = "android-app"
# 底盘导航和状态查询的单次 HTTP 超时时间。
DEFAULT_HTTP_TIMEOUT_SEC = 8.0
# 抓箱动作可能耗时较长，因此单独给抓箱请求更长超时。
DEFAULT_GRAB_TIMEOUT_SEC = 120.0
# 底盘导航任务自身的超时时间，同时作为等待任务成功的总超时。
DEFAULT_NAVIGATION_TIMEOUT_SEC = 600
# 底盘线速度上限。
DEFAULT_LINEAR_SPEED_LIMIT = 0.6
# 查询导航任务状态的轮询间隔。
DEFAULT_POLL_INTERVAL_SEC = 0.2
# YOLO 远距离检测后，底盘目标点距离箱子中心的默认距离，单位 m。
DEFAULT_YOLO_TARGET_DISTANCE_M = 1.5
# FoundationPose 精定位后，底盘最终抓取站位距离箱子中心的默认距离，单位 m。
DEFAULT_FP_TARGET_DISTANCE_M = 1.0
# 箱子中心距离机体中轴的允许横向偏差，单位 m。
DEFAULT_BOX_ALIGN_TOLERANCE_M = 0.06
# 抓箱前底盘纵向上距离箱子中心的目标距离，单位 m。
DEFAULT_BOX_TARGET_DISTANCE_M = 0.4
# 箱子中心纵向距离目标值的允许偏差，单位 m。
DEFAULT_BOX_LONGITUDINAL_TOLERANCE_M = 0.06
# 底盘朝向箱子长边的允许角度偏差，单位 deg。
DEFAULT_BOX_YAW_TOLERANCE_DEG = 5.0
# 抓箱前最多执行几次检测和侧向调整。
DEFAULT_BOX_ALIGN_MAX_ATTEMPTS = 3

# 底盘导航任务状态码，由 /navigation/query_navigation_task_status 返回。
# 这些数值来自底盘 HTTP 服务约定，行为树只负责读取并转换成 RUNNING/SUCCESS/FAILURE。
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

# =========================
# 业务配置与数据结构
# =========================
# 这些 dataclass 是树节点和底层函数之间共享的“统一输入/输出格式”。
# 节点层负责把 JSON 参数转成这些对象，底层函数只关心对象字段本身。


@dataclass
class ChassisConfig:
    """底盘 HTTP 服务配置。"""

    base_url: str
    client_id: str
    http_timeout_sec: float
    navigation_timeout_sec: int
    linear_speed_limit: float


@dataclass
class GrabConfig:
    """抓取 HTTP 服务配置。"""

    base_url: str
    http_timeout_sec: float
    arm: Optional[str]


@dataclass
class Pose2D:
    """统一的二维位姿表达。"""

    x: float
    y: float
    yaw: float


@dataclass
class BoxAlignConfig:
    """抓箱前底盘对正时使用的参数集合。"""

    enabled: bool
    tolerance_m: float
    target_distance_m: float
    longitudinal_tolerance_m: float
    yaw_tolerance_deg: float
    max_attempts: int


@dataclass
class BoxAlignmentMeasurement:
    """把 detect_box 返回结果翻译成对正业务需要的结构。"""

    lateral_error_m: float
    longitudinal_error_m: float
    yaw_error_deg: float
    box_center_x_m: float
    box_center_y_m: float
    front_axis_x: float
    front_axis_y: float


# =========================
# 配置构造函数
# =========================
# 这几组 build_* 函数的意义不是“多包一层”，而是把：
# - 默认值
# - 合法值裁剪
# - JSON / Python 参数转换
# 收口到一起，避免节点层到处重复写同样的校验逻辑。

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


def build_grab_config(
    base_url: str = DEFAULT_GRAB_URL,
    http_timeout_sec: float = DEFAULT_GRAB_TIMEOUT_SEC,
    arm: Optional[str] = None,
) -> GrabConfig:
    """根据行为树参数构造抓取服务配置。"""
    return GrabConfig(
        base_url=base_url,
        http_timeout_sec=http_timeout_sec,
        arm=arm,
    )


def build_align_config(
    enabled: bool = True,
    tolerance_m: float = DEFAULT_BOX_ALIGN_TOLERANCE_M,
    target_distance_m: float = DEFAULT_BOX_TARGET_DISTANCE_M,
    longitudinal_tolerance_m: float = DEFAULT_BOX_LONGITUDINAL_TOLERANCE_M,
    yaw_tolerance_deg: float = DEFAULT_BOX_YAW_TOLERANCE_DEG,
    max_attempts: int = DEFAULT_BOX_ALIGN_MAX_ATTEMPTS,
) -> BoxAlignConfig:
    """根据行为树参数构造抓箱前对正配置，并做基础限幅。"""
    return BoxAlignConfig(
        enabled=bool(enabled),
        tolerance_m=max(0.0, tolerance_m),
        target_distance_m=max(0.0, target_distance_m),
        longitudinal_tolerance_m=max(0.0, longitudinal_tolerance_m),
        yaw_tolerance_deg=max(0.0, yaw_tolerance_deg),
        max_attempts=max(1, max_attempts),
    )


# =========================
# 通用工具函数
# =========================
# 这部分是底层计算和返回解析时反复用到的小工具。

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

    # 所有 HTTP POST 都收口到这里，方便统一超时和统一错误处理。
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


def normalize_angle_deg(angle_deg: float) -> float:
    """把角度归一化到 [-180, 180) 区间。"""
    while angle_deg >= 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def normalize_axis_angle_rad(angle_rad: float) -> float:
    """把“轴向方向角”归一化到 [-pi/2, pi/2)。

    这里不是普通朝向角归一化，而是针对“箱子长边方向向量”做的归一化。
    因为长边正向和反向在业务上通常等价，只要保持与长边平行即可。
    """
    while angle_rad >= math.pi / 2.0:
        angle_rad -= math.pi
    while angle_rad < -math.pi / 2.0:
        angle_rad += math.pi
    return angle_rad


def transform_base_point_to_global(
    current_pose: Pose2D,
    base_x: float,
    base_y: float,
) -> Dict[str, float]:
    """将 base_link 下二维点转换到 global 坐标系。"""
    yaw_rad = math.radians(current_pose.yaw)
    return {
        "x": current_pose.x + math.cos(yaw_rad) * base_x - math.sin(yaw_rad) * base_y,
        "y": current_pose.y + math.sin(yaw_rad) * base_x + math.cos(yaw_rad) * base_y,
    }


def transform_global_point_to_base(
    current_pose: Pose2D,
    global_x: float,
    global_y: float,
) -> Dict[str, float]:
    """将 global 坐标系下二维点转换到 base_link 坐标系。"""
    yaw_rad = math.radians(current_pose.yaw)
    dx = global_x - current_pose.x
    dy = global_y - current_pose.y
    return {
        "x": math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy,
        "y": -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy,
    }


def build_yolo_approach_pose(
    current_pose: Pose2D,
    box_global_x: float,
    box_global_y: float,
    target_distance_m: float,
) -> Pose2D:
    """根据 YOLO 箱体 global 位置计算远距离粗靠近目标点。"""
    dx = box_global_x - current_pose.x
    dy = box_global_y - current_pose.y
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        raise RuntimeError("箱体中心与底盘当前位置重合，无法计算 YOLO 靠近目标")
    if distance <= target_distance_m:
        raise RuntimeError(
            "箱体距离 %.3fm 小于等于目标距离 %.3fm，无需继续 YOLO 靠近"
            % (distance, target_distance_m)
        )

    unit_x = dx / distance
    unit_y = dy / distance
    return Pose2D(
        x=box_global_x - unit_x * target_distance_m,
        y=box_global_y - unit_y * target_distance_m,
        yaw=normalize_angle_deg(math.degrees(math.atan2(unit_y, unit_x))),
    )


def build_fp_approach_pose(
    current_pose: Pose2D,
    box_center: Dict[str, float],
    front_axis: Dict[str, float],
    target_distance_m: float,
) -> Pose2D:
    """根据 FoundationPose 箱体中心和前向轴计算最终抓取站位。"""
    box_global_position = transform_base_point_to_global(
        current_pose,
        box_center["x"],
        box_center["y"],
    )
    axis_xy_norm = math.hypot(front_axis["x"], front_axis["y"])
    if axis_xy_norm < 1e-6:
        raise RuntimeError("FoundationPose frontAxis xy too small: %r" % front_axis)

    front_axis_angle_rad = normalize_axis_angle_rad(
        math.atan2(front_axis["y"], front_axis["x"])
    )
    target_yaw = normalize_angle_deg(
        current_pose.yaw + math.degrees(front_axis_angle_rad)
    )
    target_yaw_rad = math.radians(target_yaw)
    return Pose2D(
        x=box_global_position["x"] - math.cos(target_yaw_rad) * target_distance_m,
        y=box_global_position["y"] - math.sin(target_yaw_rad) * target_distance_m,
        yaw=target_yaw,
    )


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


# =========================
# 底盘位姿读取
# =========================
# 这一段负责从底盘服务获取当前位姿，并兼容不同字段层级。

def extract_chassis_current_pose(payload: Dict[str, Any]) -> Pose2D:
    # 不同服务端可能把位姿放在不同字段层级，
    # 这里按几个候选位置依次尝试提取，尽量兼容不同返回结构。
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


# =========================
# 底盘导航接口
# =========================
# 这里故意只保留“创建任务”和“查询状态”两个原子动作。
# 真正的同步等待由脚本模块处理，异步多 tick 推进由行为树节点处理。

def post_chassis_navigation(
    config: ChassisConfig,
    task_id: str,
    x: float,
    y: float,
    yaw: float,
    coarse: bool = True,
) -> Dict[str, Any]:
    """创建一次导航任务，返回底盘原始响应。"""
    # 这里发起的是“创建导航任务”请求。
    # 真正等待任务结束，是靠后续反复查询 task status 完成的。
    # 默认使用粗到点策略，精细到点节点需要在调用时显式传 coarse=False。
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
    # 关键步骤：记录每次导航请求的 coarse 参数，便于和底盘策略日志对齐排查。
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
    # 创建导航成功后，外层要靠 taskInstanceId 持续追踪这次任务。
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


# =========================
# 抓取侧 HTTP 接口
# =========================
# 这几组函数保持得很薄：只负责发请求、做最小成功判断、返回原始响应。
# 更高层的流程组合由节点层或脚本层完成。

def post_grab_box(config: GrabConfig) -> Dict[str, Any]:
    """调用抓箱接口。"""
    payload = {} if config.arm is None else {"arm": config.arm}
    url = f"{config.base_url.rstrip('/')}/grab_box"
    response_payload = post_json(url, payload, config.http_timeout_sec, "grab_box")
    if response_payload.get("success") is not True:
        raise RuntimeError(f"grab_box failed: {response_payload}")
    return response_payload


def post_detect_box(config: GrabConfig) -> Dict[str, Any]:
    """调用箱体检测接口。"""
    url = f"{config.base_url.rstrip('/')}/detect_box"
    response_payload = post_json(url, {}, config.http_timeout_sec, "detect_box")
    if response_payload.get("success") is not True:
        raise RuntimeError(f"detect_box failed: {response_payload}")
    return response_payload


def post_put_down(config: GrabConfig) -> Dict[str, Any]:
    """调用放箱接口。"""
    url = f"{config.base_url.rstrip('/')}/put_down"
    response_payload = post_json(url, {}, config.http_timeout_sec, "put_down")
    if response_payload.get("success") is not True:
        raise RuntimeError(f"put_down failed: {response_payload}")
    return response_payload


# =========================
# 对正误差解析与位姿修正
# =========================
# 这是抓箱业务里最“算法化”的一段。
# 输入是 detect_box 的原始返回，输出是：
# - 横向误差
# - 纵向误差
# - 朝向误差
# 以及必要时下一步该导航到哪里。

def extract_vector3(payload: Dict[str, Any], field_name: str) -> Dict[str, float]:
    """从响应里提取一个三维向量字段，例如 boxCenter / frontAxis。"""
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise RuntimeError(f"detect_box response missing {field_name}: {payload}")

    vector: Dict[str, float] = {}
    for axis_name in ("x", "y", "z"):
        axis_value = to_float(value.get(axis_name))
        if axis_value is None:
            raise RuntimeError(
                f"detect_box response missing {field_name}.{axis_name}: {payload}"
            )
        vector[axis_name] = axis_value
    return vector


def extract_box_alignment_measurement(
    detection_payload: Dict[str, Any],
    align_config: BoxAlignConfig,
) -> BoxAlignmentMeasurement:
    """把 detect_box 返回结果转换成对正逻辑可直接使用的误差结构。"""
    # detect_box 返回的是“箱体中心 + 朝向向量”，
    # 这里把它翻译成业务上更好理解的三种误差：
    # lateral / longitudinal / yaw。
    box_center = extract_vector3(detection_payload, "boxCenter")
    front_axis = extract_vector3(detection_payload, "frontAxis")

    axis_xy_norm = math.hypot(front_axis["x"], front_axis["y"])
    if axis_xy_norm < 1e-6:
        raise RuntimeError(f"detect_box response frontAxis xy too small: {detection_payload}")

    longitudinal_error_m = box_center["x"] - align_config.target_distance_m
    yaw_error_rad = normalize_axis_angle_rad(math.atan2(front_axis["y"], front_axis["x"]))
    return BoxAlignmentMeasurement(
        lateral_error_m=box_center["y"],
        longitudinal_error_m=longitudinal_error_m,
        yaw_error_deg=math.degrees(yaw_error_rad),
        box_center_x_m=box_center["x"],
        box_center_y_m=box_center["y"],
        front_axis_x=front_axis["x"],
        front_axis_y=front_axis["y"],
    )


def make_box_adjust_pose(
    current_pose: Pose2D,
    measurement: BoxAlignmentMeasurement,
    align_config: BoxAlignConfig,
) -> Pose2D:
    """根据箱体中心和前轴，一次性反推出机器人导航目标位姿。"""
    # 这里不再做增量式限幅微调，而是先把箱体中心从 base_link 转到世界坐标系，
    # 再沿箱体前轴反推 target_distance_m，得到机器人应该站立的位置。
    current_yaw_rad = math.radians(current_pose.yaw)
    box_world_x = (
        current_pose.x
        + math.cos(current_yaw_rad) * measurement.box_center_x_m
        - math.sin(current_yaw_rad) * measurement.box_center_y_m
    )
    box_world_y = (
        current_pose.y
        + math.sin(current_yaw_rad) * measurement.box_center_x_m
        + math.cos(current_yaw_rad) * measurement.box_center_y_m
    )

    front_axis_angle_rad = normalize_axis_angle_rad(
        math.atan2(measurement.front_axis_y, measurement.front_axis_x)
    )
    target_yaw = normalize_angle_deg(
        current_pose.yaw + math.degrees(front_axis_angle_rad)
    )
    target_yaw_rad = math.radians(target_yaw)

    return Pose2D(
        x=box_world_x - math.cos(target_yaw_rad) * align_config.target_distance_m,
        y=box_world_y - math.sin(target_yaw_rad) * align_config.target_distance_m,
        yaw=target_yaw,
    )


def is_box_aligned(
    measurement: BoxAlignmentMeasurement,
    align_config: BoxAlignConfig,
) -> bool:
    """判断当前检测误差是否已经满足抓取前的对正条件。"""
    return (
        abs(measurement.lateral_error_m) <= align_config.tolerance_m
        and abs(measurement.longitudinal_error_m) <= align_config.longitudinal_tolerance_m
        and abs(measurement.yaw_error_deg) <= align_config.yaw_tolerance_deg
    )
