"""HTTP service tree 的通用 Pydantic 模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


def parse_number(value, field_name):
    """
    解析数字字段。
    :param value: 待解析的原始字段值。
    :param field_name: 用于错误提示的字段名。
    :return: float 类型数字。
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number")


class Pose2DModel(BaseModel):
    """二维位姿模型。"""

    x: float
    y: float
    yaw: float

    @field_validator("x", "y", "yaw", mode="before")
    @classmethod
    def validate_number(cls, value, info):
        """
        校验位姿数值字段。
        :param value: x/y/yaw 的原始值。
        :param info: Pydantic 字段校验信息。
        :return: float 类型字段值。
        """
        return parse_number(value, info.field_name)


class PolygonPointModel(BaseModel):
    """二维多边形点模型。"""

    x: float
    y: float

    @field_validator("x", "y", mode="before")
    @classmethod
    def validate_number(cls, value, info):
        """
        校验多边形点数值字段。
        :param value: x/y 的原始值。
        :param info: Pydantic 字段校验信息。
        :return: float 类型字段值。
        """
        return parse_number(value, info.field_name)


class MoveBoxRequestModel(BaseModel):
    """start_move_box 请求模型。"""

    naviPoseFindBox: Pose2DModel
    validPolygon: List[PolygonPointModel]
    naviPosePlaceBox: Pose2DModel
    boxPosePlaceCenter: Pose2DModel
    heightPlacePlane: float

    @field_validator("heightPlacePlane", mode="before")
    @classmethod
    def validate_height_place_plane(cls, value):
        """
        校验放置平面高度。
        :param value: heightPlacePlane 原始值。
        :return: float 类型高度。
        """
        return parse_number(value, "heightPlacePlane")

    @field_validator("validPolygon", mode="before")
    @classmethod
    def normalize_valid_polygon(cls, value):
        """
        归一化有效区域多边形，兼容 [x, y] 和 {"x": x, "y": y} 两种点格式。
        :param value: validPolygon 原始值。
        :return: 统一成 dict 点列表后的多边形。
        """
        if not isinstance(value, list):
            raise ValueError("validPolygon must contain at least 3 points")
        polygon = []
        for point in value:
            if isinstance(point, dict):
                polygon.append(point)
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                polygon.append({"x": point[0], "y": point[1]})
            else:
                raise ValueError("validPolygon points must be [x,y] or {x,y}")
        return polygon

    @field_validator("validPolygon")
    @classmethod
    def validate_valid_polygon(cls, value):
        """
        校验有效区域至少包含 3 个点。
        :param value: 归一化后的多边形点列表。
        :return: 校验通过的多边形点列表。
        """
        if len(value) < 3:
            raise ValueError("validPolygon must contain at least 3 points")
        return value


class DirectGraspPlaceMemoryRequestModel(BaseModel):
    """基于 YOLO 记忆的直接抓箱放箱 HTTP 请求模型。

    与 ``MoveBoxRequestModel`` 的 C 点精确放置不同，此任务沿用
    ``move_box_full_direct_grasp_place_memory`` 的业务逻辑：机器人从 A 点
    搜箱、通过 YOLO/FP 对齐直接抓取，再导航到 B 点按指定高度放置，并
    返回 A 点继续下一轮。
    """

    naviPoseFindBox: Pose2DModel
    validPolygon: List[PolygonPointModel]
    naviPosePlaceBox: Pose2DModel
    heightPlacePlane: float
    targetCount: int = 1

    @field_validator("heightPlacePlane", mode="before")
    @classmethod
    def validate_height_place_plane(cls, value):
        return parse_number(value, "heightPlacePlane")

    @field_validator("targetCount", mode="before")
    @classmethod
    def validate_target_count(cls, value):
        if isinstance(value, bool):
            raise ValueError("targetCount must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("targetCount must be an integer")
        if not numeric.is_integer() or numeric < 1:
            raise ValueError("targetCount must be greater than or equal to 1")
        return int(numeric)

    @field_validator("validPolygon", mode="before")
    @classmethod
    def normalize_valid_polygon(cls, value):
        return MoveBoxRequestModel.normalize_valid_polygon(value)

    @field_validator("validPolygon")
    @classmethod
    def validate_valid_polygon(cls, value):
        return MoveBoxRequestModel.validate_valid_polygon(value)


class GraspAndPlaceRequestModel(BaseModel):
    """start_grasp_and_place 请求模型。"""

    naviPoseGrasp: Pose2DModel
    naviPosePlace: Pose2DModel
    heightGraspPlane: float
    heightPlacePlane: float
    targetCount: int

    @field_validator("heightGraspPlane", mode="before")
    @classmethod
    def validate_height_grasp_plane(cls, value):
        """
        校验抓取平面高度。
        :param value: heightGraspPlane 原始值。
        :return: float 类型高度。
        """
        return parse_number(value, "heightGraspPlane")

    @field_validator("heightPlacePlane", mode="before")
    @classmethod
    def validate_height_place_plane(cls, value):
        """
        校验放置平面高度。
        :param value: heightPlacePlane 原始值。
        :return: float 类型高度。
        """
        return parse_number(value, "heightPlacePlane")

    @field_validator("targetCount", mode="before")
    @classmethod
    def validate_target_count(cls, value):
        """
        校验目标搬运数量。
        :param value: targetCount 原始值。
        :return: int 类型目标数量。
        """
        if isinstance(value, bool):
            raise ValueError("targetCount must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("targetCount must be an integer")
        if not numeric.is_integer():
            raise ValueError("targetCount must be an integer")
        count = int(numeric)
        if count < 1:
            raise ValueError("targetCount must be greater than or equal to 1")
        return count


class NavigationRequestModel(BaseModel):
    """start_navigation 请求模型。"""

    targetPose: Pose2DModel


class StartTaskResponseModel(BaseModel):
    """任务启动接口响应。"""

    success: bool
    taskId: str
    status: str
    message: str
    taskName: str = ""


class HealthResponseModel(BaseModel):
    """健康检查接口响应。"""

    success: bool
    message: str
    endpoints: List[str]


class TaskStatusResponseModel(BaseModel):
    """任务状态查询接口响应。"""

    success: bool
    taskId: str
    status: str
    message: str
    taskName: str = ""
    currentNode: Optional[Dict[str, Any]] = None
    failedNode: Optional[Dict[str, Any]] = None
    rootStatus: Optional[str] = None
    executionState: Optional[str] = None
    tickCount: int = 0
    liveRuntime: Optional[Any] = None
    createdAt: Optional[float] = None
    finishedAt: Optional[float] = None
