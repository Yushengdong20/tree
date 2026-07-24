"""基于 FastAPI 的通用任务 HTTP 服务。"""

import json
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tree.runtime.http_service.models import (
    DirectGraspPlaceMemoryRequestModel,
    GraspAndPlaceRequestModel,
    HealthResponseModel,
    MoveBoxRequestModel,
    NavigationRequestModel,
    StartTaskResponseModel,
    TaskStatusResponseModel,
)


def _format_validation_error(exc):
    """
    格式化 FastAPI/Pydantic 校验错误。
    :param exc: RequestValidationError 实例。
    :return: 对外返回的简洁错误消息。
    """
    messages = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        message = str(error.get("msg", "invalid request"))
        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)
    if not messages:
        return "request validation failed"
    return "; ".join(messages)


def _request_client_text(request):
    """
    格式化 HTTP 调用者地址。
    :param request: FastAPI 请求对象。
    :return: ip:port 文本，缺失时返回 unknown。
    """
    client = getattr(request, "client", None)
    if client is None:
        return "unknown"
    host = getattr(client, "host", None) or "unknown"
    port = getattr(client, "port", None)
    if port is None:
        return str(host)
    return f"{host}:{port}"


def _payload_text(payload):
    """
    将请求体转换成稳定 JSON 文本。
    :param payload: dict/list 或 Pydantic model。
    :return: JSON 文本，无法序列化时返回 repr。
    """
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(payload)


def _get_ros_logger(task_manager):
    """
    从 task_manager 取 ROS logger。
    :param task_manager: TaskManager 实例。
    :return: logger 或 None。
    """
    runner = getattr(task_manager, "runner", None)
    ros = getattr(runner, "ros", None)
    if ros is None or not hasattr(ros, "get_logger"):
        return None
    return ros.get_logger()


def _log_task_api_request(task_manager, request, endpoint, payload):
    """
    记录任务启动接口调用信息。
    :param task_manager: TaskManager 实例。
    :param request: FastAPI 请求对象。
    :param endpoint: 任务 endpoint。
    :param payload: 已校验请求体。
    """
    logger = _get_ros_logger(task_manager)
    if logger is None:
        return
    user_agent = request.headers.get("user-agent", "")
    logger.info(
        "Task API request: "
        f"endpoint={endpoint}, method={request.method}, "
        f"client={_request_client_text(request)}, user_agent={user_agent!r}, "
        f"payload={_payload_text(payload)}"
    )


async def _log_task_api_validation_error(task_manager, request, message):
    """
    记录任务接口参数校验失败信息。
    :param task_manager: TaskManager 实例。
    :param request: FastAPI 请求对象。
    :param message: 格式化后的错误信息。
    """
    logger = _get_ros_logger(task_manager)
    if logger is None:
        return
    try:
        body = await request.body()
        payload = body.decode("utf-8", errors="replace")
    except Exception as exc:
        payload = f"<failed to read body: {exc}>"
    user_agent = request.headers.get("user-agent", "")
    logger.warning(
        "Task API rejected: "
        f"endpoint={request.url.path}, method={request.method}, "
        f"client={_request_client_text(request)}, user_agent={user_agent!r}, "
        f"error={message}, payload={payload}"
    )


class TaskHttpServer:
    """独立承载 service tree 任务 API，避免与 visualization server 混用。"""

    def __init__(self, task_manager, registry, host="127.0.0.1", port=8766):
        """
        初始化 HTTP 服务。
        :param task_manager: TaskManager 实例。
        :param registry: TaskRegistry 实例，用于 endpoint 路由。
        :param host: 监听地址。
        :param port: 监听端口。
        """
        self.task_manager = task_manager
        self.registry = registry
        self.host = host
        self.port = int(port)
        self.app = self._make_app()
        self._server = None
        self._thread = None

    def start(self):
        """启动后台 HTTP 服务。"""
        if self._server is not None:
            return

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self):
        """关闭后台 HTTP 服务并释放端口。"""
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None

    def _make_app(self):
        task_manager = self.task_manager
        registry = self.registry

        app = FastAPI(
            title="MercuryTree Task API",
            description="MercuryTree service tree 任务 HTTP 接口。",
            version="0.1.0",
        )

        @app.exception_handler(RequestValidationError)
        async def handle_request_validation_error(
            request: Request,
            exc: RequestValidationError,
        ):
            """
            统一 FastAPI 请求校验失败响应。
            :param request: FastAPI 请求对象。
            :param exc: 请求校验异常。
            :return: 与旧 HTTP 服务兼容的 JSON 响应。
            """
            message = _format_validation_error(exc)
            await _log_task_api_validation_error(task_manager, request, message)
            return JSONResponse(
                status_code=200,
                content=StartTaskResponseModel(
                    success=False,
                    taskId="",
                    status="REJECTED",
                    message=message,
                ).model_dump(),
            )

        @app.get("/health", response_model=HealthResponseModel)
        async def health():
            """健康检查接口。"""
            return HealthResponseModel(
                success=True,
                message="task server running",
                endpoints=registry.endpoints(),
            )

        @app.post("/api/start_move_box", response_model=StartTaskResponseModel)
        async def start_move_box(request: Request, payload: MoveBoxRequestModel):
            """
            启动 move_box 任务。
            :param request: FastAPI 请求对象，用于记录调用者信息。
            :param payload: move_box 请求模型。
            :return: 任务启动响应。
            """
            try:
                # 关键步骤：FastAPI 已完成首轮校验，这里转成 dict 交给原任务管理流程。
                _log_task_api_request(task_manager, request, "/api/start_move_box", payload)
                return task_manager.start_task_by_endpoint(
                    "/api/start_move_box",
                    payload.model_dump(),
                )
            except Exception as exc:
                return StartTaskResponseModel(
                    success=False,
                    taskId="",
                    status="REJECTED",
                    message=str(exc),
                )

        @app.post(
            "/api/start_move_box_direct_grasp_place_memory",
            response_model=StartTaskResponseModel,
        )
        async def start_move_box_direct_grasp_place_memory(
            request: Request,
            payload: DirectGraspPlaceMemoryRequestModel,
        ):
            """启动基于 YOLO 记忆的有限次数直接抓箱放箱任务。"""
            endpoint = "/api/start_move_box_direct_grasp_place_memory"
            try:
                _log_task_api_request(task_manager, request, endpoint, payload)
                return task_manager.start_task_by_endpoint(endpoint, payload.model_dump())
            except Exception as exc:
                return StartTaskResponseModel(
                    success=False,
                    taskId="",
                    status="REJECTED",
                    message=str(exc),
                )

        @app.post("/api/start_grasp_and_place", response_model=StartTaskResponseModel)
        async def start_grasp_and_place(
            request: Request,
            payload: GraspAndPlaceRequestModel,
        ):
            """
            启动 grasp_and_place 任务。
            :param request: FastAPI 请求对象，用于记录调用者信息。
            :param payload: grasp_and_place 请求模型。
            :return: 任务启动响应。
            """
            try:
                # 关键步骤：FastAPI 已完成首轮校验，这里转成 dict 交给原任务管理流程。
                _log_task_api_request(
                    task_manager,
                    request,
                    "/api/start_grasp_and_place",
                    payload,
                )
                return task_manager.start_task_by_endpoint(
                    "/api/start_grasp_and_place",
                    payload.model_dump(),
                )
            except Exception as exc:
                return StartTaskResponseModel(
                    success=False,
                    taskId="",
                    status="REJECTED",
                    message=str(exc),
                )

        @app.post("/api/start_navigation", response_model=StartTaskResponseModel)
        async def start_navigation(request: Request, payload: NavigationRequestModel):
            """
            启动 navigation 任务。
            :param request: FastAPI 请求对象，用于记录调用者信息。
            :param payload: navigation 请求模型。
            :return: 任务启动响应。
            """
            try:
                # 关键步骤：FastAPI 已完成首轮校验，这里转成 dict 交给原任务管理流程。
                _log_task_api_request(task_manager, request, "/api/start_navigation", payload)
                return task_manager.start_task_by_endpoint(
                    "/api/start_navigation",
                    payload.model_dump(),
                )
            except Exception as exc:
                return StartTaskResponseModel(
                    success=False,
                    taskId="",
                    status="REJECTED",
                    message=str(exc),
                )

        @app.get("/api/task_status", response_model=TaskStatusResponseModel)
        async def task_status(taskId: str = ""):
            """
            查询任务状态。
            :param taskId: start 接口返回的任务 ID。
            :return: 任务状态响应。
            """
            return task_manager.get_task_status(taskId.strip())

        return app
