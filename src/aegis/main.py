"""应用工厂、资源生命周期与全局错误处理。"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp

from aegis.agent.runtime import create_workflow
from aegis.agent.services import GatewayServices
from aegis.api.v1.router import router
from aegis.config import Settings, get_settings
from aegis.core.exceptions import AegisError
from aegis.core.logging import configure_logging
from aegis.core.trace_context import TraceContextMiddleware, get_trace_id
from aegis.db.qdrant_client import create_qdrant_client
from aegis.db.redis_client import create_redis_client
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.mcp.server import mount_mcp
from aegis.schemas.common import ERROR_RESPONSES


class AegisFastAPI(FastAPI):
    """将追踪层放在服务器异常处理层外，覆盖包括 500 在内的全部响应。"""

    def __init__(self, *, openapi_version: str = "3.1.0", **kwargs) -> None:
        super().__init__(**kwargs)
        self.openapi_version = openapi_version

    def build_middleware_stack(self) -> ASGIApp:
        return TraceContextMiddleware(super().build_middleware_stack())


def _error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"trace_id": get_trace_id(), "error_code": error_code, "message": message},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造独立应用；连接只在 lifespan 内初始化，并按逆序释放。"""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            engine = create_mysql_engine(settings)
            stack.push_async_callback(engine.dispose)
            application.state.mysql_engine = engine
            application.state.session_factory = create_session_factory(engine)
            redis = create_redis_client(settings)
            stack.push_async_callback(redis.aclose)
            application.state.redis = redis
            qdrant = create_qdrant_client(settings)
            stack.push_async_callback(qdrant.close)
            application.state.qdrant = qdrant
            application.state.workflow = create_workflow(
                settings, redis, qdrant, application.state.session_factory, stack
            )
            application.state.gateway_services = GatewayServices(
                application.state.workflow, application.state.session_factory, settings
            )
            structlog.get_logger(__name__).info("application_started")
            try:
                yield
            finally:
                structlog.get_logger(__name__).info("application_stopping")

    # 统一 JSON 错误响应优先于调试页，避免异常细节泄露到 HTTP 调用方。
    application = AegisFastAPI(
        title="星澜智选电商企业知识与履约工单平台",
        version="0.4.0",
        description="支持企业多租户知识检索、确定性商品规格核验、幂等履约与受控多轮排障。"
        "私有化 Dify 可导入本 OpenAPI 文档并配置管理员颁发的 Bearer 访问令牌。",
        openapi_version="3.1.0",
        servers=[
            {
                "url": settings.public_base_url,
                "description": "本地联调服务，部署时替换为私有网关地址",
            }
        ],
        openapi_tags=[
            {"name": "企业知识库", "description": "带前置权限过滤的企业混合检索"},
            {"name": "商品参数护栏", "description": "商品行级量纲与数值一致性核验"},
            {"name": "履约工单", "description": "幂等工单创建与可靠事件投递"},
            {"name": "受控工作流", "description": "受步数约束的多轮问答与排障"},
            {"name": "健康检查", "description": "服务存活与依赖就绪探测"},
        ],
        lifespan=lifespan,
        debug=False,
        responses=ERROR_RESPONSES,
    )
    # FastAPI 的版本属性单独锁定，避免框架默认值升级改变对外契约。
    application.openapi_version = "3.1.0"
    application.state.settings = settings
    application.include_router(router)
    mount_mcp(application)

    @application.exception_handler(AegisError)
    async def aegis_error_handler(request: Request, exc: AegisError) -> JSONResponse:
        return _error_response(exc.status_code, exc.error_code, exc.message)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, "VALIDATION_ERROR", "请求参数校验失败")

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        response = _error_response(exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail))
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        structlog.get_logger(__name__).error("unhandled_request_error", exc_info=exc)
        return _error_response(500, "INTERNAL_SERVER_ERROR", "服务内部错误")

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return application


app = create_app()
