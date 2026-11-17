"""存活探针与具备独立超时边界的并发就绪探针。"""

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from aegis.core.trace_context import get_trace_id
from aegis.schemas.common import LiveResponse, ReadyResponse

router = APIRouter(prefix="/health", tags=["健康检查"])


@router.get(
    "/live",
    response_model=LiveResponse,
    summary="检查服务进程存活状态",
    description="仅验证服务进程可以响应，不访问外部依赖。",
    operation_id="check_service_live",
)
async def live() -> dict[str, str | None]:
    """仅验证服务进程可响应，不访问外部依赖。"""
    return {"status": "ok", "trace_id": get_trace_id()}


async def _probe(
    name: str, action: Callable[[], Awaitable[Any]], timeout_seconds: float
) -> tuple[str, dict[str, Any]]:
    started = perf_counter()
    result: dict[str, Any] = {"status": "ok"}
    try:
        async with asyncio.timeout(timeout_seconds):
            await action()
    except Exception as exc:
        # 返回稳定错误类别，避免把凭据、连接地址或 SQL 细节暴露到探针响应。
        result.update(
            status="error", error="timeout" if isinstance(exc, TimeoutError) else "unavailable"
        )
        structlog.get_logger(__name__).warning(
            "dependency_probe_failed", component=name, error_type=type(exc).__name__
        )
    result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return name, result


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="检查服务依赖就绪状态",
    description="并行探测 MySQL、Redis 与 Qdrant；任一失败即返回未就绪状态。",
    operation_id="check_service_ready",
    responses={503: {"model": ReadyResponse, "description": "至少一个依赖不可用"}},
)
async def ready(request: Request) -> JSONResponse:
    """同时探测三项基础设施，任一失败即拒绝接收业务流量。"""

    async def mysql_ping() -> None:
        async with request.app.state.mysql_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def redis_ping() -> None:
        if not await request.app.state.redis.ping():
            raise ConnectionError("Redis 未确认 PING")

    async def qdrant_ping() -> None:
        await request.app.state.qdrant.get_collections()

    timeout = request.app.state.settings.dependency_timeout_seconds
    components = dict(
        await asyncio.gather(
            _probe("mysql", mysql_ping, timeout),
            _probe("redis", redis_ping, timeout),
            _probe("qdrant", qdrant_ping, timeout),
        )
    )
    healthy = all(component["status"] == "ok" for component in components.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "error",
            "trace_id": get_trace_id(),
            "components": components,
        },
    )
