"""请求链路上下文与最外层 ASGI 观测中间件。"""

import re
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from aegis.core.metrics import HTTP_REQUEST_DURATION, HTTP_REQUESTS

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,64}\Z")
_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}


def get_trace_id() -> str | None:
    """获取当前协程的链路标识；请求之外返回空值。"""
    return trace_id_var.get()


class TraceContextMiddleware:
    """覆盖异常处理中间件，保证未捕获异常的响应同样包含链路标识。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get("x-trace-id", "")
        # 限制长度及字符，避免日志注入和超出数据库中的 VARCHAR(64) 契约。
        trace_id = supplied if _TRACE_ID_PATTERN.fullmatch(supplied) else uuid4().hex
        token = trace_id_var.set(trace_id)
        scope.setdefault("state", {})["trace_id"] = trace_id
        started = perf_counter()
        status_code = 500

        async def send_with_trace(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Trace-ID"] = trace_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            try:
                route = getattr(scope.get("route"), "path", "__unmatched__")
                method = scope.get("method", "OTHER")
                method = method if method in _HTTP_METHODS else "OTHER"
                elapsed = perf_counter() - started
                labels = {"method": method, "route": route, "status_code": str(status_code)}
                HTTP_REQUESTS.labels(**labels).inc()
                HTTP_REQUEST_DURATION.labels(**labels).observe(elapsed)
                structlog.get_logger(__name__).info(
                    "http_request_completed", **labels, latency_ms=round(elapsed * 1000, 3)
                )
            finally:
                trace_id_var.reset(token)
