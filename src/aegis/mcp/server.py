"""标准 SSE 网关：认证、会话归属与 REST 共用领域服务。"""

from typing import Annotated
from urllib.parse import urlsplit

import structlog
from fastapi import FastAPI, HTTPException
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.authentication import AuthCredentials
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from aegis.agent.state import access_fingerprint
from aegis.api.v1.auth import resolve_access
from aegis.core.exceptions import AegisError
from aegis.core.trace_context import get_trace_id, trace_id_var
from aegis.schemas.gateway import (
    CreateTicketRequest,
    SearchRequest,
    SearchResponse,
    VerifySKURequest,
    VerifySKUResponse,
)
from aegis.workers.tickets import TicketResult


class MCPAuthentication:
    """握手与消息投递均鉴权，SDK 按完整权限指纹拒绝跨身份会话投递。"""

    def __init__(self, app: ASGIApp, application: FastAPI) -> None:
        self.app = app
        self.application = application

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        try:
            access = resolve_access(request, self.application.state.settings)
        except HTTPException as error:
            await JSONResponse(
                {"message": error.detail, "trace_id": get_trace_id()},
                status_code=error.status_code,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
            return
        scope.setdefault("state", {})["access_context"] = access
        scope["auth"] = AuthCredentials(["authenticated"])
        scope["user"] = AuthenticatedUser(
            AccessToken(token="", client_id=access_fingerprint(access), scopes=[])
        )
        await self.app(scope, receive, send)


def create_mcp_server(application: FastAPI) -> FastMCP:
    """所有工具只进行契约适配，检索、核验和工单算法由领域服务唯一实现。"""
    public = urlsplit(application.state.settings.public_base_url)
    server = FastMCP(
        "星澜智选电商企业知识与履约网关",
        instructions="先检索授权知识，再核验商品参数；需要人工处理时使用幂等键创建工单。",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", public.netloc],
            allowed_origins=[
                "http://127.0.0.1:*",
                "http://localhost:*",
                "http://[::1]:*",
                f"{public.scheme}://{public.netloc}",
            ],
        ),
    )

    async def invoke(method: str, body, ctx: Context):
        request = ctx.request_context.request
        if not isinstance(request, Request):
            raise ValueError("工具调用必须携带已认证的 HTTP 上下文")
        access = resolve_access(request, application.state.settings)
        # SSE 服务任务继承握手上下文；每次调用改用消息 POST 的追踪标识。
        token = trace_id_var.set(request.state.trace_id)
        try:
            return await getattr(application.state.gateway_services, method)(body, access)
        except (AegisError, HTTPException) as error:
            message = error.message if isinstance(error, AegisError) else error.detail
            raise ValueError(message) from None
        except Exception:
            structlog.get_logger(__name__).exception("MCP 领域调用失败", tool_method=method)
            raise ValueError("工具调用失败，请携带追踪标识联系管理员") from None
        finally:
            trace_id_var.reset(token)

    @server.tool(
        description="企业混合检索：使用认证身份的部门与职级执行前置 ACL 过滤，"
        "返回父块原文、标题路径、精排降级标记与耗时。调用参数不能提升权限。",
    )
    async def search_enterprise_kb(
        body: Annotated[SearchRequest, Field(description="查询词、召回数量与可选文档类型")],
        ctx: Context,
    ) -> SearchResponse:
        return await invoke("search", body, ctx)

    @server.tool(
        description="SKU 确定性核验：先检查父块访问权限，再按目标商品行核对候选回答，"
        "返回标准量纲数值、真值明细、通过状态和纠正提示。禁止跨型号借用参数。",
    )
    async def verify_sku_specs(
        body: Annotated[VerifySKURequest, Field(description="候选回答、目标 SKU 与父块标识")],
        ctx: Context,
    ) -> VerifySKUResponse:
        return await invoke("verify_sku", body, ctx)

    @server.tool(
        description="创建履约工单：按高危词确定优先级，按认证身份隔离幂等键；"
        "24 小时缓存与数据库唯一键防止重复创建，并通过事务发件箱写入工单事件流。",
    )
    async def create_ops_ticket(
        body: Annotated[CreateTicketRequest, Field(description="幂等键、会话、问题描述和已知槽位")],
        ctx: Context,
    ) -> TicketResult:
        return await invoke("create_ticket", body, ctx)

    return server


def mount_mcp(application: FastAPI) -> None:
    """发布 /mcp/sse 及 SDK 宣告的 /mcp/messages/ 消息端点。"""
    server = create_mcp_server(application)
    application.state.mcp_server = server
    application.mount("/mcp", MCPAuthentication(server.sse_app(), application), name="mcp")
