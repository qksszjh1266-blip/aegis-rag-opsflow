"""REST 与 MCP 共用认证入口，凭据只映射服务端配置的权限。"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aegis.config import Settings
from aegis.schemas.agent import AccessContext

bearer = HTTPBearer(
    auto_error=False, scheme_name="企业网关凭据", description="由管理员分发的企业网关访问令牌"
)


def resolve_access(request: Request, settings: Settings) -> AccessContext:
    """信任认证适配器注入或服务端令牌映射，忽略客户端部门、角色请求头。"""
    context = getattr(request.state, "access_context", None)
    if isinstance(context, AccessContext):
        return context
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and token:
        context = settings.gateway_api_tokens.get(token)
        if context is not None:
            return context
    raise HTTPException(
        status_code=401,
        detail="需要经过认证的访问上下文",
        headers={"WWW-Authenticate": "Bearer"},
    )


def authenticated_access(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> AccessContext:
    """通过显式安全方案让 Dify 导入后能够配置统一 Bearer 凭据。"""
    return resolve_access(request, request.app.state.settings)
