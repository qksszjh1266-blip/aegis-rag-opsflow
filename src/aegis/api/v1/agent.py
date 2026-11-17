"""受控对话入口；权限只接受认证中间件注入，缺少认证时拒绝服务。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from aegis.api.v1.auth import authenticated_access
from aegis.schemas.agent import AccessContext, AgentRequest, AgentResponse

router = APIRouter(prefix="/api/v1/agent", tags=["受控工作流"])


@router.post("", response_model=AgentResponse, include_in_schema=False)
@router.post(
    "/chat",
    response_model=AgentResponse,
    summary="受控多轮排障与知识问答",
    description="持有会话互斥锁，先执行高危词短路，再运行最多八步、两轮追问与十轮压缩的"
    "状态机；检索后生成回答，执行 SKU 行级量纲核验，最多重生成一次，并持久化审计日志。",
    operation_id="agent_chat",
)
async def run_agent(
    body: AgentRequest,
    request: Request,
    access: Annotated[AccessContext, Depends(authenticated_access)],
) -> AgentResponse:
    return await request.app.state.workflow.run(body, access)
