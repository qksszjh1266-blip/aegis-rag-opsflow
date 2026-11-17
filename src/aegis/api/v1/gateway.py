"""面向 Dify 等调用方的强类型领域服务入口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from aegis.api.v1.auth import authenticated_access
from aegis.schemas.agent import AccessContext
from aegis.schemas.gateway import (
    CreateTicketRequest,
    SearchRequest,
    SearchResponse,
    VerifySKURequest,
    VerifySKUResponse,
)
from aegis.workers.tickets import TicketResult

router = APIRouter(prefix="/api/v1")
Access = Annotated[AccessContext, Depends(authenticated_access)]


@router.post(
    "/kb/search",
    tags=["企业知识库"],
    response_model=SearchResponse,
    summary="检索授权范围内的企业知识",
    description="部门与职级由认证上下文绑定并下推至 Qdrant 双路 ACL 过滤，返回完整父块、"
    "来源标题路径、精排降级标记与毫秒耗时。可按文档类型进一步缩小召回范围。",
    operation_id="search_enterprise_kb",
)
async def search(body: SearchRequest, request: Request, access: Access) -> SearchResponse:
    return await request.app.state.gateway_services.search(body, access)


@router.post(
    "/guardrails/verify-sku",
    tags=["商品参数护栏"],
    response_model=VerifySKUResponse,
    summary="核验目标商品行的物理量纲与数值",
    description="对父块重新执行部门与职级鉴权，精确选择目标 SKU 行，并使用确定性算法"
    "核对候选回复。返回规范化参数、参考真值、通过状态及差异纠正提示。",
    operation_id="verify_sku_specs",
)
async def verify_sku(body: VerifySKURequest, request: Request, access: Access) -> VerifySKUResponse:
    return await request.app.state.gateway_services.verify_sku(body, access)


@router.post(
    "/tickets",
    tags=["履约工单"],
    response_model=TicketResult,
    summary="幂等创建智能履约工单",
    description="按认证身份隔离幂等键，使用 24 小时 Redis 防重缓存及数据库唯一键。"
    "高危问题自动判为 P0_EMERGENCY；工单与事务发件箱一起提交，事件写入"
    " stream:ticket:events，Redis 暂不可用时由后台发布器恢复。",
    operation_id="create_ops_ticket",
)
async def create_ticket(
    body: CreateTicketRequest, request: Request, access: Access
) -> TicketResult:
    return await request.app.state.gateway_services.create_ticket(body, access)
