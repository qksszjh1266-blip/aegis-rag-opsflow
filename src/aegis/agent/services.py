"""双协议共享领域服务，统一授权回读、行级核验、幂等工单与审计。"""

import hashlib
import json
from time import perf_counter
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aegis.agent.state import SessionContextError, access_fingerprint
from aegis.agent.workflow import REQUIRED_SLOTS, ControlledWorkflow
from aegis.config import Settings
from aegis.core.exceptions import ResourceNotFoundError
from aegis.core.trace_context import get_trace_id
from aegis.db.models import KBDocument, KBParentChunk, RAGAuditLog, WorkOrderPriority
from aegis.guardrails.sku_verifier import (
    _cell_quantities,
    _identifiers,
    _normalize,
    extract_quantities,
    load_sku_rows,
)
from aegis.schemas.agent import AccessContext
from aegis.schemas.gateway import (
    CreateTicketRequest,
    NormalizedQuantity,
    ReferenceQuantity,
    SearchHit,
    SearchRequest,
    SearchResponse,
    VerifySKURequest,
    VerifySKUResponse,
)
from aegis.workers.tickets import TicketResult


class GatewayServices:
    """协议适配器不持有算法；沿用工作流中的检索、核验器与工单服务。"""

    def __init__(
        self,
        workflow: ControlledWorkflow,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.workflow = workflow
        self.session_factory = session_factory
        self.settings = settings

    async def search(self, body: SearchRequest, access: AccessContext) -> SearchResponse:
        """部门与角色强制下推两路向量召回，调用方仅能额外缩小文档类型。"""
        started = perf_counter()
        result = await self.workflow.retriever.search(
            body.query,
            dept_scope=access.dept_scope,
            role_level=access.role_level,
            top_k=body.top_k,
            doc_type=body.doc_type,
        )
        return SearchResponse(
            hits=[
                SearchHit(
                    parent_id=hit.parent_id,
                    doc_code=hit.doc_code,
                    content_text=hit.content_text,
                    header_path=hit.header_path,
                    rrf_score=hit.rrf_score,
                    rerank_score=hit.rerank_score,
                )
                for hit in result.hits
            ],
            rerank_fallback=result.degraded_flag,
            latency_ms=(perf_counter() - started) * 1000,
            trace_id=result.trace_id,
        )

    async def verify_sku(self, body: VerifySKURequest, access: AccessContext) -> VerifySKUResponse:
        """父块标识不是授权凭据；回读前必须同时校验部门、职级与有效状态。"""
        started = perf_counter()
        trace_id = get_trace_id() or uuid4().hex
        async with self.session_factory() as session:
            parent_id = await session.scalar(
                select(KBParentChunk.parent_id)
                .join(KBDocument, KBParentChunk.doc_id == KBDocument.id)
                .where(
                    KBParentChunk.parent_id == body.parent_id,
                    KBDocument.dept_scope.in_(access.dept_scope),
                    KBDocument.min_role_level <= access.role_level,
                    KBDocument.status == "ACTIVE",
                )
            )
        if parent_id is None:
            raise ResourceNotFoundError("指定规格不存在或不在当前授权范围")
        rows = await load_sku_rows(self.session_factory, [parent_id])
        rows = [row for row in rows if _normalize(body.sku) in _identifiers(row)]
        if len(rows) != 1:
            raise ResourceNotFoundError("指定商品行不存在或无法唯一定位")
        result = self.workflow.verifier.verify(body.candidate_text, rows, query=body.sku)
        response = VerifySKUResponse(
            passed=result.passed,
            corrections=result.corrections,
            checked_count=result.checked_count,
            claims=[
                NormalizedQuantity(
                    text=item.text,
                    dimension=item.dimension,
                    values=[str(value) for value in item.values],
                    kind=item.kind,
                )
                for item in extract_quantities(body.candidate_text)
            ],
            reference_values=[
                ReferenceQuantity(
                    attribute=key,
                    text=item.text,
                    dimension=item.dimension,
                    values=[str(value) for value in item.values],
                    kind=item.kind,
                )
                for key, value in rows[0].items()
                for item in _cell_quantities(key, value)
            ],
            trace_id=trace_id,
        )
        await self._audit(
            trace_id,
            body.candidate_text,
            "SKU_VERIFIED_PASS" if result.passed else "SKU_VERIFICATION_FAILED",
            started,
            [parent_id],
        )
        return response

    async def create_ticket(self, body: CreateTicketRequest, access: AccessContext) -> TicketResult:
        """身份隔离防重键，前置判定高危优先级；数据库发件箱保证事件可恢复。"""
        started = perf_counter()
        trace_id = get_trace_id() or uuid4().hex
        identity = json.dumps([access_fingerprint(access), body.idempotency_key])
        key = "gateway:" + hashlib.sha256(identity.encode()).hexdigest()
        risks = self.workflow.precheck.check(body.message)
        priority = WorkOrderPriority.P0_EMERGENCY if risks else WorkOrderPriority.P2_NORMAL
        completeness = sum(bool(body.slots_payload.get(key)) for key in REQUIRED_SLOTS) / 3
        try:
            result = await self.workflow.tickets.create(
                idempotency_key=key,
                session_id=body.session_id,
                trace_id=trace_id,
                priority=priority,
                trigger_source="HIGH_RISK_PRECHECK" if risks else "GATEWAY_REQUEST",
                slots_payload={**body.slots_payload, "message": body.message},
                completeness_score=completeness,
            )
        except ValueError as error:
            raise SessionContextError() from error
        await self._audit(trace_id, body.message, "TICKET_CREATED", started, [])
        return result

    async def _audit(
        self,
        trace_id: str,
        query: str,
        action: str,
        started: float,
        parent_ids: list[str],
    ) -> None:
        async with self.session_factory.begin() as session:
            session.add(
                RAGAuditLog(
                    trace_id=trace_id,
                    query_text=query,
                    recalled_parent_ids=parent_ids,
                    guardrail_action=action,
                    latency_ms=round((perf_counter() - started) * 1000),
                    degraded_flag=False,
                )
            )
