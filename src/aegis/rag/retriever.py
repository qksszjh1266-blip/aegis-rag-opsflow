"""ACL 前置过滤的双路召回、父块级 RRF 与可审计精排降级。"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aegis.core.circuit_breaker import RedisCircuitBreaker
from aegis.core.trace_context import get_trace_id
from aegis.db.models import DocumentType, RAGAuditLog
from aegis.llm.providers import EmbeddingProvider
from aegis.rag.reranker import Reranker, RerankScore, validate_scores
from aegis.rag.sparse import encode_query

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievalHit:
    """返回完整父块与结构化行，保留融合分数供审计或评测使用。"""

    parent_id: str
    doc_code: str
    content_text: str
    header_path: str
    sku_rows_json: dict[str, Any] | None
    rrf_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    """融合前直接从双路结果采集的父块证据，不借助排序结果重建粗召回池。"""

    parent_id: str
    doc_code: str
    content_text: str
    header_path: str


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    degraded_flag: bool
    degraded_stage: str | None
    trace_id: str
    collection_name: str
    raw_candidates: tuple[RetrievalCandidate, ...] = ()
    dense_parent_ids: tuple[str, ...] = ()
    sparse_parent_ids: tuple[str, ...] = ()


def _capture_candidates(
    points: Sequence[models.ScoredPoint],
) -> tuple[RetrievalCandidate, ...]:
    """保留原始父块首次出现顺序；正文损坏仍保留 ID，避免误报为未召回。"""
    candidates: dict[str, RetrievalCandidate] = {}
    for point in points:
        payload = point.payload or {}
        parent_id = payload.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            continue
        values = [payload.get(key) for key in ("doc_code", "parent_content", "header_path")]
        candidates.setdefault(
            parent_id,
            RetrievalCandidate(
                parent_id, *(value if isinstance(value, str) else "" for value in values)
            ),
        )
    return tuple(candidates.values())


def build_acl_filter(dept_scope: str | Sequence[str], role_level: int) -> models.Filter:
    """部门集合必须来自认证上下文；空权限拒绝查询，绝不省略过滤器。"""
    departments = [dept_scope] if isinstance(dept_scope, str) else list(dept_scope)
    if not departments or any(
        not isinstance(item, str) or not item.strip() for item in departments
    ):
        raise ValueError("检索必须指定非空授权部门")
    if type(role_level) is not int or role_level < 0:
        raise ValueError("角色等级必须是非负整数")
    return models.Filter(
        must=[
            models.FieldCondition(key="dept_scope", match=models.MatchAny(any=departments)),
            models.FieldCondition(key="min_role_level", range=models.Range(gte=0, lte=role_level)),
        ]
    )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[models.ScoredPoint]], *, k: int = 60
) -> list[RetrievalHit]:
    """每路先按父块首次出现压缩名次，再累加 1/(k+名次)，平分保持首见顺序。"""
    if k <= 0:
        raise ValueError("RRF 平滑常数必须大于零")
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for point in ranking:
            payload = point.payload or {}
            parent_id = payload.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id or parent_id in seen:
                continue
            # 损坏的证据不可作为模型上下文，尤其不能回填未受 ACL 约束的旁路数据。
            if not isinstance(payload.get("parent_content"), str):
                continue
            if not isinstance(payload.get("doc_code"), str):
                continue
            seen.add(parent_id)
            scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (k + len(seen))
            payloads.setdefault(parent_id, payload)
    hits = [
        RetrievalHit(
            parent_id=parent_id,
            doc_code=payloads[parent_id]["doc_code"],
            content_text=payloads[parent_id]["parent_content"],
            header_path=payloads[parent_id].get("header_path", ""),
            sku_rows_json=payloads[parent_id].get("parent_sku_rows_json"),
            rrf_score=score,
        )
        for parent_id, score in scores.items()
    ]
    return sorted(hits, key=lambda hit: -hit.rrf_score)


class HybridRetriever:
    """单次查询固定物理集合；审计事务成功后才返回结果。"""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        collection_name: str = "kb_children_prod",
        circuit_breaker: RedisCircuitBreaker | None = None,
    ) -> None:
        self.client = client
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.session_factory = session_factory
        self.collection_name = collection_name
        self.circuit_breaker = circuit_breaker

    async def _rerank(self, query: str, hits: Sequence[RetrievalHit]) -> list[RerankScore]:
        documents = [hit.content_text for hit in hits]
        if self.circuit_breaker is not None:
            return await self.circuit_breaker.call(self.reranker.rerank, query, documents)
        return await self.reranker.rerank(query, documents)

    async def search(
        self,
        query: str,
        *,
        dept_scope: str | Sequence[str],
        role_level: int,
        top_k: int = 5,
        recall_limit: int = 30,
        session_id: str | None = None,
        doc_type: DocumentType | None = None,
    ) -> RetrievalResult:
        if not query.strip():
            raise ValueError("检索问题不能为空")
        if not 1 <= top_k <= recall_limit <= 500:
            raise ValueError("召回数量必须满足 1 <= top_k <= recall_limit <= 500")
        if session_id is not None and len(session_id) > 64:
            raise ValueError("会话标识不能超过 64 字符")
        acl_filter = build_acl_filter(dept_scope, role_level)
        if doc_type is not None:
            # 类型限制只缩小召回范围，不替换部门与职级约束。
            acl_filter.must.append(
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type.value))
            )
        started = perf_counter()
        trace_id = get_trace_id() or uuid4().hex
        aliases = await self.client.get_aliases()
        collection = next(
            (
                alias.collection_name
                for alias in aliases.aliases
                if alias.alias_name == self.collection_name
            ),
            self.collection_name,
        )
        dense = (await self.embedding_provider.embed([query]))[0]
        sparse = encode_query(query)
        # 两路使用同一份权限条件与物理集合，别名切换不会造成跨版本拼接。
        async with asyncio.TaskGroup() as group:
            dense_task = group.create_task(
                self.client.query_points(
                    collection,
                    query=dense,
                    using="dense",
                    query_filter=acl_filter,
                    limit=recall_limit,
                    with_payload=True,
                )
            )
            sparse_task = group.create_task(
                self.client.query_points(
                    collection,
                    query=sparse,
                    using="bm25_sparse",
                    query_filter=acl_filter,
                    limit=recall_limit,
                    with_payload=True,
                )
            )
        dense_points, sparse_points = dense_task.result().points, sparse_task.result().points
        dense_candidates = _capture_candidates(dense_points)
        sparse_candidates = _capture_candidates(sparse_points)
        raw_candidates = _capture_candidates([*dense_points, *sparse_points])
        fused = reciprocal_rank_fusion([dense_points, sparse_points])
        ranked = fused
        degraded = False
        if fused:
            try:
                scores = await asyncio.wait_for(self._rerank(query, fused), timeout=0.6)
                validate_scores(scores, len(fused))
                ranked = [
                    replace(fused[item.index], rerank_score=item.score)
                    for item in sorted(scores, key=lambda item: (-item.score, item.index))
                ]
            except Exception as error:
                # CancelledError 不属于 Exception，调用方主动取消时继续向上传播。
                degraded = True
                logger.warning(
                    "精排失败，回退 RRF",
                    trace_id=trace_id,
                    degraded_stage="rerank_fallback",
                    error_type=type(error).__name__,
                )
        stage = "rerank_fallback" if degraded else None
        hits = ranked[:top_k]
        async with self.session_factory.begin() as session:
            session.add(
                RAGAuditLog(
                    trace_id=trace_id,
                    session_id=session_id,
                    query_text=query,
                    recalled_parent_ids=[hit.parent_id for hit in fused],
                    guardrail_action="ACL_FILTERED",
                    latency_ms=round((perf_counter() - started) * 1000),
                    degraded_flag=degraded,
                    degraded_stage=stage,
                )
            )
        return RetrievalResult(
            hits,
            degraded,
            stage,
            trace_id,
            collection,
            raw_candidates,
            tuple(hit.parent_id for hit in dense_candidates),
            tuple(hit.parent_id for hit in sparse_candidates),
        )
