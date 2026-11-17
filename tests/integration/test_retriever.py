"""使用真实 Qdrant 内存引擎验证召回语义，额外启用 MySQL 审计往返。"""

import asyncio
import os
from contextlib import asynccontextmanager
from time import perf_counter
from unittest.mock import AsyncMock
from uuid import uuid4

import fakeredis.aioredis
import pytest
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import delete, select

from aegis.config import Settings
from aegis.core.circuit_breaker import RedisCircuitBreaker
from aegis.db.models import RAGAuditLog
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.rag.reranker import MockReranker, RerankScore
from aegis.rag.retriever import HybridRetriever, build_acl_filter, reciprocal_rank_fusion
from aegis.rag.sparse import encode_document


class FixedEmbedding:
    dimension = 3

    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class RecordingAudit:
    """保留真实 ORM 审计对象，只替代事务传输。"""

    def __init__(self):
        self.rows = []

    @asynccontextmanager
    async def begin(self):
        yield self

    def add(self, row):
        self.rows.append(row)


class IdentityReranker:
    async def rerank(self, query, documents):
        return [RerankScore(index, 1 / (index + 1)) for index in range(len(documents))]


def point(parent_id, *, dept="warehouse", role=1, text="扫码排障 SKU-A 额定功率 1200 W"):
    return models.PointStruct(
        id=str(uuid4()),
        vector={"dense": [1.0, 0.0, 0.0], "bm25_sparse": encode_document(text)},
        payload={
            "parent_id": parent_id,
            "doc_code": f"doc-{parent_id}",
            "dept_scope": dept,
            "min_role_level": role,
            "parent_content": text,
            "header_path": "仓储 > 扫码排障",
            "parent_sku_rows_json": {"rows": [{"SKU": "SKU-A", "功率": "1200 W"}]},
        },
    )


@pytest.fixture
async def collection():
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        "test_v1",
        vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            "bm25_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    await client.upsert(
        "test_v1",
        points=[
            point("allowed-a"),
            point("allowed-a"),
            point("allowed-b", text="SKU-B 包装重量 2.5 kg，扫码核对批次"),
            point("secret-dept", dept="finance"),
            point("secret-role", role=9),
            point("secret-both", dept="finance", role=9),
        ],
    )
    await client.update_collection_aliases(
        [
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name="test_v1", alias_name="kb_children_prod"
                )
            )
        ]
    )
    try:
        yield client
    finally:
        await client.close()


async def test_hybrid_recall_acl_rrf_parent_context_and_audit(collection):
    audit = RecordingAudit()
    retriever = HybridRetriever(collection, FixedEmbedding(), MockReranker(), audit)
    result = await retriever.search("SKU-A 功率 扫码", dept_scope="warehouse", role_level=1)
    assert {hit.parent_id for hit in result.hits} == {"allowed-a", "allowed-b"}
    assert len(result.hits) == 2
    assert all("secret" not in hit.parent_id for hit in result.hits)
    assert all(hit.sku_rows_json["rows"][0]["功率"] == "1200 W" for hit in result.hits)
    assert not result.degraded_flag
    assert result.collection_name == "test_v1"
    assert len(audit.rows) == 1
    assert audit.rows[0].guardrail_action == "ACL_FILTERED"
    assert set(audit.rows[0].recalled_parent_ids) == {"allowed-a", "allowed-b"}


async def test_permissions_can_include_multiple_departments_without_bypassing_role(collection):
    retriever = HybridRetriever(collection, FixedEmbedding(), MockReranker(), RecordingAudit())
    result = await retriever.search("SKU", dept_scope=["warehouse", "finance"], role_level=1)
    assert {hit.parent_id for hit in result.hits} == {"allowed-a", "allowed-b", "secret-dept"}


async def test_empty_authorized_recall_is_audited_without_rerank(collection):
    reranker = AsyncMock()
    audit = RecordingAudit()
    retriever = HybridRetriever(collection, FixedEmbedding(), reranker, audit)
    result = await retriever.search("SKU", dept_scope="unknown", role_level=1)
    assert result.hits == []
    assert not result.degraded_flag
    assert audit.rows[0].recalled_parent_ids == []
    reranker.rerank.assert_not_called()


@pytest.mark.parametrize(
    "dept,role", [([], 1), ("", 1), (["warehouse", ""], 1), ("warehouse", -1), ("warehouse", True)]
)
def test_invalid_acl_fails_closed(dept, role):
    with pytest.raises(ValueError):
        build_acl_filter(dept, role)


def test_rrf_deduplicates_each_route_before_scoring():
    def scored(parent, score):
        item = point(parent)
        return models.ScoredPoint(id=item.id, version=0, score=score, payload=item.payload)

    hits = reciprocal_rank_fusion(
        [
            [scored("a", 100), scored("a", 99), scored("b", 98)],
            [scored("b", 0.9), scored("c", 0.8), scored("a", 0.7)],
        ]
    )
    assert [hit.parent_id for hit in hits] == ["b", "a", "c"]
    assert hits[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[1].rrf_score == pytest.approx(1 / 61 + 1 / 63)
    assert hits[2].rrf_score == pytest.approx(1 / 62)


async def test_routes_are_concurrent_and_pin_the_same_physical_collection(collection, monkeypatch):
    original = collection.query_points
    started = []
    both_started = asyncio.Event()

    async def guarded_query(name, **kwargs):
        started.append((name, kwargs["using"], kwargs["query_filter"]))
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        return await original(name, **kwargs)

    monkeypatch.setattr(collection, "query_points", guarded_query)
    retriever = HybridRetriever(collection, FixedEmbedding(), MockReranker(), RecordingAudit())
    await retriever.search("扫码", dept_scope="warehouse", role_level=1)
    assert {item[0] for item in started} == {"test_v1"}
    assert {item[1] for item in started} == {"dense", "bm25_sparse"}
    assert started[0][2] == started[1][2] == build_acl_filter("warehouse", 1)


async def test_rerank_timeout_returns_original_rrf_and_persists_audit(collection):
    cancelled = asyncio.Event()

    class SlowReranker:
        async def rerank(self, query, documents):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.set()

    baseline = await HybridRetriever(
        collection, FixedEmbedding(), IdentityReranker(), RecordingAudit()
    ).search("扫码", dept_scope="warehouse", role_level=1)
    audit = RecordingAudit()
    retriever = HybridRetriever(collection, FixedEmbedding(), SlowReranker(), audit)
    started = perf_counter()
    result = await retriever.search("扫码", dept_scope="warehouse", role_level=1)
    elapsed = perf_counter() - started
    assert 0.58 <= elapsed < 0.95
    assert cancelled.is_set()
    assert [hit.parent_id for hit in result.hits] == [hit.parent_id for hit in baseline.hits]
    assert all(hit.rerank_score is None for hit in result.hits)
    assert result.degraded_flag and result.degraded_stage == "rerank_fallback"
    assert audit.rows[0].degraded_flag
    assert audit.rows[0].degraded_stage == "rerank_fallback"
    assert audit.rows[0].latency_ms >= 580


async def test_open_circuit_falls_back_without_calling_provider(collection):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        breaker = RedisCircuitBreaker(redis, "test-rerank", failure_threshold=1)

        async def fail():
            raise RuntimeError("模拟上游失败")

        with pytest.raises(RuntimeError):
            await breaker.call(fail)
        reranker = AsyncMock()
        audit = RecordingAudit()
        retriever = HybridRetriever(
            collection, FixedEmbedding(), reranker, audit, circuit_breaker=breaker
        )
        result = await retriever.search("扫码", dept_scope="warehouse", role_level=1)
        assert result.degraded_flag
        assert result.hits
        assert audit.rows[0].degraded_stage == "rerank_fallback"
        reranker.rerank.assert_not_called()
    finally:
        await redis.aclose()


async def test_corrupt_rerank_response_falls_back(collection):
    reranker = AsyncMock()
    reranker.rerank.return_value = [RerankScore(999, 0.8)]
    result = await HybridRetriever(collection, FixedEmbedding(), reranker, RecordingAudit()).search(
        "扫码", dept_scope="warehouse", role_level=1
    )
    assert result.degraded_flag
    assert {hit.parent_id for hit in result.hits} == {"allowed-a", "allowed-b"}


async def test_caller_cancellation_is_not_swallowed(collection):
    started = asyncio.Event()

    class WaitingReranker:
        async def rerank(self, query, documents):
            started.set()
            await asyncio.Event().wait()

    audit = RecordingAudit()
    retriever = HybridRetriever(collection, FixedEmbedding(), WaitingReranker(), audit)
    task = asyncio.create_task(retriever.search("扫码", dept_scope="warehouse", role_level=1))
    await asyncio.wait_for(started.wait(), timeout=0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert audit.rows == []


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("AEGIS_INTEGRATION") != "1", reason="需要真实 MySQL")
async def test_timeout_audit_roundtrip_in_mysql(collection):
    engine = create_mysql_engine(Settings())
    factory = create_session_factory(engine)
    session_id = uuid4().hex

    class SlowReranker:
        async def rerank(self, query, documents):
            await asyncio.sleep(1)

    try:
        result = await HybridRetriever(
            collection, FixedEmbedding(), SlowReranker(), factory
        ).search("扫码故障", dept_scope="warehouse", role_level=1, session_id=session_id)
        async with factory() as session:
            row = await session.scalar(
                select(RAGAuditLog).where(RAGAuditLog.session_id == session_id)
            )
            assert row is not None
            assert row.trace_id == result.trace_id
            assert row.degraded_flag
            assert row.degraded_stage == "rerank_fallback"
            assert set(row.recalled_parent_ids) == {"allowed-a", "allowed-b"}
    finally:
        async with factory.begin() as session:
            await session.execute(delete(RAGAuditLog).where(RAGAuditLog.session_id == session_id))
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("AEGIS_INTEGRATION") != "1", reason="需要真实 MySQL 和 Qdrant")
async def test_real_qdrant_hybrid_acl_timeout_and_mysql_audit():
    from aegis.db.models import DocumentType
    from aegis.db.qdrant_client import create_qdrant_client
    from aegis.rag.indexer import QdrantIndexer
    from aegis.rag.parser import DocumentMetadata, MarkdownParser

    settings = Settings()
    client = create_qdrant_client(settings)
    engine = create_mysql_engine(settings)
    factory = create_session_factory(engine)
    suffix = uuid4().hex
    collection_name = f"test_retrieval_{suffix}"
    alias_name = f"test_retrieval_alias_{suffix}"
    indexer = QdrantIndexer(client, FixedEmbedding(), collection_name, alias_name)
    allowed = None

    class SlowReranker:
        async def rerank(self, query, documents):
            await asyncio.sleep(1)

    try:
        await indexer.ensure_collection()
        for dept, role in [("warehouse", 1), ("finance", 1), ("warehouse", 9)]:
            document = MarkdownParser().parse(
                "# SKU 参数\n\n| SKU | 功率 | 重量 |\n| --- | --- | --- |\n"
                "| SKU-A | 1200 W | 2.5 kg |\n",
                DocumentMetadata(
                    doc_code=f"{suffix}-{dept}-{role}",
                    doc_type=DocumentType.SKU_SPEC,
                    dept_scope=dept,
                    min_role_level=role,
                ),
            )
            await indexer.index_document(document)
            if dept == "warehouse" and role == 1:
                allowed = document
        await indexer.switch_alias()
        result = await HybridRetriever(
            client,
            FixedEmbedding(),
            SlowReranker(),
            factory,
            collection_name=alias_name,
        ).search("SKU-A 功率", dept_scope="warehouse", role_level=1, session_id=suffix)
        assert result.degraded_flag
        assert result.collection_name == collection_name
        assert {hit.parent_id for hit in result.hits} == {p.parent_id for p in allowed.parents}
        assert all(hit.doc_code == allowed.metadata.doc_code for hit in result.hits)
        assert any(hit.sku_rows_json for hit in result.hits)
        async with factory() as session:
            row = await session.scalar(select(RAGAuditLog).where(RAGAuditLog.session_id == suffix))
            assert row.degraded_stage == "rerank_fallback"
            assert row.degraded_flag
            assert row.recalled_parent_ids == [hit.parent_id for hit in result.hits]
        info = await client.get_collection(collection_name)
        assert {"dept_scope", "min_role_level", "parent_id", "doc_code"} <= set(info.payload_schema)
    finally:
        async with factory.begin() as session:
            await session.execute(delete(RAGAuditLog).where(RAGAuditLog.session_id == suffix))
        aliases = await client.get_aliases()
        if any(alias.alias_name == alias_name for alias in aliases.aliases):
            await client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias_name)
                    )
                ]
            )
        if await client.collection_exists(collection_name):
            await client.delete_collection(collection_name)
        await client.close()
        await engine.dispose()


async def test_raw_candidates_are_dense_sparse_union_before_ranking_cutoff(collection):
    retriever = HybridRetriever(collection, FixedEmbedding(), MockReranker(), RecordingAudit())
    result = await retriever.search("SKU-A 功率", dept_scope="warehouse", role_level=1, top_k=1)
    raw_ids = {hit.parent_id for hit in result.raw_candidates}
    assert raw_ids == set(result.dense_parent_ids) | set(result.sparse_parent_ids)
    assert raw_ids == {"allowed-a", "allowed-b"}
    assert len(result.hits) == 1
    assert {hit.parent_id for hit in result.hits} < raw_ids
    assert all(
        hit.doc_code and hit.content_text and hit.header_path for hit in result.raw_candidates
    )


async def test_raw_trace_keeps_recalled_parent_even_when_invalid_payload_cannot_be_fused(
    collection,
):
    corrupt = point("corrupt-target")
    corrupt.payload["parent_content"] = None
    await collection.upsert("test_v1", points=[corrupt])
    result = await HybridRetriever(
        collection, FixedEmbedding(), MockReranker(), RecordingAudit()
    ).search("SKU-A 功率", dept_scope="warehouse", role_level=1)
    assert "corrupt-target" in {item.parent_id for item in result.raw_candidates}
    assert "corrupt-target" not in {item.parent_id for item in result.hits}
    item = next(item for item in result.raw_candidates if item.parent_id == "corrupt-target")
    assert item.doc_code == "doc-corrupt-target" and item.content_text == ""
