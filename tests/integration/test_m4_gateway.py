"""真实容器上的双协议共享服务 REST 验收；测试数据与线上别名隔离。"""

import asyncio
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import uvicorn
from alembic.config import Config
from mcp import ClientSession
from mcp.client.sse import sse_client
from qdrant_client import models
from sqlalchemy import delete, select

from aegis.agent.state import SessionState
from aegis.config import Settings
from aegis.db.models import (
    DocumentType,
    KBDocument,
    KBParentChunk,
    OpsTicketOutbox,
    OpsWorkOrder,
    RAGAuditLog,
    WorkOrderPriority,
)
from aegis.main import create_app
from aegis.rag.parser import DocumentMetadata, MarkdownParser
from aegis.rag.syncer import publish_snapshot
from aegis.schemas.agent import AccessContext
from aegis.workers.tickets import TICKET_STREAM, publication_key
from alembic import command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("AEGIS_INTEGRATION") != "1", reason="需要真实三类基础服务"),
]
_ROOT = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def serve_app(app):
    """在随机本地端口启动真实协议传输，资源生命周期由容器夹具管理。"""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


@pytest.fixture
async def gateway():
    """使用四份真实种子原文创建独立快照，所有读写仍经过真实数据库与 Redis。"""
    await asyncio.to_thread(command.upgrade, Config(str(_ROOT / "alembic.ini")), "head")
    identity = uuid4().hex
    tokens = {
        "warehouse": AccessContext(subject_id=identity, dept_scope=["warehouse"], role_level=1),
        "customer": AccessContext(
            subject_id=identity + "-customer", dept_scope=["customer_service"], role_level=1
        ),
        "finance-low": AccessContext(
            subject_id=identity + "-finance-low", dept_scope=["FINANCE"], role_level=1
        ),
        "warehouse-high": AccessContext(
            subject_id=identity + "-warehouse-high", dept_scope=["warehouse"], role_level=5
        ),
        "finance": AccessContext(
            subject_id=identity + "-finance", dept_scope=["FINANCE"], role_level=3
        ),
    }
    settings = Settings(
        provider_mode="mock",
        llm_provider_mode="mock",
        embedding_provider_mode="mock",
        rerank_provider_mode="mock",
        embedding_dimension=32,
        gateway_api_tokens=tokens,
    )
    app = create_app(settings)
    alias = "m4-gateway-" + identity
    documents = []
    for index, (filename, kind, dept, level) in enumerate(
        [
            ("ec_sku_specs.md", DocumentType.SKU_SPEC, "warehouse", 1),
            ("ec_warehouse_sop.md", DocumentType.WAREHOUSE_SOP, "warehouse", 1),
            ("ec_after_sales_rules.md", DocumentType.AFTER_SALES, "customer_service", 1),
            ("ec_finance_internal.md", DocumentType.FINANCE_RULE, "FINANCE", 3),
        ]
    ):
        documents.append(
            MarkdownParser().parse(
                (_ROOT / "data" / "seed_docs" / filename).read_text(encoding="utf-8"),
                DocumentMetadata(
                    doc_code=f"m4-{identity}-{index}",
                    doc_type=kind,
                    dept_scope=dept,
                    min_role_level=level,
                ),
            )
        )
    async with app.router.lifespan_context(app):
        published = await publish_snapshot(
            documents,
            engine=app.state.mysql_engine,
            client=app.state.qdrant,
            embedding_provider=app.state.workflow.retriever.embedding_provider,
            embedding_fingerprint="m4-gateway-mock-v1",
            alias_name=alias,
        )
        app.state.workflow.retriever.collection_name = alias
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
                headers={"Authorization": "Bearer warehouse", "X-Trace-ID": identity},
            ) as client:
                yield {
                    "app": app,
                    "client": client,
                    "identity": identity,
                    "documents": documents,
                }
        finally:
            async with app.state.session_factory.begin() as session:
                orders = list(
                    await session.scalars(
                        select(OpsWorkOrder).where(OpsWorkOrder.trace_id == identity)
                    )
                )
                for order in orders:
                    outbox = await session.scalar(
                        select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == order.id)
                    )
                    if outbox:
                        if outbox.redis_message_id:
                            await app.state.redis.xdel(TICKET_STREAM, outbox.redis_message_id)
                        await app.state.redis.delete(publication_key(outbox.event_id))
                        await session.delete(outbox)
                        await session.flush()
                    await app.state.redis.delete(f"idem:ticket:{order.idempotency_key}")
                    await session.delete(order)
                await session.execute(delete(RAGAuditLog).where(RAGAuditLog.trace_id == identity))
                codes = [doc.metadata.doc_code for doc in documents]
                ids = select(KBDocument.id).where(KBDocument.doc_code.in_(codes))
                await session.execute(delete(KBParentChunk).where(KBParentChunk.doc_id.in_(ids)))
                await session.execute(delete(KBDocument).where(KBDocument.doc_code.in_(codes)))
            await app.state.redis.delete(f"fsm:session:{identity}", f"lock:session:{identity}")
            await app.state.qdrant.update_collection_aliases(
                [models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))]
            )
            await app.state.qdrant.delete_collection(published.collection_name)


async def test_finance_acl_blocks_low_role_and_cross_department_without_leakage(gateway):
    client = gateway["client"]
    query = {"query": "财务供应商季度返利阶梯返点比例", "top_k": 10, "doc_type": "FINANCE_RULE"}
    for caller in ("customer", "finance-low", "warehouse-high"):
        result = await client.post(
            "/api/v1/kb/search", json=query, headers={"Authorization": f"Bearer {caller}"}
        )
        assert result.status_code == 200, result.text
        assert result.json()["hits"] == []
        assert result.json()["trace_id"] == gateway["identity"]
        assert "6.8%" not in result.text and "4.5%" not in result.text
    privileged = await client.post(
        "/api/v1/kb/search", json=query, headers={"Authorization": "Bearer finance"}
    )
    assert privileged.status_code == 200, privileged.text
    assert privileged.json()["hits"]
    assert "6.8%" in privileged.text
    assert all(
        hit["doc_code"] == gateway["documents"][3].metadata.doc_code
        for hit in privileged.json()["hits"]
    )
    # 认证上下文来自服务端映射，外部调用方不能用权限请求头或正文提权。
    anonymous = await client.post(
        "/api/v1/kb/search",
        json=query,
        headers={"Authorization": "", "X-Dept-Scope": "FINANCE", "X-Role-Level": "5"},
    )
    assert anonymous.status_code == 401
    forged = await client.post(
        "/api/v1/kb/search", json={**query, "dept_scope": ["FINANCE"], "role_level": 5}
    )
    assert forged.status_code == 422


async def test_sku_search_normalized_verification_and_parent_acl(gateway):
    client = gateway["client"]
    response = await client.post(
        "/api/v1/kb/search",
        json={"query": "EC-ih200-Std 净重", "top_k": 10, "doc_type": "SKU_SPEC"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["latency_ms"] >= 0 and result["rerank_fallback"] is False
    hit = next(hit for hit in result["hits"] if "| EC-ih200-Std |" in hit["content_text"])
    assert hit["header_path"] and hit["parent_id"]
    body = {
        "candidate_text": "EC-ih200-Std 净重1800g，容量1500ml，额定功率1.8kW。",
        "sku": "EC-ih200-Std",
        "parent_id": hit["parent_id"],
    }
    passed = await client.post("/api/v1/guardrails/verify-sku", json=body)
    assert passed.status_code == 200, passed.text
    assert passed.json()["passed"] is True
    assert passed.json()["checked_count"] == 3
    assert passed.json()["claims"] and passed.json()["reference_values"]
    failed = await client.post(
        "/api/v1/guardrails/verify-sku",
        json={**body, "candidate_text": "EC-ih200-Std 净重2050g。"},
    )
    assert failed.status_code == 200, failed.text
    assert failed.json()["passed"] is False
    assert "1.8 kg" in failed.json()["corrections"]
    forbidden = await client.post(
        "/api/v1/guardrails/verify-sku",
        json=body,
        headers={"Authorization": "Bearer customer"},
    )
    assert forbidden.status_code in (403, 404)
    assert "1.8 kg" not in forbidden.text


async def test_concurrent_ticket_idempotency_stream_and_subject_isolation(gateway):
    client, app = gateway["client"], gateway["app"]
    body = {
        "idempotency_key": "double-click-" + gateway["identity"],
        "session_id": gateway["identity"],
        "message": "饮水机冒烟漏电",
        "slots_payload": {"device_id": "EC-ih200-Std"},
    }
    replies = await asyncio.gather(*(client.post("/api/v1/tickets", json=body) for _ in range(3)))
    assert all(reply.status_code in (200, 201) for reply in replies), [r.text for r in replies]
    numbers = {reply.json()["ticket_no"] for reply in replies}
    assert len(numbers) == 1
    ticket = replies[0].json()
    assert ticket["priority"] == "P0_EMERGENCY"
    assert 86390 <= await app.state.redis.ttl(f"idem:ticket:{ticket['idempotency_key']}") <= 86400
    async with app.state.session_factory() as session:
        orders = list(
            await session.scalars(select(OpsWorkOrder).where(OpsWorkOrder.ticket_no.in_(numbers)))
        )
        assert len(orders) == 1
        outbox = await session.scalar(
            select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == orders[0].id)
        )
        assert outbox.published and outbox.stream_name == TICKET_STREAM
        events = await app.state.redis.xrange(
            TICKET_STREAM, outbox.redis_message_id, outbox.redis_message_id
        )
        assert len(events) == 1 and events[0][1]["ticket_no"] == ticket["ticket_no"]
    other = await client.post(
        "/api/v1/tickets", json=body, headers={"Authorization": "Bearer customer"}
    )
    assert other.status_code in (200, 201), other.text
    assert other.json()["ticket_no"] != ticket["ticket_no"]


async def test_chat_sku_fsm_replay_and_audit_real_storage(gateway):
    client, app = gateway["client"], gateway["app"]
    body = {
        "session_id": gateway["identity"],
        "request_id": "qa-1",
        "message": "EC-ih200-Std 的净重是多少？",
    }
    first = await client.post("/api/v1/agent/chat", json=body)
    assert first.status_code == 200, first.text
    result = first.json()
    assert result["guardrail_action"] == "SKU_VERIFIED_PASS"
    assert "1.8 kg" in result["answer"] and "2.05" not in result["answer"]
    assert result["steps"] <= 8 and result["input_tokens"] > 0
    replay = await client.post("/api/v1/agent/chat", json=body)
    assert replay.json() == result
    state = SessionState.model_validate_json(
        await app.state.redis.get(f"fsm:session:{gateway['identity']}")
    )
    assert state.turn_count == 1
    async with app.state.session_factory() as session:
        audit = await session.scalar(
            select(RAGAuditLog).where(
                RAGAuditLog.trace_id == gateway["identity"],
                RAGAuditLog.guardrail_action == "SKU_VERIFIED_PASS",
            )
        )
        assert audit is not None and audit.recalled_parent_ids


async def test_high_risk_chat_short_circuits_before_models_and_fsm(gateway):
    client, app = gateway["client"], gateway["app"]
    app.state.workflow.retriever = AsyncMock()
    app.state.workflow.llm = AsyncMock()
    result = await client.post(
        "/api/v1/agent/chat",
        json={
            "session_id": gateway["identity"],
            "request_id": "emergency-1",
            "message": "饮水机冒烟漏电",
        },
    )
    assert result.status_code == 200, result.text
    response = result.json()
    assert response["guardrail_action"] == "PRECHECK_SHORT_CIRCUIT"
    assert response["steps"] == 1
    assert response["input_tokens"] == response["output_tokens"] == 0
    app.state.workflow.retriever.search.assert_not_called()
    app.state.workflow.llm.generate.assert_not_called()
    assert await app.state.redis.get(f"fsm:session:{gateway['identity']}") is None
    async with app.state.session_factory() as session:
        order = await session.scalar(
            select(OpsWorkOrder).where(OpsWorkOrder.ticket_no == response["ticket_no"])
        )
        assert order.priority == WorkOrderPriority.P0_EMERGENCY


async def test_mcp_real_storage_tools_share_acl_verifier_and_ticket_service(gateway):
    app, identity = gateway["app"], gateway["identity"]
    async with serve_app(app) as base:
        async with sse_client(
            base + "/mcp/sse",
            headers={"Authorization": "Bearer warehouse", "X-Trace-ID": identity},
        ) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                registered = (await session.list_tools()).tools
                assert {tool.name for tool in registered} == {
                    "search_enterprise_kb",
                    "verify_sku_specs",
                    "create_ops_ticket",
                }
                denied = await session.call_tool(
                    "search_enterprise_kb",
                    {"body": {"query": "供应商季度返利比例", "doc_type": "FINANCE_RULE"}},
                )
                assert not denied.isError, denied
                assert denied.structuredContent["hits"] == []
                searched = await session.call_tool(
                    "search_enterprise_kb",
                    {"body": {"query": "EC-ih200-Std 净重", "top_k": 10, "doc_type": "SKU_SPEC"}},
                )
                assert not searched.isError, searched
                parent = next(
                    hit
                    for hit in searched.structuredContent["hits"]
                    if "| EC-ih200-Std |" in hit["content_text"]
                )
                verified = await session.call_tool(
                    "verify_sku_specs",
                    {
                        "body": {
                            "candidate_text": "EC-ih200-Std 净重1800g。",
                            "sku": "EC-ih200-Std",
                            "parent_id": parent["parent_id"],
                        }
                    },
                )
                assert not verified.isError, verified
                assert verified.structuredContent["passed"] is True
                body = {
                    "session_id": identity,
                    "idempotency_key": "mcp-" + identity,
                    "message": "饮水机冒烟漏电",
                }
                created = await session.call_tool("create_ops_ticket", {"body": body})
                repeated = await session.call_tool("create_ops_ticket", {"body": body})
                assert not created.isError and not repeated.isError
                assert created.structuredContent["priority"] == "P0_EMERGENCY"
                assert (
                    created.structuredContent["ticket_no"]
                    == repeated.structuredContent["ticket_no"]
                )
                ticket_no = created.structuredContent["ticket_no"]
        async with app.state.session_factory() as sql:
            ticket = await sql.scalar(
                select(OpsWorkOrder).where(OpsWorkOrder.ticket_no == ticket_no)
            )
            assert ticket is not None
            outbox = await sql.scalar(
                select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == ticket.id)
            )
            assert outbox.published and outbox.stream_name == TICKET_STREAM
