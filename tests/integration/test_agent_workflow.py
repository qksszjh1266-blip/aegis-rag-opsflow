"""真实 Redis、MySQL、Qdrant 上的受控工作流与护栏端到端验收。"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from alembic.config import Config
from qdrant_client import models
from sqlalchemy import delete, select

from aegis.agent.state import SessionState
from aegis.agent.workflow import ControlledWorkflow, Node
from aegis.config import Settings
from aegis.db.models import (
    BadcaseCategory,
    KBDocument,
    KBParentChunk,
    OpsTicketOutbox,
    OpsWorkOrder,
    RAGAuditLog,
    WorkOrderPriority,
)
from aegis.db.qdrant_client import create_qdrant_client
from aegis.db.redis_client import create_redis_client
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.llm.generation import MockGenerationClient
from aegis.llm.providers import MockEmbeddingProvider
from aegis.rag.parser import DocumentMetadata, MarkdownParser
from aegis.rag.reranker import MockReranker
from aegis.rag.retriever import HybridRetriever
from aegis.rag.syncer import publish_snapshot
from aegis.schemas.agent import AccessContext, AgentRequest
from aegis.workers.tickets import TicketService, publication_key
from alembic import command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("AEGIS_INTEGRATION") != "1", reason="需要真实三类基础服务"),
]


_ALEMBIC_PATH = str(Path(__file__).resolve().parents[2] / "alembic.ini")


@pytest.fixture
async def environment():
    config = Config(_ALEMBIC_PATH)
    await asyncio.to_thread(command.upgrade, config, "head")
    settings = Settings()
    engine = create_mysql_engine(settings)
    sessions = create_session_factory(engine)
    redis = create_redis_client(settings)
    qdrant = create_qdrant_client(settings)
    identifier = uuid4().hex
    alias = "workflow-" + identifier
    stream = "test:workflow:events:" + identifier
    embeddings = MockEmbeddingProvider(32)
    doc = MarkdownParser().parse(
        "# 规格\n| SKU | 净重 | 容量 | 额定功率 | 温度上限 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| SKU-A | 0.5 kg | 1.5 L | 1.2 kW | 95 ℃ |\n"
        "| SKU-B | 0.8 kg | 2 L | 1.8 kW | 80 ℃ |\n",
        DocumentMetadata(
            doc_code="workflow-" + identifier,
            doc_type="SKU_SPEC",
            dept_scope="warehouse",
            min_role_level=1,
        ),
    )
    published = await publish_snapshot(
        [doc],
        engine=engine,
        client=qdrant,
        embedding_provider=embeddings,
        embedding_fingerprint="workflow-mock-v1",
        alias_name=alias,
    )
    tickets = TicketService(redis, sessions, stream_name=stream)
    retriever = HybridRetriever(qdrant, embeddings, MockReranker(), sessions, collection_name=alias)
    access = AccessContext(subject_id=identifier, dept_scope=["warehouse"], role_level=1)
    try:
        yield {
            "session_id": identifier,
            "redis": redis,
            "sessions": sessions,
            "tickets": tickets,
            "retriever": retriever,
            "access": access,
            "stream": stream,
        }
    finally:
        async with sessions.begin() as session:
            orders = list(
                await session.scalars(
                    select(OpsWorkOrder).where(OpsWorkOrder.session_id == identifier)
                )
            )
            for order in orders:
                outbox = await session.scalar(
                    select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == order.id)
                )
                if outbox:
                    await redis.delete(publication_key(outbox.event_id))
                    await session.delete(outbox)
                    await session.flush()
                await redis.delete(f"idem:ticket:{order.idempotency_key}")
                await session.delete(order)
            await session.execute(delete(RAGAuditLog).where(RAGAuditLog.session_id == identifier))
            stored = await session.scalar(
                select(KBDocument).where(KBDocument.doc_code == doc.metadata.doc_code)
            )
            await session.execute(delete(KBParentChunk).where(KBParentChunk.doc_id == stored.id))
            await session.delete(stored)
        await redis.delete(f"fsm:session:{identifier}", f"lock:session:{identifier}", stream)
        await qdrant.update_collection_aliases(
            [models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))]
        )
        await qdrant.delete_collection(published.collection_name)
        await qdrant.close()
        await redis.aclose()
        await engine.dispose()


def workflow(env, llm=None, cls=ControlledWorkflow):
    return cls(
        env["redis"],
        env["sessions"],
        env["retriever"],
        llm or MockGenerationClient(),
        env["tickets"],
    )


def request(env, message, *, number=1, scenario="qa"):
    return AgentRequest(
        session_id=env["session_id"],
        request_id=f"request-{number}",
        message=message,
        scenario=scenario,
    )


async def test_high_risk_zero_model_and_idempotent_ticket_real_storage(environment):
    env = environment
    llm = MockGenerationClient()
    agent = workflow(env, llm)
    agent.retriever = AsyncMock()
    body = request(env, "设备漏电并冒烟，需要立即处理")
    first = await agent.run(body, env["access"])
    second = await agent.run(body, env["access"])
    assert first.ticket_no == second.ticket_no
    assert first.guardrail_action == "PRECHECK_SHORT_CIRCUIT"
    assert llm.calls == []
    agent.retriever.search.assert_not_called()
    assert await env["redis"].xlen(env["stream"]) == 1
    assert await env["redis"].get(f"fsm:session:{env['session_id']}") is None
    async with env["sessions"]() as session:
        orders = list(
            await session.scalars(
                select(OpsWorkOrder).where(OpsWorkOrder.session_id == env["session_id"])
            )
        )
        assert len(orders) == 1
        assert orders[0].priority == WorkOrderPriority.P0_EMERGENCY
        assert orders[0].trigger_source == "HIGH_RISK_PRECHECK"
        assert 86390 <= await env["redis"].ttl(f"idem:ticket:{orders[0].idempotency_key}") <= 86400
        logs = list(
            await session.scalars(
                select(RAGAuditLog).where(RAGAuditLog.session_id == env["session_id"])
            )
        )
        assert all(row.guardrail_action == "PRECHECK_SHORT_CIRCUIT" for row in logs)


@pytest.mark.parametrize(
    "answers,expected,ticket",
    [
        (["SKU-A 净重500g，容量1500ml，额定功率1200W。"], "SKU_VERIFIED_PASS", False),
        (["SKU-A 净重800g。", "SKU-A 净重500g。"], "SKU_REGENERATED_PASS", False),
        (["SKU-A 净重800g。", "SKU-A 净重900g。"], "SKU_HALLUCINATION_BLOCKED", True),
    ],
)
async def test_sql_truth_units_regeneration_and_safe_block(environment, answers, expected, ticket):
    env = environment
    llm = MockGenerationClient(responses=answers)
    result = await workflow(env, llm).run(request(env, "SKU-A 的净重是多少？"), env["access"])
    assert result.guardrail_action == expected
    assert bool(result.ticket_no) == ticket
    assert len(llm.calls) == len(answers)
    assert result.steps <= 8
    async with env["sessions"]() as session:
        audit = await session.scalar(
            select(RAGAuditLog).where(
                RAGAuditLog.session_id == env["session_id"],
                RAGAuditLog.guardrail_action == expected,
            )
        )
        assert audit is not None
        if ticket:
            assert "800" not in result.answer and "900" not in result.answer
            assert audit.badcase_category == BadcaseCategory.HALLUCINATION
            order = await session.scalar(
                select(OpsWorkOrder).where(OpsWorkOrder.ticket_no == result.ticket_no)
            )
            assert order.priority == WorkOrderPriority.P1_HIGH


async def test_two_round_slot_limit_and_completeness_persist(environment):
    env = environment
    agent = workflow(env)
    first = await agent.run(request(env, "帮我排障", scenario="troubleshooting"), env["access"])
    assert first.asked_slots == ["device_id", "order_or_wave"]
    second = await agent.run(request(env, "设备编号：D-1", number=2), env["access"])
    assert second.asked_slots == ["order_or_wave", "issue"]
    third = await agent.run(request(env, "还找不到其他信息", number=3), env["access"])
    assert third.guardrail_action == "SLOT_LIMIT_HANDOFF"
    assert third.completeness_score == pytest.approx(1 / 3)
    async with env["sessions"]() as session:
        order = await session.scalar(
            select(OpsWorkOrder).where(OpsWorkOrder.ticket_no == third.ticket_no)
        )
        assert order.slots_payload["device_id"] == "D-1"
        assert order.context_summary == ""
        assert order.completeness_score == pytest.approx(1 / 3)


async def test_tenth_turn_compresses_and_retry_does_not_advance(environment):
    env = environment
    llm = MockGenerationClient()
    agent = workflow(env, llm)
    for number in range(1, 11):
        result = await agent.run(request(env, "SKU-A 净重", number=number), env["access"])
        assert result.ticket_no is None
    raw = await env["redis"].get(f"fsm:session:{env['session_id']}")
    state = SessionState.model_validate_json(raw)
    assert state.turn_count == state.summarized_turns == 10
    assert state.context_summary and len(state.history) == 4
    assert sum(call["purpose"] == "summary" for call in llm.calls) == 1
    before = len(llm.calls)
    await agent.run(request(env, "SKU-A 净重", number=10), env["access"])
    assert len(llm.calls) == before
    assert await env["redis"].get(f"fsm:session:{env['session_id']}") == raw


async def test_eight_step_limit_stops_loop_and_creates_real_ticket(environment):
    class LoopingWorkflow(ControlledWorkflow):
        async def _execute_node(self, node, turn):
            await super()._execute_node(Node.LOAD, turn)
            return Node.LOAD

    env = environment
    result = await workflow(env, cls=LoopingWorkflow).run(request(env, "测试循环"), env["access"])
    assert result.steps == 8
    assert result.guardrail_action == "STEP_LIMIT_HANDOFF"
    assert result.ticket_no
    assert await env["redis"].get(f"lock:session:{env['session_id']}") is None
