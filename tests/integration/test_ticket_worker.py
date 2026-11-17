"""真实 Redis 与 MySQL 的工单幂等、发件箱恢复、派单及死信协同。"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from alembic.config import Config
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import delete, func, select

from aegis.config import Settings
from aegis.db.models import OpsTicketOutbox, OpsWorkOrder, WorkOrderPriority
from aegis.db.redis_client import create_redis_client
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.workers.consumer import TicketWorker
from aegis.workers.tickets import TicketService, publication_key
from alembic import command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AEGIS_INTEGRATION") != "1",
        reason="工单联调需真实 MySQL、Redis 并设置 AEGIS_INTEGRATION=1",
    ),
]


_ALEMBIC_CONFIG = str(Path(__file__).resolve().parents[2] / "alembic.ini")


@pytest.fixture
async def storage():
    config = Config(_ALEMBIC_CONFIG)
    await asyncio.to_thread(command.upgrade, config, "head")
    settings = Settings()
    engine = create_mysql_engine(settings)
    sessions = create_session_factory(engine)
    redis = create_redis_client(settings)
    suffix = uuid4().hex
    stream = f"stream:ticket:test:{suffix}"
    dlq = f"stream:ticket:dlq:test:{suffix}"
    service = TicketService(redis, sessions, stream_name=stream)
    worker = TicketWorker(redis, sessions, stream_name=stream, dlq_name=dlq, reclaim_idle_ms=0)
    values = {
        "idempotency_key": f"ticket-test-{suffix}",
        "session_id": suffix,
        "trace_id": suffix,
        "priority": WorkOrderPriority.P0_EMERGENCY,
        "trigger_source": "HIGH_RISK_PRECHECK",
        "slots_payload": {"现象": "冒烟", "设备编号": "设备-01"},
        "completeness_score": 2 / 3,
        "context_summary": "操作员报告异常发热",
    }
    try:
        yield service, worker, values
    finally:
        async with sessions.begin() as session:
            orders = list(
                await session.scalars(select(OpsWorkOrder).where(OpsWorkOrder.session_id == suffix))
            )
            ids = [order.id for order in orders]
            events = list(
                await session.scalars(
                    select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id.in_(ids))
                )
            )
            keys = [f"idem:ticket:{order.idempotency_key}" for order in orders]
            keys += [publication_key(event.event_id) for event in events]
            await session.execute(delete(OpsWorkOrder).where(OpsWorkOrder.session_id == suffix))
        keys += [stream, dlq, worker.failure_key]
        keys += [key async for key in redis.scan_iter(match=worker.lock_prefix + "*")]
        await redis.delete(*keys)
        await redis.aclose()
        await engine.dispose()


async def test_real_concurrent_idempotency_24h_cache_and_normal_dispatch(storage):
    service, worker, values = storage
    results = await asyncio.gather(*(service.create(**values) for _ in range(6)))
    assert len({ticket.id for ticket in results}) == 1
    assert len({ticket.ticket_no for ticket in results}) == 1
    assert await service.redis.xlen(service.stream_name) == 1
    cache_key = f"idem:ticket:{values['idempotency_key']}"
    assert 86380 <= await service.redis.ttl(cache_key) <= 86400
    async with service.session_factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(OpsWorkOrder)
            .where(OpsWorkOrder.idempotency_key == values["idempotency_key"])
        )
        assert count == 1
    # 模拟高速缓存到期，MySQL 唯一键与已发布发件箱仍禁止第二条流消息。
    await service.redis.delete(cache_key)
    assert (await service.create(**values)).id == results[0].id
    assert await service.redis.xlen(service.stream_name) == 1
    assert await worker.run_once() == 1
    assert (await service.redis.xpending(service.stream_name, "cg_ops_worker"))["pending"] == 0
    async with service.session_factory() as session:
        order = await session.get(OpsWorkOrder, results[0].id)
        assert order.status == "DISPATCHED"
    assert (await service.create(**values)).status == "DISPATCHED"
    assert await service.redis.xlen(service.stream_name) == 1


async def test_real_redis_publish_outage_is_recovered_from_mysql_outbox(storage):
    service, worker, values = storage
    with patch.object(service.redis, "eval", side_effect=RedisConnectionError("模拟断连")):
        ticket = await service.create(**values)
    async with service.session_factory() as session:
        event = await session.scalar(
            select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == ticket.id)
        )
        assert not event.published
        assert (await session.get(OpsWorkOrder, ticket.id)).status == "PENDING"
    assert await service.redis.xlen(service.stream_name) == 0
    assert await worker.run_once() == 1
    async with service.session_factory() as session:
        assert (await session.get(OpsWorkOrder, ticket.id)).status == "DISPATCHED"
    assert await service.redis.xlen(service.stream_name) == 1


async def test_real_published_event_survives_outbox_commit_failure_without_duplicate(storage):
    service, worker, values = storage
    original = service._publish

    async def publish_then_crash(row):
        await original(row)
        raise RuntimeError("模拟流已写入但数据库回写前退出")

    with patch.object(service, "_publish", side_effect=publish_then_crash):
        with pytest.raises(RuntimeError, match="数据库回写前退出"):
            await service.create(**values)
    assert await service.redis.xlen(service.stream_name) == 1
    ticket = await service.create(**values)
    assert await service.redis.xlen(service.stream_name) == 1
    async with service.session_factory() as session:
        event = await session.scalar(
            select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == ticket.id)
        )
        assert event.published
        assert event.redis_message_id
    assert await worker.run_once() == 1


async def test_real_three_failures_move_full_event_to_dlq_and_ack(storage):
    service, worker, values = storage
    ticket = await service.create(**values)
    worker.processor = AsyncMock(side_effect=RuntimeError("模拟派单下游连续故障"))
    for attempt in range(1, 4):
        assert await worker.run_once() == 1
        pending = await service.redis.xpending(service.stream_name, worker.group_name)
        assert pending["pending"] == (0 if attempt == 3 else 1)
        assert await service.redis.xlen(worker.dlq_name) == (1 if attempt == 3 else 0)
    rows = await service.redis.xrange(worker.dlq_name)
    payload = json.loads(rows[0][1]["payload"])
    assert payload["ticket_id"] == str(ticket.id)
    assert payload["priority"] == "P0_EMERGENCY"
    assert json.loads(payload["slots_payload"]) == values["slots_payload"]
    assert payload["context_summary"] == values["context_summary"]
    assert rows[0][1]["attempts"] == "3"
    assert "连续故障" in rows[0][1]["failure_reason"]
    assert await worker.run_once() == 0
    assert worker.processor.await_count == 3


async def test_real_stale_pending_reclaimed_and_ack_loss_does_not_repeat_business_update(storage):
    service, worker, values = storage
    ticket = await service.create(**values)
    await worker.ensure_group()
    messages = await service.redis.xreadgroup(
        worker.group_name, "lost-consumer", {service.stream_name: ">"}, count=1
    )
    message_id, payload = messages[0][1][0]
    await worker.dispatch(payload)
    async with service.session_factory() as session:
        first_timestamp = (await session.get(OpsWorkOrder, ticket.id)).updated_at
    assert await worker.run_once() == 1
    async with service.session_factory() as session:
        recovered = await session.get(OpsWorkOrder, ticket.id)
        assert recovered.status == "DISPATCHED"
        assert recovered.updated_at == first_timestamp
    assert (await service.redis.xpending(service.stream_name, worker.group_name))["pending"] == 0
    assert await worker.process_message(message_id, payload) is False


async def test_late_create_receipt_does_not_overwrite_dispatched_cache(storage):
    service, worker, values = storage
    publish = service.publish_ticket

    async def dispatch_before_creation_returns(ticket_id):
        await publish(ticket_id)
        assert await worker.run_once() == 1

    with patch.object(service, "publish_ticket", side_effect=dispatch_before_creation_returns):
        ticket = await service.create(**values)
    assert ticket.status == "PENDING"
    assert (await service.create(**values)).status == "DISPATCHED"
    assert await service.redis.xlen(service.stream_name) == 1
