"""离线验证工单高速幂等、消费者失败边界与取消语义。"""

import asyncio
import json
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from aegis.db.models import WorkOrderPriority
from aegis.workers.consumer import TicketWorker
from aegis.workers.tickets import IDEMPOTENCY_TTL, TicketResult, TicketService


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


def request_values():
    return {
        "idempotency_key": "request-1",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "priority": WorkOrderPriority.P0_EMERGENCY,
        "trigger_source": "HIGH_RISK_PRECHECK",
        "slots_payload": {"现象": "冒烟"},
        "completeness_score": 0.2,
        "context_summary": "等待人工处置",
    }


async def test_cached_ticket_returns_without_database_or_stream_call(redis):
    factory = AsyncMock()
    service = TicketService(redis, factory)
    values = request_values()
    cached = TicketResult(id=1, ticket_no="OPS-1", status="PENDING", **values)
    await redis.set("idem:ticket:request-1", cached.model_dump_json(), ex=IDEMPOTENCY_TTL)
    assert await service.create(**values) == cached
    factory.assert_not_called()
    assert await redis.xlen(service.stream_name) == 0
    assert 86390 <= await redis.ttl("idem:ticket:request-1") <= 86400


async def test_cached_ticket_cannot_be_read_by_another_session(redis):
    values = request_values()
    cached = TicketResult(id=1, ticket_no="OPS-1", status="PENDING", **values)
    await redis.set("idem:ticket:request-1", cached.model_dump_json(), ex=IDEMPOTENCY_TTL)
    with pytest.raises(ValueError, match="其他会话"):
        await TicketService(redis, AsyncMock()).create(**(values | {"session_id": "intruder"}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"idempotency_key": ""},
        {"session_id": "a" * 65},
        {"completeness_score": 1.1},
        {"completeness_score": float("nan")},
        {"slots_payload": {"非法数值": float("nan")}},
    ],
)
async def test_invalid_ticket_rejected_before_io(redis, overrides):
    factory = AsyncMock()
    with pytest.raises(ValueError):
        await TicketService(redis, factory).create(**(request_values() | overrides))
    factory.assert_not_called()
    assert await redis.dbsize() == 0


async def worker_with_event(redis, processor):
    worker = TicketWorker(redis, AsyncMock(), processor=processor)
    worker.publisher.publish_pending = AsyncMock(return_value=0)
    await worker.ensure_group()
    payload = {"ticket_id": "1", "ticket_no": "OPS-1", "trace_id": "trace-1", "detail": "完整载荷"}
    message_id = await redis.xadd(worker.stream_name, payload)
    return worker, message_id, payload


async def test_three_failures_move_complete_payload_to_dlq_and_ack(redis):
    processor = AsyncMock(side_effect=RuntimeError("派单系统不可用"))
    worker, message_id, payload = await worker_with_event(redis, processor)
    for attempt in (1, 2, 3):
        assert await worker.run_once() == 1
        pending = await redis.xpending(worker.stream_name, worker.group_name)
        assert pending["pending"] == (0 if attempt == 3 else 1)
        assert await redis.xlen(worker.dlq_name) == (1 if attempt == 3 else 0)
    assert processor.await_count == 3
    rows = await redis.xrange(worker.dlq_name)
    dlq = rows[0][1]
    assert dlq["attempts"] == "3"
    assert dlq["source_message_id"] == message_id
    assert json.loads(dlq["payload"]) == payload
    assert "派单系统不可用" in dlq["failure_reason"]
    assert await worker.process_message(message_id, payload) is False
    assert await redis.xlen(worker.dlq_name) == 1
    assert await redis.hlen(worker.failure_key) == 0


async def test_success_acks_and_clears_previous_failure_count(redis):
    processor = AsyncMock(side_effect=[RuntimeError("暂时断连"), None])
    worker, message_id, _ = await worker_with_event(redis, processor)
    await worker.run_once()
    assert await redis.hget(worker.failure_key, message_id) == "1"
    await worker.run_once()
    assert (await redis.xpending(worker.stream_name, worker.group_name))["pending"] == 0
    assert await redis.hget(worker.failure_key, message_id) is None
    assert await redis.xlen(worker.dlq_name) == 0


async def test_cancellation_leaves_pending_without_counting_failure(redis):
    started = asyncio.Event()

    async def wait_for_cancellation(payload):
        started.set()
        await asyncio.Event().wait()

    worker, message_id, _ = await worker_with_event(redis, wait_for_cancellation)
    task = asyncio.create_task(worker.run_once())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await redis.xpending(worker.stream_name, worker.group_name))["pending"] == 1
    assert await redis.hget(worker.failure_key, message_id) is None
    assert await redis.get(worker.lock_prefix + message_id) is None


async def test_concurrent_delivery_of_same_message_has_single_processor(redis):
    started, finish = asyncio.Event(), asyncio.Event()

    async def slow_processor(payload):
        started.set()
        await finish.wait()

    processor = AsyncMock(side_effect=slow_processor)
    worker, message_id, payload = await worker_with_event(redis, processor)
    await redis.xreadgroup(worker.group_name, worker.consumer_name, {worker.stream_name: ">"})
    running = asyncio.create_task(worker.process_message(message_id, payload))
    await started.wait()
    assert await worker.process_message(message_id, payload) is False
    finish.set()
    assert await running is True
    processor.assert_awaited_once()
    assert (await redis.xpending(worker.stream_name, worker.group_name))["pending"] == 0


async def test_reclaim_cursor_advances_past_nonexpired_pending_entries(redis):
    worker = TicketWorker(redis, AsyncMock())
    redis.xautoclaim = AsyncMock(
        side_effect=[
            ["20-0", [], []],
            ["0-0", [("21-0", {"ticket_id": "1"})], []],
        ]
    )
    assert await worker._read_reclaimed(1) == []
    assert await worker._read_reclaimed(1) == [("21-0", {"ticket_id": "1"})]
    calls = redis.xautoclaim.await_args_list
    assert calls[0].kwargs["start_id"] == "0-0"
    assert calls[1].kwargs["start_id"] == "20-0"


async def test_pending_failures_do_not_starve_new_messages(redis):
    async def process(payload):
        if payload["ticket_id"] == "1":
            raise RuntimeError("旧消息持续失败")

    processor = AsyncMock(side_effect=process)
    worker, _, _ = await worker_with_event(redis, processor)
    await worker.run_once(count=1)
    fresh_id = await redis.xadd(worker.stream_name, {"ticket_id": "2"})
    await worker.run_once(count=1)
    assert {call.args[0]["ticket_id"] for call in processor.await_args_list} == {"1", "2"}
    assert (
        await redis.xpending_range(worker.stream_name, worker.group_name, fresh_id, fresh_id, 1)
        == []
    )


@pytest.mark.parametrize("fail", [False, True])
async def test_expired_message_lock_cannot_ack_or_record_failure_for_new_owner(redis, fail):
    worker, message_id, payload = await worker_with_event(redis, AsyncMock())
    await redis.xreadgroup(worker.group_name, worker.consumer_name, {worker.stream_name: ">"})

    async def lose_lease(payload):
        await redis.set(worker.lock_prefix + message_id, "new-owner", ex=30)
        if fail:
            raise RuntimeError("迟到的旧持有者失败")

    worker.processor = lose_lease
    await worker.process_message(message_id, payload)
    assert (await redis.xpending(worker.stream_name, worker.group_name))["pending"] == 1
    assert await redis.hget(worker.failure_key, message_id) is None
    assert await redis.xlen(worker.dlq_name) == 0
    assert await redis.get(worker.lock_prefix + message_id) == "new-owner"


async def test_message_trace_context_is_propagated_and_restored(redis):
    from aegis.core.trace_context import get_trace_id, trace_id_var

    async def check_trace(payload):
        assert get_trace_id() == payload["trace_id"]

    token = trace_id_var.set("outside-message")
    try:
        worker, _, _ = await worker_with_event(redis, check_trace)
        await worker.run_once()
        assert get_trace_id() == "outside-message"
    finally:
        trace_id_var.reset(token)
