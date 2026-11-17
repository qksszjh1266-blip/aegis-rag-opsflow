"""Redis Streams 工单派单、待确认消息恢复与三次失败死信转移。"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from uuid import uuid4

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aegis.core.trace_context import trace_id_var
from aegis.db.models import OpsWorkOrder
from aegis.workers.tickets import (
    IDEMPOTENCY_TTL,
    TICKET_DLQ,
    TICKET_GROUP,
    TICKET_STREAM,
    TicketResult,
    TicketService,
)

logger = structlog.get_logger(__name__)

_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""
_ACK_SCRIPT = """
if redis.call('GET', KEYS[3]) ~= ARGV[3] then return -1 end
local result = redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[2], ARGV[2])
return result
"""
_FAILURE_SCRIPT = """
if redis.call('GET', KEYS[4]) ~= ARGV[5] then return -1 end
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1)
if #pending == 0 then return 0 end
local attempts = redis.call('HINCRBY', KEYS[3], ARGV[2], 1)
if attempts >= 3 then
    redis.call('XADD', KEYS[2], '*', 'source_stream', KEYS[1], 'source_message_id', ARGV[2],
        'payload', ARGV[3], 'failure_reason', ARGV[4], 'attempts', tostring(attempts))
    redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
    redis.call('HDEL', KEYS[3], ARGV[2])
end
return attempts
"""


def _decode(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


class TicketWorker:
    """默认派单只修改 MySQL 状态；自定义派单处理器必须自行实现业务幂等。"""

    def __init__(
        self,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        consumer_name: str | None = None,
        stream_name: str = TICKET_STREAM,
        dlq_name: str = TICKET_DLQ,
        group_name: str = TICKET_GROUP,
        reclaim_idle_ms: int = 30000,
        processing_timeout: float = 10.0,
        processor: Callable[[dict[str, str]], Awaitable[None]] | None = None,
    ) -> None:
        if reclaim_idle_ms < 0 or not 0 < processing_timeout <= 60:
            raise ValueError("回收空闲时间必须非负，处理超时必须在 0 至 60 秒内")
        self.redis = redis
        self.session_factory = session_factory
        self.consumer_name = consumer_name or f"ops-{uuid4().hex}"
        self.stream_name = stream_name
        self.dlq_name = dlq_name
        self.group_name = group_name
        self.reclaim_idle_ms = reclaim_idle_ms
        self.processing_timeout = processing_timeout
        self.processor = processor or self.dispatch
        identity = hashlib.sha256(f"{stream_name}:{group_name}".encode()).hexdigest()[:32]
        self.failure_key = f"retry:ticket:{identity}"
        self.lock_prefix = f"lock:ticket-message:{identity}:"
        self.publisher = TicketService(redis, session_factory, stream_name=stream_name)
        self._claim_cursor = "0-0"
        self._pending_cursor = "0-0"
        self._source_offset = 0

    async def ensure_group(self) -> None:
        """从历史消息起点创建消费者组，先于 worker 创建的事件也可被消费。"""
        try:
            await self.redis.xgroup_create(
                self.stream_name, self.group_name, id="0-0", mkstream=True
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def dispatch(self, payload: dict[str, str]) -> None:
        """事务内检查事件身份并更新状态；ACK 丢失后的重投不会重复执行派单。"""
        ticket_id = int(payload["ticket_id"])
        async with self.session_factory.begin() as session:
            order = await session.scalar(
                select(OpsWorkOrder).where(OpsWorkOrder.id == ticket_id).with_for_update()
            )
            if order is None or order.ticket_no != payload["ticket_no"]:
                raise ValueError("派单事件与工单身份不匹配")
            if order.session_id != payload["session_id"]:
                raise ValueError("派单事件会话与工单不匹配")
            if order.status == "PENDING":
                order.status = "DISPATCHED"
                await session.flush()
            ticket = TicketResult.model_validate(order)
        try:
            await self.redis.set(
                f"idem:ticket:{ticket.idempotency_key}",
                ticket.model_dump_json(),
                ex=IDEMPOTENCY_TTL,
            )
        except RedisError:
            # 缓存刷新不改变已提交的业务结果，下一次数据库读取仍能得到正确状态。
            logger.warning("已派单工单的缓存刷新失败", ticket_no=ticket.ticket_no)

    async def process_message(self, message_id: str | bytes, fields: Mapping) -> bool:
        """在消息独立的链路上下文中处理，退出时恢复消费者原有上下文。"""
        payload = {_decode(key): _decode(value) for key, value in fields.items()}
        trace_token = trace_id_var.set(payload.get("trace_id"))
        try:
            return await self._process_message(message_id, payload)
        finally:
            trace_id_var.reset(trace_token)

    async def _process_message(self, message_id: str | bytes, fields: Mapping) -> bool:
        """只处理待确认消息；返回是否取得处理锁，失败计数与死信迁移均原子完成。"""
        message_id = _decode(message_id)
        payload = {_decode(key): _decode(value) for key, value in fields.items()}
        token = uuid4().hex
        lock_key = self.lock_prefix + message_id
        acquired = await self.redis.set(
            lock_key, token, nx=True, px=round((self.processing_timeout + 5) * 1000)
        )
        if not acquired:
            return False
        try:
            pending = await self.redis.xpending_range(
                self.stream_name, self.group_name, message_id, message_id, 1
            )
            if not pending:
                return False
            try:
                await asyncio.wait_for(self.processor(payload), timeout=self.processing_timeout)
            except Exception as error:
                # 主动取消不记失败，消息留在 PEL 中等待其他消费者恢复。
                attempts = await self.redis.eval(
                    _FAILURE_SCRIPT,
                    4,
                    self.stream_name,
                    self.dlq_name,
                    self.failure_key,
                    lock_key,
                    self.group_name,
                    message_id,
                    json.dumps(payload, ensure_ascii=False),
                    f"{type(error).__name__}: {error}",
                    token,
                )
                logger.warning(
                    "派单处理失败",
                    message_id=message_id,
                    attempts=attempts,
                    dead_lettered=attempts >= 3,
                    trace_id=payload.get("trace_id"),
                )
            else:
                await self.redis.eval(
                    _ACK_SCRIPT,
                    3,
                    self.stream_name,
                    self.failure_key,
                    lock_key,
                    self.group_name,
                    message_id,
                    token,
                )
            return True
        finally:
            # 令牌匹配避免过期持有者误删另一消费者的处理锁。
            await self.redis.eval(_RELEASE_SCRIPT, 1, lock_key, token)

    async def _read_own(self, count: int) -> list:
        """轮转本消费者的待确认游标，避免前面的慢消息阻塞后面的重试。"""
        result = await self.redis.xreadgroup(
            self.group_name,
            self.consumer_name,
            {self.stream_name: self._pending_cursor},
            count=count,
        )
        messages = list(result[0][1]) if result else []
        if not messages and self._pending_cursor != "0-0":
            self._pending_cursor = "0-0"
            result = await self.redis.xreadgroup(
                self.group_name, self.consumer_name, {self.stream_name: "0-0"}, count=count
            )
            messages = list(result[0][1]) if result else []
        if messages:
            self._pending_cursor = _decode(messages[-1][0])
        return messages

    async def _read_reclaimed(self, count: int) -> list:
        """续用服务端扫描游标，避免大量未到期消息遮挡队尾的失联消息。"""
        result = await self.redis.xautoclaim(
            self.stream_name,
            self.group_name,
            self.consumer_name,
            min_idle_time=self.reclaim_idle_ms,
            start_id=self._claim_cursor,
            count=count,
        )
        self._claim_cursor = _decode(result[0])
        return list(result[1])

    async def _read_fresh(self, count: int) -> list:
        result = await self.redis.xreadgroup(
            self.group_name, self.consumer_name, {self.stream_name: ">"}, count=count
        )
        return list(result[0][1]) if result else []

    async def run_once(self, *, count: int = 10) -> int:
        """恢复发件箱后轮换三类消息的读取优先级，保证重试与新事件都有机会。"""
        if not 1 <= count <= 1000:
            raise ValueError("消费批次必须为 1 至 1000")
        await self.ensure_group()
        await self.publisher.publish_pending(limit=count)
        readers = (self._read_own, self._read_reclaimed, self._read_fresh)
        messages: list = []
        seen: set[str] = set()
        offset = self._source_offset
        self._source_offset = (offset + 1) % len(readers)
        for step in range(len(readers)):
            for message_id, fields in await readers[(offset + step) % len(readers)](
                count - len(messages)
            ):
                identity = _decode(message_id)
                if identity not in seen:
                    messages.append((message_id, fields))
                    seen.add(identity)
            if len(messages) >= count:
                break
        processed = 0
        for message_id, fields in messages:
            processed += await self.process_message(message_id, fields)
        return processed

    async def run_forever(self, *, poll_interval: float = 0.5) -> None:
        """可由进程信号取消的工作循环，依赖抖动不会永久退出消费者。"""
        if not 0 < poll_interval <= 60:
            raise ValueError("轮询间隔必须在 0 至 60 秒内")
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("工单消费者本轮执行失败，稍后重试")
            await asyncio.sleep(poll_interval)
