"""双层幂等工单服务与可恢复的事务发件箱发布器。"""

import json
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aegis.db.models import OpsTicketOutbox, OpsWorkOrder, WorkOrderPriority

TICKET_STREAM = "stream:ticket:events"
TICKET_DLQ = "stream:ticket:dlq"
TICKET_GROUP = "cg_ops_worker"
IDEMPOTENCY_TTL = 86400
logger = structlog.get_logger(__name__)

# 事件去重标记不使用 24 小时缓存租期；MySQL 回写前进程退出仍可安全重放。
# 发布标记与流必须一同持久化/恢复，不能单独清空发布标记。
_PUBLISH_SCRIPT = """
local existing = redis.call('GET', KEYS[2])
if existing then return existing end
local fields = cjson.decode(ARGV[1])
local args = {}
for key, value in pairs(fields) do
    table.insert(args, key)
    table.insert(args, value)
end
local message_id = redis.call('XADD', KEYS[1], '*', unpack(args))
redis.call('SET', KEYS[2], message_id)
return message_id
"""


class TicketResult(BaseModel):
    """可缓存的工单回执，重复请求返回原始工单身份。"""

    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: int = Field(description="工单数据库标识")
    ticket_no: str = Field(description="履约工单唯一编号")
    idempotency_key: str = Field(description="经过认证身份隔离的服务端防重键")
    session_id: str = Field(description="工单所属业务会话")
    trace_id: str = Field(description="创建工单的全链路追踪标识")
    priority: WorkOrderPriority = Field(description="确定性判定的工单优先级")
    trigger_source: str = Field(description="触发人工履约的来源")
    slots_payload: dict[str, Any] = Field(description="已采集排障槽位与问题描述")
    completeness_score: float = Field(description="排障槽位完整度")
    context_summary: str | None = Field(description="转人工时的会话摘要")
    status: str = Field(description="工单当前处理状态")


class _TicketRequest(BaseModel):
    """在任何缓存/数据库操作前校验持久化字段边界。"""

    idempotency_key: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=64)
    trace_id: str = Field(min_length=1, max_length=64)
    priority: WorkOrderPriority
    trigger_source: str = Field(min_length=1, max_length=64)
    slots_payload: dict[str, Any]
    completeness_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    context_summary: str | None = Field(default=None, max_length=16000)


def publication_key(event_id: str) -> str:
    return f"published:ticket:{event_id}"


class TicketService:
    """Redis 高速缓存配合 MySQL 唯一键；数据库事务是工单创建的事实源。"""

    def __init__(
        self,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        stream_name: str = TICKET_STREAM,
    ) -> None:
        if not stream_name or len(stream_name) > 128:
            raise ValueError("工单流名称长度必须为 1 至 128")
        self.redis = redis
        self.session_factory = session_factory
        self.stream_name = stream_name

    @staticmethod
    def _check_identity(ticket: TicketResult, request: _TicketRequest) -> None:
        if (
            ticket.session_id != request.session_id
            or ticket.idempotency_key != request.idempotency_key
        ):
            raise ValueError("工单幂等键已由其他会话或不同规范化键占用")

    async def create(
        self,
        *,
        idempotency_key: str,
        session_id: str,
        trace_id: str,
        priority: WorkOrderPriority,
        trigger_source: str,
        slots_payload: dict[str, Any],
        completeness_score: float,
        context_summary: str | None = None,
    ) -> TicketResult:
        request = _TicketRequest(
            idempotency_key=idempotency_key,
            session_id=session_id,
            trace_id=trace_id,
            priority=priority,
            trigger_source=trigger_source,
            slots_payload=slots_payload,
            completeness_score=completeness_score,
            context_summary=context_summary,
        )
        # JSON 只接受有限数值，不允许数据库 JSON 与 Redis 缓存解释不一致。
        json.dumps(request.slots_payload, ensure_ascii=False, allow_nan=False)
        key = f"idem:ticket:{request.idempotency_key}"
        try:
            cached = await self.redis.get(key)
        except RedisError:
            cached = None
        if cached is not None:
            ticket = TicketResult.model_validate_json(cached)
            self._check_identity(ticket, request)
            return ticket
        try:
            async with self.session_factory.begin() as session:
                order = OpsWorkOrder(
                    ticket_no=f"OPS-{uuid4().hex}",
                    **request.model_dump(),
                )
                session.add(order)
                await session.flush()
                ticket = TicketResult.model_validate(order)
                event_id = uuid4().hex
                payload = {
                    "event_id": event_id,
                    "ticket_id": str(order.id),
                    "ticket_no": order.ticket_no,
                    "idempotency_key": order.idempotency_key,
                    "session_id": order.session_id,
                    "trace_id": order.trace_id,
                    "priority": order.priority.value,
                    "trigger_source": order.trigger_source,
                    "slots_payload": json.dumps(order.slots_payload, ensure_ascii=False),
                    "completeness_score": str(order.completeness_score),
                    "context_summary": order.context_summary or "",
                }
                session.add(
                    OpsTicketOutbox(
                        event_id=event_id,
                        ticket_id=order.id,
                        stream_name=self.stream_name,
                        payload=payload,
                    )
                )
        except IntegrityError:
            # 并发唯一键冲突后必须换一个事务，避开 REPEATABLE READ 的旧快照。
            async with self.session_factory() as session:
                order = await session.scalar(
                    select(OpsWorkOrder).where(
                        OpsWorkOrder.idempotency_key == request.idempotency_key
                    )
                )
                if order is None:
                    raise
                ticket = TicketResult.model_validate(order)
                self._check_identity(ticket, request)
        try:
            await self.publish_ticket(ticket.id)
            # 消费者可能已经刷新到 DISPATCHED，迟到的创建回执不能覆盖新状态。
            await self.redis.set(key, ticket.model_dump_json(), ex=IDEMPOTENCY_TTL, nx=True)
        except RedisError:
            # 工单和事件已可靠提交；后台发布器恢复，不因 Redis 抖动创建第二张工单。
            logger.warning("工单已提交，派单事件等待恢复发布", ticket_no=ticket.ticket_no)
        return ticket

    async def _publish(self, row: OpsTicketOutbox) -> None:
        if row.published:
            return
        message_id = await self.redis.eval(
            _PUBLISH_SCRIPT,
            2,
            row.stream_name,
            publication_key(row.event_id),
            json.dumps(row.payload, ensure_ascii=False),
        )
        row.redis_message_id = message_id.decode() if isinstance(message_id, bytes) else message_id
        row.published = True

    async def publish_ticket(self, ticket_id: int) -> None:
        """锁定单条发件箱记录，发布成功再提交发送状态。"""
        async with self.session_factory.begin() as session:
            row = await session.scalar(
                select(OpsTicketOutbox)
                .where(OpsTicketOutbox.ticket_id == ticket_id)
                .with_for_update()
            )
            if row is not None:
                await self._publish(row)

    async def publish_pending(self, *, limit: int = 100) -> int:
        """多实例通过 SKIP LOCKED 分摊恢复任务，不重复派单。"""
        if not 1 <= limit <= 1000:
            raise ValueError("发件箱扫描批次必须为 1 至 1000")
        async with self.session_factory.begin() as session:
            rows = list(
                await session.scalars(
                    select(OpsTicketOutbox)
                    .where(
                        OpsTicketOutbox.published.is_(False),
                        OpsTicketOutbox.stream_name == self.stream_name,
                    )
                    .order_by(OpsTicketOutbox.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for row in rows:
                await self._publish(row)
            return len(rows)
