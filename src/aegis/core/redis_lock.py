"""使用随机所有权令牌和原子释放脚本实现会话互斥。"""

import asyncio
from types import TracebackType
from uuid import uuid4

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from aegis.core.exceptions import SessionLockBusyError
from aegis.core.metrics import LOCK_CONFLICTS

logger = structlog.get_logger(__name__)

_RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisSessionLock:
    """固定租期的会话锁；持锁业务必须在租期内完成，不自动续租。"""

    def __init__(self, redis: Redis, session_id: str, ttl: int = 15) -> None:
        if not session_id or len(session_id) > 64:
            raise ValueError("会话标识长度必须为 1 至 64 个字符")
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
            raise ValueError("锁租期必须为正整数秒")
        self.redis = redis
        self.key = f"lock:session:{session_id}"
        self.ttl = ttl
        self._token: str | None = None

    @property
    def ownership_token(self) -> str:
        """提供给原子状态写入脚本；令牌不得暴露给客户端或写入日志。"""
        if self._token is None:
            raise RuntimeError("会话锁尚未取得")
        return self._token

    async def acquire(self) -> None:
        """单次尝试加锁，冲突立即返回业务异常。"""
        if self._token is not None:
            raise RuntimeError("同一锁实例不能重复或并发加锁")
        token = uuid4().hex
        self._token = token
        try:
            acquired = await self.redis.set(self.key, token, nx=True, ex=self.ttl)
        except BaseException:
            self._token = None
            raise
        if not acquired:
            self._token = None
            LOCK_CONFLICTS.inc()
            raise SessionLockBusyError("当前会话正在处理中，请稍后重试")

    async def release(self) -> None:
        """仅删除属于本次持有者的锁；释放失败时保留令牌供重试。"""
        token = self._token
        if token is not None:
            await self.redis.eval(_RELEASE_SCRIPT, 1, self.key, token)
            # 并发释放的慢响应不能清空此实例下一次加锁的令牌。
            if self._token == token:
                self._token = None

    async def __aenter__(self) -> "RedisSessionLock":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # 即使请求被取消，也让已开始的原子释放完成；网络中断时仍由 TTL 兜底。
        try:
            await asyncio.shield(self.release())
        except RedisError:
            if exc is None:
                raise
            # 释放故障不能覆盖业务异常或取消；保留所有权令牌供调用方重试。
            logger.exception("会话锁释放失败，等待租期到期", lock_key=self.key)
