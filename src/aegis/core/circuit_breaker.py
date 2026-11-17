"""以 Redis 为事实源的滑动窗口三态熔断器。"""

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import ParamSpec, TypeVar
from uuid import uuid4

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from aegis.core.exceptions import CircuitBreakerOpenError
from aegis.core.metrics import CIRCUIT_REJECTIONS, CIRCUIT_STATE

P = ParamSpec("P")
T = TypeVar("T")
logger = structlog.get_logger(__name__)


class CircuitState(StrEnum):
    """共享状态名称与对外指标保持一一对应。"""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


_STATE_VALUES = {CircuitState.CLOSED: 0, CircuitState.OPEN: 1, CircuitState.HALF_OPEN: 2}

# 所有时间判断统一使用 Redis 时钟，避免多应用节点时钟偏差。
# generation 为每次转换的代际，阻止上一个状态的慢请求覆盖当前状态。
_ADMIT_SCRIPT = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation') or '0')
local recovery = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local probe_timeout = tonumber(ARGV[3])
local token = ARGV[4]
if state == 'OPEN' then
    local opened_at = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
    if now - opened_at < recovery then
        return {0, state, generation}
    end
    generation = generation + 1
    state = 'HALF_OPEN'
    redis.call('HSET', KEYS[1], 'state', state, 'generation', generation, 'successes', 0)
    redis.call('DEL', KEYS[3])
end
if state == 'HALF_OPEN' then
    if redis.call('ZCOUNT', KEYS[3], '-inf', now) > 0 then
        generation = generation + 1
        redis.call('HSET', KEYS[1], 'state', 'OPEN', 'generation', generation,
                   'opened_at', now, 'successes', 0)
        redis.call('DEL', KEYS[3])
        return {0, 'OPEN', generation}
    end
    local successes = tonumber(redis.call('HGET', KEYS[1], 'successes') or '0')
    if redis.call('ZCARD', KEYS[3]) + successes >= limit then
        return {0, state, generation}
    end
    redis.call('ZADD', KEYS[3], now + probe_timeout, token)
else
    redis.call('HSETNX', KEYS[1], 'state', 'CLOSED')
    redis.call('HSETNX', KEYS[1], 'generation', generation)
end
return {1, state, generation}
"""

_FINISH_SCRIPT = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local state = redis.call('HGET', KEYS[1], 'state') or 'CLOSED'
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation') or '0')
if state ~= ARGV[1] or generation ~= tonumber(ARGV[2]) then
    return state
end
local success = ARGV[3] == '1'
local token = ARGV[4]
local window = tonumber(ARGV[5])
local threshold = tonumber(ARGV[6])
local probe_limit = tonumber(ARGV[7])
if state == 'HALF_OPEN' then
    local deadline = redis.call('ZSCORE', KEYS[3], token)
    if not deadline then
        return state
    end
    redis.call('ZREM', KEYS[3], token)
    if not success or tonumber(deadline) <= now then
        redis.call('HSET', KEYS[1], 'state', 'OPEN', 'generation', generation + 1,
                   'opened_at', now, 'successes', 0)
        redis.call('DEL', KEYS[3])
        return 'OPEN'
    end
    local successes = redis.call('HINCRBY', KEYS[1], 'successes', 1)
    if successes >= probe_limit then
        redis.call('HSET', KEYS[1], 'state', 'CLOSED', 'generation', generation + 1,
                   'successes', 0)
        redis.call('HDEL', KEYS[1], 'opened_at')
        redis.call('DEL', KEYS[2], KEYS[3])
        return 'CLOSED'
    end
elseif state == 'CLOSED' then
    redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - window)
    if not success then
        redis.call('ZADD', KEYS[2], now, token)
        redis.call('PEXPIRE', KEYS[2], math.ceil(window * 1000))
        if redis.call('ZCARD', KEYS[2]) >= threshold then
            redis.call('HSET', KEYS[1], 'state', 'OPEN', 'generation', generation + 1,
                       'opened_at', now, 'successes', 0)
            return 'OPEN'
        end
    end
end
return state
"""


class RedisCircuitBreaker:
    """共享失败窗口和半开试探名额，支持多个进程并发使用。

    half_open_max_calls 同时表示半开恢复所需成功数和最大并发数。
    probe_timeout 限制半开租期；超时回调不能关闭熔断器，后续请求会重新熔断。
    调用方仍应为上游网络调用设置超时，本类不会强行终止业务协程。
    """

    def __init__(
        self,
        redis: Redis,
        service_name: str,
        *,
        window_seconds: float = 30,
        failure_threshold: int = 5,
        recovery_timeout: float = 15,
        half_open_max_calls: int = 1,
        probe_timeout: float = 30,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", service_name) is None:
            raise ValueError("服务名必须是 1 至 64 位字母、数字、下划线、点或连字符")
        for name, value in {
            "window_seconds": window_seconds,
            "recovery_timeout": recovery_timeout,
            "probe_timeout": probe_timeout,
        }.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        for name, value in {
            "failure_threshold": failure_threshold,
            "half_open_max_calls": half_open_max_calls,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        self.redis = redis
        self.service_name = service_name
        self.key = f"cb:provider:{service_name}"
        self.failure_key = f"{self.key}:failures"
        self.probe_key = f"{self.key}:probes"
        self.window_seconds = window_seconds
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.probe_timeout = probe_timeout

    def _observe(self, state: str | bytes) -> CircuitState:
        result = CircuitState(state.decode() if isinstance(state, bytes) else state)
        CIRCUIT_STATE.labels(service=self.service_name).set(_STATE_VALUES[result])
        return result

    async def get_state(self) -> CircuitState:
        """读取共享状态并刷新当前进程的指标；冷却结束由调用准入触发转换。"""
        state = await self.redis.hget(self.key, "state") or "CLOSED"
        return self._observe(state)

    async def _finish(
        self, state: CircuitState, generation: int, token: str, *, success: bool
    ) -> None:
        try:
            result = await self.redis.eval(
                _FINISH_SCRIPT,
                3,
                self.key,
                self.failure_key,
                self.probe_key,
                state.value,
                generation,
                int(success),
                token,
                self.window_seconds,
                self.failure_threshold,
                self.half_open_max_calls,
            )
        except RedisError:
            # 记录失败不覆盖已得到的业务结果；下一次准入仍必须成功访问 Redis。
            logger.exception("熔断器结果记录失败", service=self.service_name)
        else:
            self._observe(result)

    async def call(self, func: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        """准入后调用异步函数；所有拒绝均在执行上游函数之前完成。"""
        token = uuid4().hex
        allowed, raw_state, generation = await self.redis.eval(
            _ADMIT_SCRIPT,
            3,
            self.key,
            self.failure_key,
            self.probe_key,
            self.recovery_timeout,
            self.half_open_max_calls,
            self.probe_timeout,
            token,
        )
        state = self._observe(raw_state)
        if not allowed:
            CIRCUIT_REJECTIONS.labels(service=self.service_name).inc()
            raise CircuitBreakerOpenError(f"服务 {self.service_name} 暂时熔断，请稍后重试")
        try:
            result = await func(*args, **kwargs)
        except asyncio.CancelledError:
            # 取消不算普通上游错误，但半开名额必须回收并重新冷却。
            if state == CircuitState.HALF_OPEN:
                await asyncio.shield(self._finish(state, generation, token, success=False))
            raise
        except Exception:
            await self._finish(state, generation, token, success=False)
            raise
        else:
            await self._finish(state, generation, token, success=True)
            return result
