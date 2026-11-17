"""验证共享熔断状态、原子试探准入及陈旧回调隔离。"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import ConnectionError

from aegis.core.circuit_breaker import CircuitState, RedisCircuitBreaker
from aegis.core.exceptions import CircuitBreakerOpenError
from aegis.core.metrics import CIRCUIT_REJECTIONS, CIRCUIT_STATE


@pytest.fixture
async def redis():
    async with FakeRedis(decode_responses=True) as client:
        yield client


async def succeed(value="成功"):
    return value


async def fail():
    raise ValueError("上游不可用")


async def open_breaker(breaker):
    for _ in range(breaker.failure_threshold):
        with pytest.raises(ValueError, match="上游不可用"):
            await breaker.call(fail)
    assert await breaker.get_state() == CircuitState.OPEN


async def skip_cooldown(redis, breaker):
    # 除完整流转用例外，移动服务端熔断时间以避免慢速或不稳定的睡眠。
    await redis.hset(breaker.key, "opened_at", 0)


async def test_closed_calls_preserve_arguments_and_results(redis):
    breaker = RedisCircuitBreaker(redis, "embedding")
    assert breaker.key == "cb:provider:embedding"
    assert await breaker.get_state() == CircuitState.CLOSED
    assert await breaker.call(succeed, value="原样返回") == "原样返回"
    assert await breaker.get_state() == CircuitState.CLOSED


async def test_closed_open_half_open_closed_cycle(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", recovery_timeout=0.03)
    await open_breaker(breaker)
    assert CIRCUIT_STATE.labels(service="embedding")._value.get() == 1
    before = CIRCUIT_REJECTIONS.labels(service="embedding")._value.get()
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(succeed)
    assert CIRCUIT_REJECTIONS.labels(service="embedding")._value.get() == before + 1
    await asyncio.sleep(0.04)

    async def probe():
        assert await breaker.get_state() == CircuitState.HALF_OPEN
        assert CIRCUIT_STATE.labels(service="embedding")._value.get() == 2
        return "已恢复"

    assert await breaker.call(probe) == "已恢复"
    assert await breaker.get_state() == CircuitState.CLOSED
    assert await redis.zcard(breaker.failure_key) == 0
    assert CIRCUIT_STATE.labels(service="embedding")._value.get() == 0


async def test_success_does_not_clear_closed_failure_window(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=2)
    with pytest.raises(ValueError):
        await breaker.call(fail)
    await breaker.call(succeed)
    with pytest.raises(ValueError):
        await breaker.call(fail)
    assert await breaker.get_state() == CircuitState.OPEN


async def test_expired_failures_leave_sliding_window(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=2, window_seconds=0.03)
    with pytest.raises(ValueError):
        await breaker.call(fail)
    await asyncio.sleep(0.04)
    with pytest.raises(ValueError):
        await breaker.call(fail)
    assert await breaker.get_state() == CircuitState.CLOSED
    assert await redis.zcard(breaker.failure_key) == 1


async def test_half_open_failure_reopens_and_refreshes_cooldown(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1)
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)
    with pytest.raises(ValueError):
        await breaker.call(fail)
    assert await breaker.get_state() == CircuitState.OPEN
    assert float(await redis.hget(breaker.key, "opened_at")) > 0
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(succeed)


async def test_half_open_allows_one_probe_across_instances(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1)
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def probe():
        entered.set()
        await release.wait()

    first = asyncio.create_task(breaker.call(probe))
    await entered.wait()
    peers = [RedisCircuitBreaker(redis, "embedding", failure_threshold=1) for _ in range(20)]
    results = await asyncio.gather(*(peer.call(succeed) for peer in peers), return_exceptions=True)
    assert all(isinstance(result, CircuitBreakerOpenError) for result in results)
    assert await redis.zcard(breaker.probe_key) == 1
    release.set()
    await first
    assert await breaker.get_state() == CircuitState.CLOSED


async def test_half_open_requires_configured_number_of_successes(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1, half_open_max_calls=2)
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)
    await breaker.call(succeed)
    assert await breaker.get_state() == CircuitState.HALF_OPEN
    await breaker.call(succeed)
    assert await breaker.get_state() == CircuitState.CLOSED


async def test_half_open_limits_concurrency_and_remaining_success_budget(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1, half_open_max_calls=2)
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)
    entered = [asyncio.Event(), asyncio.Event()]
    releases = [asyncio.Event(), asyncio.Event()]

    async def probe(index):
        entered[index].set()
        await releases[index].wait()

    tasks = [asyncio.create_task(breaker.call(probe, index)) for index in range(2)]
    await asyncio.gather(*(event.wait() for event in entered))
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(succeed)
    releases[0].set()
    await tasks[0]
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(succeed)
    releases[1].set()
    await tasks[1]
    assert await breaker.get_state() == CircuitState.CLOSED


async def test_late_closed_failure_cannot_pollute_new_generation(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_failure():
        entered.set()
        await release.wait()
        await fail()

    slow = asyncio.create_task(breaker.call(slow_failure))
    await entered.wait()
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)
    await breaker.call(succeed)
    release.set()
    with pytest.raises(ValueError):
        await slow
    assert await breaker.get_state() == CircuitState.CLOSED
    assert await redis.zcard(breaker.failure_key) == 0


async def test_expired_probe_reopens_and_late_success_cannot_close(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1)
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_probe():
        entered.set()
        await release.wait()

    slow = asyncio.create_task(breaker.call(slow_probe))
    await entered.wait()
    token = (await redis.zrange(breaker.probe_key, 0, -1))[0]
    await redis.zadd(breaker.probe_key, {token: 0})
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(succeed)
    release.set()
    await slow
    assert await breaker.get_state() == CircuitState.OPEN


async def test_probe_finishing_after_lease_expiry_reopens(redis):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1)
    await open_breaker(breaker)
    await skip_cooldown(redis, breaker)

    async def expired_probe():
        token = (await redis.zrange(breaker.probe_key, 0, -1))[0]
        await redis.zadd(breaker.probe_key, {token: 0})

    await breaker.call(expired_probe)
    assert await breaker.get_state() == CircuitState.OPEN


@pytest.mark.parametrize("half_open", [False, True])
async def test_cancellation_reclaims_half_open_probe_without_counting_closed_failure(
    redis, half_open
):
    breaker = RedisCircuitBreaker(redis, "embedding", failure_threshold=1)
    if half_open:
        await open_breaker(breaker)
        await skip_cooldown(redis, breaker)
    entered = asyncio.Event()

    async def cancelled_call():
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(breaker.call(cancelled_call))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    expected = CircuitState.OPEN if half_open else CircuitState.CLOSED
    assert await breaker.get_state() == expected
    assert await redis.zcard(breaker.probe_key) == 0
    if not half_open:
        assert await redis.zcard(breaker.failure_key) == 0


async def test_redis_admission_failure_does_not_call_upstream(redis, monkeypatch):
    breaker = RedisCircuitBreaker(redis, "embedding")
    upstream = AsyncMock()
    monkeypatch.setattr(redis, "eval", AsyncMock(side_effect=ConnectionError("连接中断")))
    with pytest.raises(ConnectionError):
        await breaker.call(upstream)
    upstream.assert_not_awaited()


async def test_record_failure_preserves_upstream_success(redis, monkeypatch):
    breaker = RedisCircuitBreaker(redis, "embedding")
    monkeypatch.setattr(
        redis,
        "eval",
        AsyncMock(side_effect=[[1, "CLOSED", 0], ConnectionError("记录失败")]),
    )
    assert await breaker.call(succeed) == "成功"


async def test_bytes_responses_are_supported():
    async with FakeRedis(decode_responses=False) as client:
        breaker = RedisCircuitBreaker(client, "embedding", failure_threshold=1)
        await open_breaker(breaker)
        await skip_cooldown(client, breaker)
        await breaker.call(succeed)
        assert await breaker.get_state() == CircuitState.CLOSED


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_seconds": 0},
        {"window_seconds": float("nan")},
        {"recovery_timeout": -1},
        {"probe_timeout": 0},
        {"failure_threshold": 1.5},
        {"failure_threshold": True},
        {"half_open_max_calls": 0},
    ],
)
def test_invalid_configuration(redis, kwargs):
    with pytest.raises(ValueError):
        RedisCircuitBreaker(redis, "embedding", **kwargs)
