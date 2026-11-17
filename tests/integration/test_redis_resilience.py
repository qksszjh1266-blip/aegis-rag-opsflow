"""在真实 Redis 上复核 Lua 原子语义和跨实例并发边界。"""

import asyncio
import os
from uuid import uuid4

import pytest

from aegis.config import Settings
from aegis.core.circuit_breaker import CircuitState, RedisCircuitBreaker
from aegis.core.exceptions import CircuitBreakerOpenError, SessionLockBusyError
from aegis.core.redis_lock import RedisSessionLock
from aegis.db.redis_client import create_redis_client

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("AEGIS_INTEGRATION") != "1", reason="真实 Redis 测试需 AEGIS_INTEGRATION=1"
    ),
]


@pytest.fixture
async def real_redis():
    client = create_redis_client(Settings())
    try:
        await client.ping()
        yield client
    finally:
        await client.aclose()


async def test_real_lock_expiration_preserves_new_owner(real_redis):
    session_id = f"test-{uuid4().hex}"
    old = RedisSessionLock(real_redis, session_id)
    new = RedisSessionLock(real_redis, session_id)
    try:
        await old.acquire()
        with pytest.raises(SessionLockBusyError):
            await new.acquire()
        # 使用真实 Redis 过期机制加速场景，避免测试等待完整的默认十五秒。
        await real_redis.pexpire(old.key, 10)
        await asyncio.sleep(0.03)
        await new.acquire()
        new_token = await real_redis.get(new.key)
        await old.release()
        assert await real_redis.get(new.key) == new_token
        await new.release()
        assert not await real_redis.exists(new.key)
    finally:
        await real_redis.delete(old.key)


async def test_real_breaker_shares_atomic_probe_between_clients(real_redis):
    service = f"test-{uuid4().hex}"
    second_client = create_redis_client(Settings())
    first = RedisCircuitBreaker(real_redis, service, failure_threshold=2, recovery_timeout=0.05)
    second = RedisCircuitBreaker(second_client, service, failure_threshold=2, recovery_timeout=0.05)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def fail():
        raise OSError("模拟供应商连接中断")

    async def probe():
        started.set()
        await finish.wait()
        return "recovered"

    task = None
    try:
        for breaker in (first, second):
            with pytest.raises(OSError):
                await breaker.call(fail)
        assert await first.get_state() is CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            await first.call(probe)
        assert not started.is_set()
        await asyncio.sleep(0.06)
        task = asyncio.create_task(first.call(probe))
        await asyncio.wait_for(started.wait(), 2)
        assert await second.get_state() is CircuitState.HALF_OPEN
        with pytest.raises(CircuitBreakerOpenError):
            await second.call(probe)
        finish.set()
        assert await task == "recovered"
        assert await second.get_state() is CircuitState.CLOSED
        assert await real_redis.zcard(first.failure_key) == 0
    finally:
        finish.set()
        if task is not None:
            await task
        await real_redis.delete(first.key, first.failure_key, first.probe_key)
        await second_client.aclose()
