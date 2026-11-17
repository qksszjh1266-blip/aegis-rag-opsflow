"""验证会话锁互斥、释放及所有权过期边界。"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import ConnectionError

from aegis.core.exceptions import SessionLockBusyError
from aegis.core.metrics import LOCK_CONFLICTS
from aegis.core.redis_lock import RedisSessionLock


@pytest.fixture
async def redis():
    async with FakeRedis(decode_responses=True) as client:
        yield client


async def test_lock_key_ttl_and_normal_release(redis):
    async with RedisSessionLock(redis, "session-1") as lock:
        assert lock.key == "lock:session:session-1"
        assert await redis.get(lock.key)
        assert 0 < await redis.ttl(lock.key) <= 15
    assert await redis.get(lock.key) is None


async def test_concurrent_acquisition_has_exactly_one_owner(redis):
    locks = [RedisSessionLock(redis, "same-session") for _ in range(20)]
    before = LOCK_CONFLICTS._value.get()
    results = await asyncio.gather(*(lock.acquire() for lock in locks), return_exceptions=True)
    assert results.count(None) == 1
    assert sum(isinstance(result, SessionLockBusyError) for result in results) == 19
    assert LOCK_CONFLICTS._value.get() - before == 19
    winner = locks[results.index(None)]
    await winner.release()
    assert await redis.exists(winner.key) == 0


async def test_expired_owner_cannot_delete_new_owner(redis):
    old_owner = RedisSessionLock(redis, "session-1")
    new_owner = RedisSessionLock(redis, "session-1")
    await old_owner.acquire()
    old_token = await redis.get(old_owner.key)
    await redis.pexpire(old_owner.key, 1)
    await asyncio.sleep(0.02)
    await new_owner.acquire()
    new_token = await redis.get(new_owner.key)
    assert old_token != new_token
    await old_owner.release()
    assert await redis.get(new_owner.key) == new_token
    await new_owner.release()
    assert await redis.get(new_owner.key) is None


async def test_exception_releases_lock(redis):
    with pytest.raises(ValueError, match="业务失败"):
        async with RedisSessionLock(redis, "session-1"):
            raise ValueError("业务失败")
    assert await redis.exists("lock:session:session-1") == 0


async def test_cancelled_request_releases_lock(redis):
    entered = asyncio.Event()

    async def work():
        async with RedisSessionLock(redis, "session-1"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await redis.exists("lock:session:session-1") == 0


async def test_release_failure_preserves_business_exception_and_retry_token(redis, monkeypatch):
    lock = RedisSessionLock(redis, "session-1")
    original_error = ValueError("业务失败")
    original_eval = redis.eval
    monkeypatch.setattr(redis, "eval", AsyncMock(side_effect=ConnectionError("释放连接中断")))
    with pytest.raises(ValueError) as caught:
        async with lock:
            raise original_error
    assert caught.value is original_error
    assert await redis.exists(lock.key) == 1
    assert await redis.ttl(lock.key) > 0
    monkeypatch.setattr(redis, "eval", original_eval)
    await lock.release()
    assert await redis.exists(lock.key) == 0


async def test_release_failure_preserves_cancellation_and_retry_token(redis, monkeypatch):
    lock = RedisSessionLock(redis, "session-1")
    entered = asyncio.Event()
    original_eval = redis.eval
    monkeypatch.setattr(redis, "eval", AsyncMock(side_effect=ConnectionError("释放连接中断")))

    async def work():
        async with lock:
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert await redis.exists(lock.key) == 1
    monkeypatch.setattr(redis, "eval", original_eval)
    await lock.release()
    assert await redis.exists(lock.key) == 0


async def test_release_failure_without_original_exception_propagates(redis, monkeypatch):
    lock = RedisSessionLock(redis, "session-1")
    original_eval = redis.eval
    release_error = ConnectionError("释放连接中断")
    monkeypatch.setattr(redis, "eval", AsyncMock(side_effect=release_error))
    with pytest.raises(ConnectionError) as caught:
        async with lock:
            pass
    assert caught.value is release_error
    assert await redis.exists(lock.key) == 1
    monkeypatch.setattr(redis, "eval", original_eval)
    await lock.release()
    assert await redis.exists(lock.key) == 0


async def test_lock_instance_can_be_reused_but_not_reentered(redis):
    lock = RedisSessionLock(redis, "session-1")
    async with lock:
        with pytest.raises(RuntimeError):
            await lock.acquire()
    async with lock:
        assert await redis.exists(lock.key) == 1
    await lock.release()


@pytest.mark.parametrize("ttl", [0, -1, 1.5, True])
def test_invalid_lock_ttl(redis, ttl):
    with pytest.raises(ValueError):
        RedisSessionLock(redis, "session-1", ttl=ttl)
