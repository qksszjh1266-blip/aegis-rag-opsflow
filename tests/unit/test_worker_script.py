"""消费者脚本在单轮结束及取消时都关闭连接资源。"""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_ops_worker.py"
_SPEC = importlib.util.spec_from_file_location("run_ops_worker", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT_MODULE)


@pytest.mark.parametrize("cancel", [False, True])
async def test_worker_script_closes_resources_after_single_run_or_cancellation(monkeypatch, cancel):
    engine = Mock(dispose=AsyncMock())
    redis = Mock(aclose=AsyncMock())
    started = asyncio.Event()

    async def forever(**kwargs):
        started.set()
        await asyncio.Event().wait()

    worker = Mock(run_once=AsyncMock(return_value=1), run_forever=AsyncMock(side_effect=forever))
    monkeypatch.setattr(_SCRIPT_MODULE, "create_mysql_engine", Mock(return_value=engine))
    monkeypatch.setattr(_SCRIPT_MODULE, "create_redis_client", Mock(return_value=redis))
    monkeypatch.setattr(_SCRIPT_MODULE, "create_session_factory", Mock())
    monkeypatch.setattr(_SCRIPT_MODULE, "configure_logging", Mock())
    monkeypatch.setattr(_SCRIPT_MODULE, "TicketWorker", Mock(return_value=worker))
    task = asyncio.create_task(
        _SCRIPT_MODULE.run(once=not cancel, consumer="test-worker", poll_interval=0.1)
    )
    if cancel:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
        worker.run_once.assert_awaited_once()
    redis.aclose.assert_awaited_once()
    engine.dispose.assert_awaited_once()
