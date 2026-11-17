"""应用链路、错误边界、资源回收及真实探针执行逻辑的单元测试。"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
import structlog
from fastapi import Request

from aegis import main
from aegis.config import Settings
from aegis.core.exceptions import (
    CircuitBreakerOpenError,
    ResourceNotFoundError,
    SessionLockBusyError,
)
from aegis.core.logging import configure_logging
from aegis.core.trace_context import get_trace_id


class FakeEngine:
    """替代网络连接，保留异步连接与执行语义。"""

    def __init__(self) -> None:
        self.closed = False
        self.statements: list[str] = []
        self.action = self.succeed

    async def succeed(self) -> None:
        await asyncio.sleep(0)

    @asynccontextmanager
    async def connect(self):
        yield self

    async def execute(self, statement):
        self.statements.append(str(statement))
        await self.action()

    async def dispose(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self) -> None:
        self.closed = False
        self.action = self.succeed

    async def succeed(self) -> bool:
        await asyncio.sleep(0)
        return True

    async def ping(self) -> bool:
        return await self.action()

    async def aclose(self) -> None:
        self.closed = True


class FakeQdrant:
    def __init__(self) -> None:
        self.closed = False
        self.action = self.succeed

    async def succeed(self):
        await asyncio.sleep(0)
        return SimpleNamespace(collections=[])

    async def get_collections(self):
        return await self.action()

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def dependencies(monkeypatch):
    engine, redis, qdrant = FakeEngine(), FakeRedis(), FakeQdrant()
    monkeypatch.setattr(main, "create_mysql_engine", lambda settings: engine)
    monkeypatch.setattr(main, "create_session_factory", lambda engine: object())
    monkeypatch.setattr(main, "create_redis_client", lambda settings: redis)
    monkeypatch.setattr(main, "create_qdrant_client", lambda settings: qdrant)
    return engine, redis, qdrant


@pytest.fixture
def application(dependencies):
    application = main.create_app(Settings(_env_file=None, dependency_timeout_seconds=0.1))

    @application.get("/test/failure/{kind}")
    async def failure(kind: str):
        errors = {
            "missing": ResourceNotFoundError,
            "locked": SessionLockBusyError,
            "circuit": CircuitBreakerOpenError,
            "unexpected": RuntimeError,
        }
        raise errors[kind]("测试错误")

    @application.get("/test/validate")
    async def validate(count: int):
        return {"count": count}

    @application.get("/test/context/{item}")
    async def context(item: str, request: Request):
        await asyncio.sleep(0)
        structlog.get_logger("test_context").info("上下文日志", item=item)
        return {"trace_id": get_trace_id(), "state_trace_id": request.state.trace_id}

    return application


@asynccontextmanager
async def client_for(application):
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            yield client


async def test_trace_id_propagation_generation_and_reset(application):
    async with client_for(application) as client:
        supplied = await client.get("/health/live", headers={"X-Trace-ID": "request-123"})
        generated = await client.get("/health/live")
    assert supplied.json() == {"status": "ok", "trace_id": "request-123"}
    assert supplied.headers["x-trace-id"] == "request-123"
    trace_id = generated.json()["trace_id"]
    assert generated.headers["x-trace-id"] == trace_id
    assert UUID(trace_id).version == 4
    assert len(trace_id) == 32
    assert get_trace_id() is None


async def test_unsafe_trace_id_is_replaced(application):
    async with client_for(application) as client:
        response = await client.get("/health/live", headers={"X-Trace-ID": "x" * 65})
    assert UUID(response.json()["trace_id"]).version == 4


@pytest.mark.parametrize(
    ("path", "status", "code"),
    [
        ("/test/failure/missing", 404, "RESOURCE_NOT_FOUND"),
        ("/test/failure/locked", 409, "SESSION_LOCK_BUSY"),
        ("/test/failure/circuit", 503, "CIRCUIT_BREAKER_OPEN"),
        ("/test/failure/unexpected", 500, "INTERNAL_SERVER_ERROR"),
        ("/test/validate?count=bad", 422, "VALIDATION_ERROR"),
        ("/unknown", 404, "HTTP_404"),
    ],
)
async def test_errors_include_matching_trace(application, path, status, code):
    async with client_for(application) as client:
        response = await client.get(path, headers={"X-Trace-ID": "failure-trace"})
    assert response.status_code == status
    assert response.headers["x-trace-id"] == "failure-trace"
    assert response.json()["trace_id"] == "failure-trace"
    assert response.json()["error_code"] == code
    assert response.json()["message"]
    if status == 500:
        assert "测试错误" not in response.text
    assert get_trace_id() is None


async def test_concurrent_requests_keep_isolated_context(application, capsys):
    configure_logging()
    async with client_for(application) as client:
        responses = await asyncio.gather(
            *(
                client.get(f"/test/context/{index}", headers={"X-Trace-ID": f"trace-{index}"})
                for index in range(10)
            )
        )
    for index, response in enumerate(responses):
        assert response.json() == {"trace_id": f"trace-{index}", "state_trace_id": f"trace-{index}"}
    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    contextual = [event for event in events if event.get("event") == "上下文日志"]
    assert len(contextual) == 10
    for event in contextual:
        assert event["trace_id"] == f"trace-{event['item']}"
    assert get_trace_id() is None


async def test_readiness_queries_every_dependency(application, dependencies):
    async with client_for(application) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["components"]) == {"mysql", "redis", "qdrant"}
    for component in response.json()["components"].values():
        assert component["status"] == "ok"
        assert component["latency_ms"] >= 0
    assert dependencies[0].statements == ["SELECT 1"]
    assert all(dependency.closed for dependency in dependencies)


async def test_readiness_probes_are_concurrent(application, dependencies):
    started = 0
    all_started = asyncio.Event()

    async def parallel_probe():
        nonlocal started
        started += 1
        if started == 3:
            all_started.set()
        await all_started.wait()
        return True

    for dependency in dependencies:
        dependency.action = parallel_probe
    async with client_for(application) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 200
    assert started == 3


@pytest.mark.parametrize("failed_index", [0, 1, 2])
async def test_readiness_failure_returns_503(application, dependencies, failed_index):
    async def fail():
        raise ConnectionError("密码等连接细节不应泄露")

    dependencies[failed_index].action = fail
    async with client_for(application) as client:
        response = await client.get("/health/ready", headers={"X-Trace-ID": "probe-failed"})
    assert response.status_code == 503
    assert response.json()["trace_id"] == "probe-failed"
    assert sum(c["status"] == "error" for c in response.json()["components"].values()) == 1
    assert "密码" not in response.text


async def test_readiness_timeout_and_cancellation(application, dependencies):
    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    dependencies[2].action = slow
    async with client_for(application) as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["components"]["qdrant"]["error"] == "timeout"
    assert cancelled.is_set()


async def test_live_does_not_probe_dependencies(application, dependencies):
    async def forbidden():
        pytest.fail("存活探针不应访问任何依赖")

    for dependency in dependencies:
        dependency.action = forbidden
    async with client_for(application) as client:
        response = await client.get("/health/live")
    assert response.status_code == 200


async def test_metrics_use_route_template_and_status(application):
    async with client_for(application) as client:
        await client.get("/test/context/private-session-id")
        await client.get("/nonexistent/private-session-id")
        await client.get("/test/failure/unexpected")
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert 'route="/test/context/{item}"' in response.text
    assert 'route="__unmatched__"' in response.text
    assert 'status_code="500"' in response.text
    assert "private-session-id" not in response.text
    assert "_bucket{" in response.text


async def test_partial_initialization_releases_prior_resources(monkeypatch, dependencies):
    def fail(settings):
        raise RuntimeError("Qdrant 初始化失败")

    monkeypatch.setattr(main, "create_qdrant_client", fail)
    application = main.create_app(Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="初始化失败"):
        async with application.router.lifespan_context(application):
            pytest.fail("初始化失败后不应进入运行状态")
    assert dependencies[0].closed
    assert dependencies[1].closed
    assert not dependencies[2].closed


async def test_cleanup_continues_when_one_close_fails(monkeypatch, dependencies):
    async def fail_close():
        raise RuntimeError("关闭失败")

    monkeypatch.setattr(dependencies[2], "close", fail_close)
    application = main.create_app(Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="关闭失败"):
        async with application.router.lifespan_context(application):
            pass
    assert dependencies[0].closed
    assert dependencies[1].closed
