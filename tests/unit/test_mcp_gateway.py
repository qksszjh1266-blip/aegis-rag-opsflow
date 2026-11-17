"""通过真实 TCP 验证官方 MCP 客户端握手、注册表、鉴权及工具契约。"""

import asyncio
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError

from aegis.config import Settings
from aegis.main import create_app
from aegis.schemas.agent import AccessContext
from aegis.schemas.gateway import SearchResponse, VerifySKUResponse
from aegis.workers.tickets import TicketResult


@asynccontextmanager
async def serve_app(app):
    """绑定操作系统分配端口，确保握手测试不依赖固定端口或外部进程。"""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


def app_with_services():
    access = AccessContext(subject_id="mcp-user", dept_scope=["warehouse"], role_level=1)
    other = AccessContext(subject_id="other", dept_scope=["FINANCE"], role_level=3)
    app = create_app(
        Settings(_env_file=None, gateway_api_tokens={"test-key": access, "other": other})
    )
    app.state.gateway_services = SimpleNamespace(
        search=AsyncMock(
            return_value=SearchResponse(
                hits=[], rerank_fallback=False, latency_ms=1, trace_id="test"
            )
        ),
        verify_sku=AsyncMock(
            return_value=VerifySKUResponse(
                passed=True,
                corrections="",
                checked_count=1,
                claims=[],
                reference_values=[],
                trace_id="test",
            )
        ),
        create_ticket=AsyncMock(
            return_value=TicketResult(
                id=1,
                ticket_no="OPS-test",
                idempotency_key="test",
                session_id="session",
                trace_id="test",
                priority="P0_EMERGENCY",
                trigger_source="HIGH_RISK_PRECHECK",
                slots_payload={},
                completeness_score=0,
                context_summary=None,
                status="PENDING",
            )
        ),
    )
    return app, access


async def test_mcp_sse_tools_execute_with_authenticated_acl():
    app, access = app_with_services()
    async with serve_app(app) as base:
        async with sse_client(
            base + "/mcp/sse", headers={"Authorization": "Bearer test-key"}
        ) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                assert {tool.name for tool in tools} == {
                    "search_enterprise_kb",
                    "verify_sku_specs",
                    "create_ops_ticket",
                }
                for tool in tools:
                    assert tool.description
                    assert "body" in tool.inputSchema["properties"]
                    assert "ctx" not in tool.inputSchema["properties"]
                requests = {
                    "search_enterprise_kb": {"query": "净重", "top_k": 3},
                    "verify_sku_specs": {
                        "candidate_text": "EC-ih200-Std 净重 1800g",
                        "sku": "EC-ih200-Std",
                        "parent_id": "parent-test",
                    },
                    "create_ops_ticket": {
                        "session_id": "session",
                        "idempotency_key": "test",
                        "message": "冒烟",
                    },
                }
                for name, body in requests.items():
                    result = await session.call_tool(name, {"body": body})
                    assert not result.isError, result
                # 非法参数必须在领域服务执行前被拒绝。
                try:
                    result = await session.call_tool(
                        "search_enterprise_kb", {"body": {"query": "净重", "top_k": 0}}
                    )
                    assert result.isError
                except McpError:
                    pass
    for method in ("search", "verify_sku", "create_ticket"):
        mock = getattr(app.state.gateway_services, method)
        mock.assert_awaited_once()
        assert mock.call_args.args[1] == access


async def test_mcp_rejects_unauthenticated_and_cross_identity_messages():
    app, _ = app_with_services()
    async with serve_app(app) as base:
        async with httpx.AsyncClient(base_url=base) as client:
            assert (await client.get("/mcp/sse")).status_code == 401
            assert (await client.post("/mcp/messages/")).status_code == 401
            async with client.stream(
                "GET", "/mcp/sse", headers={"Authorization": "Bearer test-key"}
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        endpoint = line[6:]
                        break
                assert endpoint.startswith("/mcp/messages/?session_id=")
                rejected = await client.post(
                    endpoint,
                    headers={"Authorization": "Bearer other"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
                assert rejected.status_code == 404


async def test_mcp_internal_errors_do_not_disclose_details():
    app, _ = app_with_services()
    app.state.gateway_services.search.side_effect = RuntimeError("数据库私密连接信息")
    async with serve_app(app) as base:
        async with sse_client(
            base + "/mcp/sse", headers={"Authorization": "Bearer test-key"}
        ) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_enterprise_kb", {"body": {"query": "净重"}}
                )
                assert result.isError
                assert "数据库私密连接信息" not in result.model_dump_json()


async def test_mcp_private_gateway_host_keeps_dns_rebinding_protection():
    access = AccessContext(subject_id="private", dept_scope=["warehouse"], role_level=1)
    app = create_app(
        Settings(
            _env_file=None,
            public_base_url="https://ops.example.internal/aegis",
            gateway_api_tokens={"private-key": access},
        )
    )
    async with serve_app(app) as base:
        async with httpx.AsyncClient(base_url=base) as client:
            headers = {"Authorization": "Bearer private-key", "Host": "ops.example.internal"}
            async with client.stream("GET", "/mcp/sse", headers=headers) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        break
            denied = await client.get("/mcp/sse", headers={**headers, "Host": "evil.invalid"})
            assert denied.status_code == 421
