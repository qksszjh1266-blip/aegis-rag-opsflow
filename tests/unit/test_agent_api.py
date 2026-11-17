"""REST 入口只信任认证层权限，禁止用户伪造 ACL 字段。"""

from unittest.mock import AsyncMock

import httpx

from aegis.api.v1.agent import authenticated_access
from aegis.config import Settings
from aegis.main import create_app
from aegis.schemas.agent import AccessContext, AgentResponse


async def test_unauthenticated_agent_rejects_forged_permission_headers():
    app = create_app(Settings(_env_file=None))
    app.state.workflow = AsyncMock()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/agent",
            json={"session_id": "session", "request_id": "request", "message": "查看财务规则"},
            headers={"X-Department": "finance", "X-Role-Level": "99"},
        )
    assert response.status_code == 401
    app.state.workflow.run.assert_not_called()


async def test_authenticated_agent_uses_dependency_and_validates_contract():
    app = create_app(Settings(_env_file=None))
    access = AccessContext(subject_id="operator-1", dept_scope=["warehouse"], role_level=1)
    app.dependency_overrides[authenticated_access] = lambda: access
    app.state.workflow = AsyncMock()
    app.state.workflow.run.return_value = AgentResponse(
        answer="请补充设备编号。", trace_id="test", guardrail_action="SLOT_CLARIFICATION"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {"session_id": "session", "request_id": "request", "message": "扫描失败"}
        response = await client.post("/api/v1/agent", json=body)
        assert response.status_code == 200
        request, received_access = app.state.workflow.run.call_args.args
        assert request.message == "扫描失败"
        assert received_access == access
        forbidden = await client.post("/api/v1/agent", json={**body, "role_level": 99})
        assert forbidden.status_code == 422
        invalid = await client.post("/api/v1/agent", json={**body, "request_id": ""})
        assert invalid.status_code == 422
    app.state.workflow.run.assert_awaited_once()
