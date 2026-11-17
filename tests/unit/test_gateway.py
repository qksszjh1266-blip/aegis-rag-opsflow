"""双协议公共契约、鉴权、量纲回读和身份隔离防重的回归测试。"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.openapi.models import OpenAPI
from jsonschema import Draft202012Validator

from aegis.agent.services import GatewayServices
from aegis.config import Settings
from aegis.core.exceptions import ResourceNotFoundError
from aegis.db.models import WorkOrderPriority
from aegis.guardrails.precheck import HighRiskPrecheck
from aegis.guardrails.sku_verifier import SKUVerifier
from aegis.main import create_app
from aegis.schemas.agent import AccessContext
from aegis.schemas.common import ErrorResponse
from aegis.schemas.gateway import CreateTicketRequest, VerifySKURequest

ACCESS = AccessContext(subject_id="客服甲", dept_scope=["warehouse"], role_level=1)


def test_openapi_has_31_typed_business_contracts_and_unique_operations():
    document = create_app(Settings(_env_file=None)).openapi()
    OpenAPI.model_validate(document)
    assert document["openapi"] == "3.1.0"
    for schema in document["components"]["schemas"].values():
        Draft202012Validator.check_schema(schema)
    ids = []
    for path in (
        "/api/v1/kb/search",
        "/api/v1/guardrails/verify-sku",
        "/api/v1/tickets",
        "/api/v1/agent/chat",
    ):
        operation = document["paths"][path]["post"]
        assert operation["summary"] and operation["description"]
        assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        assert operation["security"]
        assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }
        ids.append(operation["operationId"])
    assert len(ids) == len(set(ids))


async def test_bearer_identity_wins_over_forged_headers_and_body_is_strict():
    app = create_app(Settings(_env_file=None, gateway_api_tokens={"测试令牌": ACCESS}))
    service = AsyncMock()
    app.state.gateway_services = service
    service.search.return_value = {
        "hits": [],
        "rerank_fallback": False,
        "latency_ms": 0,
        "trace_id": "测试",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # HTTP 标头只允许 ASCII，因此令牌在配置中也使用 ASCII。
        app.state.settings.gateway_api_tokens = {"test-token": ACCESS}
        headers = {"Authorization": "Bearer test-token", "X-Role-Level": "999"}
        response = await client.post(
            "/api/v1/kb/search", json={"query": "财务返利"}, headers=headers
        )
        assert response.status_code == 200
        assert service.search.call_args.args[1] == ACCESS
        for body in (
            {"query": "财务返利", "role_level": 999},
            {"query": "财务返利", "top_k": "5"},
            {"query": "  "},
        ):
            response = await client.post("/api/v1/kb/search", json=body, headers=headers)
            assert response.status_code == 422
            assert ErrorResponse.model_validate(response.json()).error_code == "VALIDATION_ERROR"
        response = await client.post(
            "/api/v1/kb/search", json={"query": "财务返利"}, headers={"X-Role-Level": "999"}
        )
        assert response.status_code == 401
    service.search.assert_awaited_once()


class Sessions:
    def __init__(self, parent_id="parent"):
        self.parent_id = parent_id
        self.audit = []

    @asynccontextmanager
    async def __call__(self):
        yield SimpleNamespace(scalar=AsyncMock(return_value=self.parent_id))

    @asynccontextmanager
    async def begin(self):
        yield SimpleNamespace(add=self.audit.append)


async def test_verifier_denies_unauthorized_parent_before_reading_rows(monkeypatch):
    rows = AsyncMock()
    monkeypatch.setattr("aegis.agent.services.load_sku_rows", rows)
    service = GatewayServices(SimpleNamespace(), Sessions(None), Settings(_env_file=None))
    with pytest.raises(ResourceNotFoundError):
        await service.verify_sku(
            VerifySKURequest(candidate_text="净重 1 kg", sku="EC-1", parent_id="secret"), ACCESS
        )
    rows.assert_not_awaited()


async def test_verifier_exposes_normalized_truth_and_rejects_wrong_row_value(monkeypatch):
    monkeypatch.setattr(
        "aegis.agent.services.load_sku_rows",
        AsyncMock(return_value=[{"SKU": "EC-1", "净重": "1.2 kg"}]),
    )
    sessions = Sessions()
    service = GatewayServices(
        SimpleNamespace(verifier=SKUVerifier()), sessions, Settings(_env_file=None)
    )
    request = VerifySKURequest(candidate_text="EC-1 净重 1200 g", sku="EC-1", parent_id="parent")
    passed = await service.verify_sku(request, ACCESS)
    assert passed.passed and passed.checked_count == 1
    assert passed.claims[0].values == ["1200"]
    assert passed.reference_values[0].values == ["1200.0"]
    failed = await service.verify_sku(
        request.model_copy(update={"candidate_text": "EC-1 净重 1200 kg"}), ACCESS
    )
    assert not failed.passed and "1.2 kg" in failed.corrections
    assert len(sessions.audit) == 2


async def test_ticket_isolates_identity_and_high_risk_sets_priority():
    tickets = SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(ticket_no="ticket")))
    service = GatewayServices(
        SimpleNamespace(tickets=tickets, precheck=HighRiskPrecheck()),
        Sessions(),
        Settings(_env_file=None),
    )
    body = CreateTicketRequest(
        idempotency_key="same-key", session_id="session", message="饮水机冒烟漏电"
    )
    await service.create_ticket(body, ACCESS)
    first = tickets.create.call_args.kwargs
    await service.create_ticket(body, ACCESS.model_copy(update={"subject_id": "客服乙"}))
    second = tickets.create.call_args.kwargs
    assert first["priority"] == WorkOrderPriority.P0_EMERGENCY
    assert first["idempotency_key"] != second["idempotency_key"]
    assert first["trigger_source"] == "HIGH_RISK_PRECHECK"
