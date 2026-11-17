"""M4 一键验收：真实存储、双协议注册表与可选千问调用；任何断言失败均非零退出。"""

import asyncio
import json
import os
from time import perf_counter
from uuid import uuid4

import httpx
from pydantic import SecretStr
from qdrant_client import models
from sqlalchemy import select

from aegis.config import ProviderMode, Settings
from aegis.db.models import DocumentType, KBDocument, OpsWorkOrder
from aegis.main import create_app
from aegis.schemas.agent import AccessContext


async def post(client, path, body, token):
    response = await client.post(path, json=body, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()


def report(message, **details):
    print("[通过] " + message + (" " + json.dumps(details, ensure_ascii=False) if details else ""))


async def verify_seeds(app):
    """同时检查数据库有效文档及线上别名实际向量，不能只凭本地文件判定入库。"""
    expected = {
        "EC-SKU-SPECS": DocumentType.SKU_SPEC,
        "EC-WAREHOUSE-SOP": DocumentType.WAREHOUSE_SOP,
        "EC-AFTER-SALES-RULES": DocumentType.AFTER_SALES,
        "EC-FINANCE-INTERNAL": DocumentType.FINANCE_RULE,
    }
    async with app.state.session_factory() as session:
        documents = list(
            await session.scalars(
                select(KBDocument).where(
                    KBDocument.doc_code.in_(expected), KBDocument.status == "ACTIVE"
                )
            )
        )
    assert {doc.doc_code: doc.doc_type for doc in documents} == expected, (
        "缺少四大星澜智选电商种子文档，请先运行 uv run python scripts/init_kb.py"
    )
    for code in expected:
        result = await app.state.qdrant.count(
            "kb_children_prod",
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_code", match=models.MatchValue(value=code))]
            ),
            exact=True,
        )
        assert result.count > 0, f"线上别名缺少 {code} 切片"
    report("四类星澜智选电商文档已实际入库，MySQL 与 Qdrant 线上别名一致")


async def run():
    configured = Settings()
    # 临时验收凭据只驻留当前进程，不修改部署认证配置。
    low_token, warehouse_token, finance_token = (uuid4().hex for _ in range(3))
    identities = {
        low_token: AccessContext(
            subject_id="smoke-service", dept_scope=["customer_service"], role_level=1
        ),
        warehouse_token: AccessContext(
            subject_id="smoke-warehouse", dept_scope=["warehouse"], role_level=1
        ),
        finance_token: AccessContext(
            subject_id="smoke-finance", dept_scope=["FINANCE"], role_level=3
        ),
    }
    baseline = configured.model_copy(
        update={
            "llm_provider_mode": ProviderMode.MOCK,
            "gateway_api_tokens": identities,
            "log_level": "WARNING",
        }
    )
    app = create_app(baseline)
    async with app.router.lifespan_context(app):
        await verify_seeds(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            query = {"query": "供应商季度阶梯返利 返点比例 毛利红线", "doc_type": "FINANCE_RULE"}
            denied = await post(client, "/api/v1/kb/search", query, low_token)
            assert denied["hits"] == [], denied
            allowed = await post(client, "/api/v1/kb/search", query, finance_token)
            assert allowed["hits"], "正向财务对照必须有召回，防止空库假通过"
            report("普通客服财务敏感文档召回为 0，越权泄漏率 0%；财务正向对照通过")
            recalled = await post(
                client,
                "/api/v1/kb/search",
                {
                    "query": "EC-ih200-Std EC-ih200-Max 净重 SKU 参数表",
                    "doc_type": "SKU_SPEC",
                    "top_k": 30,
                },
                warehouse_token,
            )
            parent = next(
                hit for hit in recalled["hits"] if "| EC-ih200-Std |" in hit["content_text"]
            )
            for sku, weight in (("EC-ih200-Std", "1800g"), ("EC-ih200-Max", "2050g")):
                checked = await post(
                    client,
                    "/api/v1/guardrails/verify-sku",
                    {
                        "candidate_text": f"{sku} 净重 {weight}。",
                        "sku": sku,
                        "parent_id": parent["parent_id"],
                    },
                    warehouse_token,
                )
                assert checked["passed"] and checked["checked_count"] == 1, checked
            bad = await post(
                client,
                "/api/v1/guardrails/verify-sku",
                {
                    "candidate_text": "EC-ih200-Std 净重2050g。",
                    "sku": "EC-ih200-Std",
                    "parent_id": parent["parent_id"],
                },
                warehouse_token,
            )
            assert not bad["passed"] and bad["corrections"]
            answer = await post(
                client,
                "/api/v1/agent/chat",
                {
                    "session_id": uuid4().hex,
                    "request_id": uuid4().hex,
                    "message": "EC-ih200-Std 的净重是多少？请仅回答这个 SKU 的净重。",
                },
                warehouse_token,
            )
            assert answer["guardrail_action"] in {"SKU_VERIFIED_PASS", "SKU_REGENERATED_PASS"}, (
                answer
            )
            verified_answer = await post(
                client,
                "/api/v1/guardrails/verify-sku",
                {
                    "candidate_text": answer["answer"],
                    "sku": "EC-ih200-Std",
                    "parent_id": parent["parent_id"],
                },
                warehouse_token,
            )
            assert verified_answer["passed"] and verified_answer["checked_count"] > 0, answer
            report("同系列易混 SKU 问答、单位换算和错误参数拦截通过", answer=answer["answer"])
            # 红线分支应跳过模型与检索；不把真实数据库耗时宣称为零毫秒。
            before = len(app.state.workflow.llm.calls)
            body = {
                "session_id": uuid4().hex,
                "request_id": uuid4().hex,
                "message": "饮水机冒烟漏电",
            }
            started = perf_counter()
            urgent = await post(client, "/api/v1/agent/chat", body, warehouse_token)
            elapsed = (perf_counter() - started) * 1000
            duplicate = await post(client, "/api/v1/agent/chat", body, warehouse_token)
            assert urgent["guardrail_action"] == "PRECHECK_SHORT_CIRCUIT"
            assert urgent["steps"] == 1 and len(app.state.workflow.llm.calls) == before
            assert urgent["ticket_no"] == duplicate["ticket_no"]
            async with app.state.session_factory() as session:
                ticket = await session.scalar(
                    select(OpsWorkOrder).where(OpsWorkOrder.ticket_no == urgent["ticket_no"])
                )
                assert ticket.priority.value == "P0_EMERGENCY"
            report(
                "高危词单步短路，模型调用 0 次，P0 工单重试防重通过", latency_ms=round(elapsed, 2)
            )
            ticket_body = {
                "session_id": uuid4().hex,
                "idempotency_key": uuid4().hex,
                "message": "请核对发货波次",
            }
            first = await post(client, "/api/v1/tickets", ticket_body, warehouse_token)
            second = await post(client, "/api/v1/tickets", ticket_body, warehouse_token)
            assert first["ticket_no"] == second["ticket_no"]
            assert await app.state.redis.xlen("stream:ticket:events") > 0
            document = (await client.get("/openapi.json")).json()
            assert document["openapi"] == "3.1.0"
            assert {
                "/api/v1/kb/search",
                "/api/v1/guardrails/verify-sku",
                "/api/v1/tickets",
                "/api/v1/agent/chat",
            } <= document["paths"].keys()
            tools = await app.state.mcp_server.list_tools()
            assert {tool.name for tool in tools} == {
                "search_enterprise_kb",
                "verify_sku_specs",
                "create_ops_ticket",
            }
            report("OpenAPI 3.1.0、四个核心接口、MCP 三工具注册表与工单事件流通过")
    key = configured.llm_api_key or configured.provider_api_key
    if not key and os.getenv("DASHSCOPE_API_KEY"):
        key = SecretStr(os.environ["DASHSCOPE_API_KEY"])
    if key:
        real = configured.model_copy(
            update={
                "llm_api_key": key,
                "llm_provider_mode": (
                    configured.llm_provider_mode
                    if configured.llm_provider_mode != ProviderMode.MOCK
                    else ProviderMode.DASHSCOPE
                ),
                "gateway_api_tokens": identities,
                "log_level": "WARNING",
            }
        )
        app = create_app(real)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                answer = await post(
                    client,
                    "/api/v1/agent/chat",
                    {
                        "session_id": uuid4().hex,
                        "request_id": uuid4().hex,
                        "message": "EC-ih200-Std 的净重是多少？"
                        "请只陈述这个 SKU 的净重，使用克作单位。",
                    },
                    warehouse_token,
                )
                assert answer["guardrail_action"] in {
                    "SKU_VERIFIED_PASS",
                    "SKU_REGENERATED_PASS",
                }, answer
                assert answer["input_tokens"] > 0 and answer["output_tokens"] > 0, answer
                verified_answer = await post(
                    client,
                    "/api/v1/guardrails/verify-sku",
                    {
                        "candidate_text": answer["answer"],
                        "sku": "EC-ih200-Std",
                        "parent_id": parent["parent_id"],
                    },
                    warehouse_token,
                )
                assert verified_answer["passed"] and verified_answer["checked_count"] > 0, answer
                report(
                    "真实千问端到端生成与后置核验通过",
                    model=app.state.workflow.llm.model,
                    answer=answer["answer"],
                    input_tokens=answer["input_tokens"],
                    output_tokens=answer["output_tokens"],
                )
    else:
        print(
            "[说明] 未检测到 LLM_API_KEY、PROVIDER_API_KEY 或 DASHSCOPE_API_KEY，"
            "真实千问分支未执行。"
        )
    report("M4 必需冒烟检查全部完成")


if __name__ == "__main__":
    asyncio.run(run())
