"""面试四幕脚本的真实 CLI、持久化证据与失败退出码回归。"""

import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

import pytest
from alembic.config import Config
from sqlalchemy import delete, select

from aegis.config import Settings
from aegis.db.models import BadcaseCategory, OpsTicketOutbox, OpsWorkOrder, RAGAuditLog
from aegis.db.redis_client import create_redis_client
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.workers.tickets import TicketResult, publication_key
from alembic import command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("AEGIS_INTEGRATION") != "1", reason="需要真实三类基础服务"),
]
ROOT = Path(__file__).resolve().parents[2]


async def invoke_demo(environment, output):
    started = perf_counter()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(ROOT / "scripts" / "demo_interview_chaos.py"),
        "--json-output",
        str(output),
        cwd=ROOT,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    return process.returncode, stdout.decode(), stderr.decode(), perf_counter() - started


async def clear_demo(sessions, redis, run_id):
    """只删本次演示证据，保留共享种子、业务流和其他会话。"""
    prefix = f"chaos:{run_id}:"
    async with sessions.begin() as session:
        orders = list(
            await session.scalars(
                select(OpsWorkOrder).where(OpsWorkOrder.session_id.startswith(prefix))
            )
        )
        for order in orders:
            event = await session.scalar(
                select(OpsTicketOutbox).where(OpsTicketOutbox.ticket_id == order.id)
            )
            if event is not None:
                await redis.delete(publication_key(event.event_id))
                await session.delete(event)
                await session.flush()
            await redis.delete(f"idem:ticket:{order.idempotency_key}")
            await session.delete(order)
        await session.execute(delete(RAGAuditLog).where(RAGAuditLog.session_id.startswith(prefix)))
    keys = [
        key
        for act in ("rerank", "sku-pass", "sku-block", "p0", "acl")
        for key in (f"fsm:session:{prefix}{act}", f"lock:session:{prefix}{act}")
    ]
    await redis.delete(*keys, f"demo:chaos:{run_id}:tickets")


@pytest.fixture
async def seeded_environment(monkeypatch):
    for variable in (
        "PROVIDER_MODE",
        "LLM_PROVIDER_MODE",
        "EMBEDDING_PROVIDER_MODE",
        "RERANK_PROVIDER_MODE",
    ):
        monkeypatch.setenv(variable, "mock")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")
    await asyncio.to_thread(command.upgrade, Config(str(ROOT / "alembic.ini")), "head")
    spec = importlib.util.spec_from_file_location(
        "chaos_seed_loader", ROOT / "scripts" / "init_kb.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    await module.initialize(ROOT / "data" / "seed_docs")
    return dict(os.environ)


async def test_four_acts_cli_and_real_mysql_redis_evidence(seeded_environment, tmp_path):
    report_file = tmp_path / "chaos.json"
    # 即使宿主配置真实供应商且没有 Key，演示也必须只使用显式注入的离线模型。
    environment = {
        **seeded_environment,
        "PROVIDER_MODE": "dashscope",
        "LLM_PROVIDER_MODE": "dashscope",
    }
    environment.pop("PROVIDER_API_KEY", None)
    environment.pop("LLM_API_KEY", None)
    code, stdout, stderr, elapsed = await invoke_demo(environment, report_file)
    settings = Settings()
    engine = create_mysql_engine(settings)
    sessions = create_session_factory(engine)
    redis = create_redis_client(settings)
    run_id = re.search(r"run_id=([a-f0-9]{32})", stdout)
    try:
        assert code == 0, stdout + stderr
        assert elapsed < 15
        report = json.loads(report_file.read_text(encoding="utf-8"))
        assert report["elapsed_ms"] < 15000
        assert all(
            label in stdout
            for label in (
                "第一幕",
                "第二幕",
                "第三幕",
                "第四幕",
                "【输入请求】",
                "【注入故障】",
                "【系统实时防线动作】",
                "【MySQL/Redis 落库证据】",
                "【端到端耗时】",
                "四幕全部通过",
            )
        )
        rerank, sku, emergency, acl = (
            report["acts"][key] for key in ("rerank", "sku", "emergency", "acl")
        )
        assert rerank["cancelled"] and 550 <= rerank["rerank_ms"] < 850
        assert rerank["hit_count"] > 0 and rerank["degraded_flag"]
        assert rerank["degraded_stage"] == "rerank_fallback"
        assert {"EC-WAREHOUSE-SOP", "EC-AFTER-SALES-RULES"} <= set(rerank["raw_doc_codes"])
        assert sku["pass_action"] == "SKU_VERIFIED_PASS"
        assert sku["block_action"] == "SKU_HALLUCINATION_BLOCKED"
        assert sku["generation_calls"] == 2 and "2.1 kW" in sku["feedback"]
        assert "2.2 L" in sku["feedback"]
        assert "9999" not in sku["answer"] and sku["priority"] == "P1_HIGH"
        assert emergency["model_calls"] == 0 and emergency["steps"] == 1
        assert emergency["ticket_nos"] == [emergency["ticket_no"]] * 3
        assert emergency["db_fallback_same_ticket"]
        assert 86380 <= emergency["ttl"] <= 86400
        cached = TicketResult.model_validate_json(await redis.get(emergency["redis_key"]))
        assert cached.ticket_no == emergency["ticket_no"]
        assert cached.idempotency_key == emergency["idempotency_key"]
        assert acl["authorized_hits"] > 0
        assert acl["unauthorized_hits"] == acl["unauthorized_raw"] == 0
        assert acl["leakage_terms"] == [] and acl["action"] == "NO_EVIDENCE"
        async with sessions() as session:
            audit = await session.get(RAGAuditLog, rerank["audit_id"])
            assert audit.trace_id == rerank["trace_id"] and audit.degraded_flag
            assert audit.degraded_stage == "rerank_fallback"
            blocked = await session.get(RAGAuditLog, sku["block_audit_id"])
            assert blocked.badcase_category == BadcaseCategory.HALLUCINATION
            assert blocked.guardrail_action == "SKU_HALLUCINATION_BLOCKED"
            passed = await session.get(RAGAuditLog, sku["pass_audit_id"])
            assert passed.guardrail_action == "SKU_VERIFIED_PASS"
            orders = list(
                await session.scalars(
                    select(OpsWorkOrder).where(OpsWorkOrder.session_id.in_(report["session_ids"]))
                )
            )
            assert len(orders) == 2
            assert {order.priority.value for order in orders} == {"P0_EMERGENCY", "P1_HIGH"}
            p0_audits = list(
                await session.scalars(
                    select(RAGAuditLog).where(
                        RAGAuditLog.session_id == cached.session_id,
                        RAGAuditLog.guardrail_action == "PRECHECK_SHORT_CIRCUIT",
                    )
                )
            )
            assert len(p0_audits) == 3
            events = list(
                await session.scalars(
                    select(OpsTicketOutbox).where(
                        OpsTicketOutbox.ticket_id.in_([order.id for order in orders])
                    )
                )
            )
            assert len(events) == 2 and all(event.published for event in events)
        assert await redis.xlen(report["stream"]) == 2
    finally:
        if run_id:
            await clear_demo(sessions, redis, run_id.group(1))
        await redis.aclose()
        await engine.dispose()


@pytest.mark.parametrize(
    "overrides",
    [
        {"EMBEDDING_DIMENSION": "32"},
        {"REDIS_URL": "redis://127.0.0.1:1/0", "DEPENDENCY_TIMEOUT_SECONDS": "0.2"},
    ],
)
async def test_cli_rejects_wrong_vectors_or_unavailable_storage(
    seeded_environment, tmp_path, overrides
):
    output = tmp_path / "must-not-exist.json"
    code, stdout, _, _ = await invoke_demo({**seeded_environment, **overrides}, output)
    assert code != 0 and "【演示失败】" in stdout
    assert "四幕全部通过" not in stdout
    assert not output.exists()
