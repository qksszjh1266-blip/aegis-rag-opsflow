"""显式启用的真实存储验证，仅清理本测试写入的数据。"""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.exc import IntegrityError

from aegis.config import Settings
from aegis.db.base import Base
from aegis.db.models import (
    BadcaseCategory,
    DocumentType,
    KBDocument,
    KBParentChunk,
    OpsWorkOrder,
    RAGAuditLog,
    WorkOrderPriority,
)
from aegis.db.session import create_mysql_engine, create_session_factory
from alembic import command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AEGIS_INTEGRATION") != "1",
        reason="真实存储测试需启动容器并设置 AEGIS_INTEGRATION=1",
    ),
]


def upgrade_twice() -> None:
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    command.upgrade(config, "head")
    command.upgrade(config, "head")


@pytest.fixture
async def migrated_engine():
    # Alembic 自行创建事件循环，放到线程以避开测试的异步循环。
    await asyncio.to_thread(upgrade_twice)
    engine = create_mysql_engine(Settings())
    try:
        yield engine
    finally:
        await engine.dispose()


async def test_migration_is_idempotent_and_matches_models(migrated_engine):
    async with migrated_engine.connect() as connection:
        assert await connection.scalar(text("SELECT version_num FROM alembic_version")) == "0002"
        tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        assert set(Base.metadata.tables) <= set(tables)
        differences = await connection.run_sync(
            lambda sync: compare_metadata(MigrationContext.configure(sync), Base.metadata)
        )
        assert differences == []


async def test_four_tables_crud_and_idempotency(migrated_engine):
    suffix = uuid4().hex
    session_factory = create_session_factory(migrated_engine)
    doc_code = f"test-doc-{suffix}"
    trace_id = suffix
    sku_rows = {"SKU-001": {"重量": {"数值": 12.5, "单位": "kg"}, "颜色": "蓝色"}}
    try:
        async with session_factory.begin() as session:
            doc = KBDocument(
                doc_code=doc_code,
                title="单元货架规格",
                doc_type=DocumentType.SKU_SPEC,
                dept_scope="warehouse",
                sha256_hash="a" * 64,
                index_version="v1",
            )
            session.add(doc)
            await session.flush()
            chunk = KBParentChunk(
                parent_id=f"parent-{suffix}",
                doc_id=doc.id,
                header_path="商品>货架>参数",
                content_text="货架额定承重与包装信息。" * 100,
                is_sku_table=True,
                sku_rows_json=sku_rows,
            )
            order = OpsWorkOrder(
                ticket_no=f"ticket-{suffix}",
                idempotency_key=f"idem-{suffix}",
                session_id=suffix,
                trace_id=trace_id,
                priority=WorkOrderPriority.P2_NORMAL,
                trigger_source="integration_test",
                slots_payload={"商品": "SKU-001"},
            )
            audit = RAGAuditLog(
                trace_id=trace_id,
                session_id=suffix,
                query_text="货架重量是多少？",
                recalled_parent_ids=[chunk.parent_id],
                guardrail_action="ALLOW",
                latency_ms=12,
            )
            session.add_all([chunk, order, audit])
            await session.flush()
            ids = doc.id, chunk.id, order.id, audit.id
            assert doc.status == "ACTIVE"
            assert doc.min_role_level == 1
            assert order.status == "PENDING"
            assert order.completeness_score == 0.0
            assert audit.badcase_category == BadcaseCategory.NONE
            assert audit.degraded_flag is False
            assert all(item.created_at is not None for item in (doc, chunk, order, audit))

        async with session_factory.begin() as session:
            doc = await session.get(KBDocument, ids[0])
            chunk = await session.get(KBParentChunk, ids[1])
            order = await session.get(OpsWorkOrder, ids[2])
            audit = await session.get(RAGAuditLog, ids[3])
            assert chunk.sku_rows_json == sku_rows
            assert order.slots_payload == {"商品": "SKU-001"}
            assert audit.recalled_parent_ids == [chunk.parent_id]
            records = (doc, chunk, order, audit)
            previous_timestamps = tuple(item.updated_at for item in records)
            doc.title = "更新后的货架规格"
            chunk.sku_rows_json = {"SKU-001": {"颜色": "白色"}}
            order.status = "COMPLETED"
            audit.degraded_flag = True
            await session.flush()
            # 同秒更新也应保留微秒精度，且可直接读取数据库生成的时间戳。
            for item, previous_timestamp in zip(records, previous_timestamps, strict=True):
                assert item.updated_at >= item.created_at
                assert item.updated_at >= previous_timestamp

        async with session_factory() as session:
            assert (await session.get(KBDocument, ids[0])).title == "更新后的货架规格"
            assert (await session.get(KBParentChunk, ids[1])).sku_rows_json["SKU-001"][
                "颜色"
            ] == "白色"
            assert (await session.get(OpsWorkOrder, ids[2])).status == "COMPLETED"
            assert (await session.get(RAGAuditLog, ids[3])).degraded_flag is True
            duplicate = OpsWorkOrder(
                ticket_no=f"duplicate-{suffix}",
                idempotency_key=f"idem-{suffix}",
                session_id=suffix,
                trace_id=trace_id,
                priority=WorkOrderPriority.P2_NORMAL,
                trigger_source="integration_test",
                slots_payload={},
            )
            session.add(duplicate)
            with pytest.raises(IntegrityError):
                await session.flush()
            await session.rollback()
    finally:
        async with session_factory.begin() as session:
            await session.execute(delete(RAGAuditLog).where(RAGAuditLog.trace_id == trace_id))
            await session.execute(delete(OpsWorkOrder).where(OpsWorkOrder.trace_id == trace_id))
            await session.execute(
                delete(KBParentChunk).where(KBParentChunk.parent_id == f"parent-{suffix}")
            )
            await session.execute(delete(KBDocument).where(KBDocument.doc_code == doc_code))

    async with session_factory() as session:
        assert (
            await session.scalar(select(KBDocument.id).where(KBDocument.doc_code == doc_code))
            is None
        )
        assert await session.get(KBParentChunk, ids[1]) is None
        assert await session.get(OpsWorkOrder, ids[2]) is None
        assert await session.get(RAGAuditLog, ids[3]) is None
