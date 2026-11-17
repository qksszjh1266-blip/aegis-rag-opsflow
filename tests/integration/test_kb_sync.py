"""真实 MySQL 和 Qdrant 验证快照发布及故障时的线上读取隔离。"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from alembic.config import Config
from qdrant_client import models
from sqlalchemy import delete, select

from aegis.config import Settings
from aegis.db.models import KBDocument, KBParentChunk
from aegis.db.qdrant_client import create_qdrant_client
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.llm.providers import MockEmbeddingProvider
from aegis.rag.parser import DocumentMetadata, MarkdownParser
from aegis.rag.syncer import publish_snapshot
from alembic import command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("AEGIS_INTEGRATION") != "1",
        reason="真实快照测试需启动 MySQL、Qdrant 并设置 AEGIS_INTEGRATION=1",
    ),
]

_SOURCE = (
    "# 恒温器规格\n## 商品参数\n"
    "| SKU | 净重 | 功率 | 温度上限 |\n| --- | --- | --- | --- |\n"
    "| HEAT-01 | 2.5 kg | 800 W | 80 ℃ |\n"
    "| HEAT-01-PRO | 3.0 kg | 1200 W | 95 ℃ |\n"
)
_ALEMBIC_CONFIG = str(Path(__file__).resolve().parents[2] / "alembic.ini")


@pytest.fixture
async def storage():
    config = Config(_ALEMBIC_CONFIG)
    await asyncio.to_thread(command.upgrade, config, "head")
    settings = Settings()
    engine = create_mysql_engine(settings)
    client = create_qdrant_client(settings)
    provider = MockEmbeddingProvider(32)
    suffix = uuid4().hex
    metadata = DocumentMetadata(
        doc_code=f"sync-test-{suffix}",
        doc_type="SKU_SPEC",
        dept_scope="warehouse",
        min_role_level=1,
    )
    alias = f"sync-test-{suffix}"
    with patch.object(client, "create_collection", wraps=client.create_collection) as created:
        try:
            yield (
                {
                    "engine": engine,
                    "client": client,
                    "embedding_provider": provider,
                    "embedding_fingerprint": "mock-test-v1",
                    "alias_name": alias,
                },
                metadata,
            )
        finally:
            aliases = await client.get_aliases()
            if any(item.alias_name == alias for item in aliases.aliases):
                await client.update_collection_aliases(
                    change_aliases_operations=[
                        models.DeleteAliasOperation(
                            delete_alias=models.DeleteAlias(alias_name=alias)
                        )
                    ]
                )
            for name in {call.kwargs["collection_name"] for call in created.call_args_list}:
                if await client.collection_exists(name):
                    await client.delete_collection(name)
            factory = create_session_factory(engine)
            async with factory.begin() as session:
                doc_id = await session.scalar(
                    select(KBDocument.id).where(KBDocument.doc_code == metadata.doc_code)
                )
                if doc_id is not None:
                    await session.execute(
                        delete(KBParentChunk).where(KBParentChunk.doc_id == doc_id)
                    )
                    await session.execute(delete(KBDocument).where(KBDocument.id == doc_id))
            await provider.aclose()
            await client.close()
            await engine.dispose()


async def alias_target(client, alias_name):
    return next(
        item.collection_name
        for item in (await client.get_aliases()).aliases
        if item.alias_name == alias_name
    )


async def test_sku_rows_persist_in_mysql_and_identical_snapshot_skips_embedding(storage):
    kwargs, metadata = storage
    document = MarkdownParser().parse(_SOURCE, metadata)
    first = await publish_snapshot([document], **kwargs)
    assert not first.skipped
    factory = create_session_factory(kwargs["engine"])
    async with factory() as session:
        stored = await session.scalar(
            select(KBDocument).where(KBDocument.doc_code == metadata.doc_code)
        )
        parents = (
            await session.scalars(select(KBParentChunk).where(KBParentChunk.doc_id == stored.id))
        ).all()
        assert stored.sha256_hash == document.sha256_hash
        assert stored.index_version == first.collection_name
        rows = [
            row for parent in parents if parent.is_sku_table for row in parent.sku_rows_json["rows"]
        ]
        assert rows == [
            {"SKU": "HEAT-01", "净重": "2.5 kg", "功率": "800 W", "温度上限": "80 ℃"},
            {"SKU": "HEAT-01-PRO", "净重": "3.0 kg", "功率": "1200 W", "温度上限": "95 ℃"},
        ]
    with patch.object(kwargs["embedding_provider"], "embed", new_callable=AsyncMock) as embed:
        second = await publish_snapshot([document], **kwargs)
        embed.assert_not_awaited()
    assert second.skipped
    assert second.collection_name == first.collection_name


async def test_changed_content_publishes_new_points_and_retains_previous_snapshot(storage):
    kwargs, metadata = storage
    original = MarkdownParser().parse(_SOURCE, metadata)
    first = await publish_snapshot([original], **kwargs)
    changed = MarkdownParser().parse(_SOURCE.replace("800 W", "900 W"), metadata)
    second = await publish_snapshot([changed], **kwargs)
    assert second.collection_name != first.collection_name
    points, _ = await kwargs["client"].scroll(kwargs["alias_name"], limit=100)
    assert {point.id for point in points} == {child.child_id for child in changed.children}
    assert {point.id for point in points}.isdisjoint(child.child_id for child in original.children)
    assert all(point.payload["sha256_hash"] == changed.sha256_hash for point in points)
    assert await kwargs["client"].collection_exists(first.collection_name)
    factory = create_session_factory(kwargs["engine"])
    async with factory() as session:
        old = await session.scalar(
            select(KBParentChunk).where(KBParentChunk.parent_id == original.parents[0].parent_id)
        )
        assert old is not None


async def test_acl_change_without_content_change_rebuilds_and_revokes_old_department(storage):
    kwargs, metadata = storage
    original = MarkdownParser().parse(_SOURCE, metadata)
    first = await publish_snapshot([original], **kwargs)
    changed = MarkdownParser().parse(
        _SOURCE, metadata.model_copy(update={"dept_scope": "finance", "min_role_level": 3})
    )
    assert changed.sha256_hash == original.sha256_hash
    second = await publish_snapshot([changed], **kwargs)
    assert second.collection_name != first.collection_name
    old_access = models.Filter(
        must=[
            models.FieldCondition(key="dept_scope", match=models.MatchValue(value="warehouse")),
            models.FieldCondition(key="min_role_level", range=models.Range(lte=1)),
        ]
    )
    count = await kwargs["client"].count(kwargs["alias_name"], count_filter=old_access, exact=True)
    assert count.count == 0
    points, _ = await kwargs["client"].scroll(kwargs["alias_name"], limit=100)
    assert points and all(point.payload["dept_scope"] == "finance" for point in points)
    assert all(point.payload["min_role_level"] == 3 for point in points)


async def test_embedding_failure_keeps_old_alias_and_mysql_version(storage):
    kwargs, metadata = storage
    original = MarkdownParser().parse(_SOURCE, metadata)
    first = await publish_snapshot([original], **kwargs)
    changed = MarkdownParser().parse(_SOURCE.replace("800 W", "900 W"), metadata)
    with patch.object(
        kwargs["embedding_provider"], "embed", side_effect=RuntimeError("模拟模型断连")
    ):
        with pytest.raises(RuntimeError, match="模拟模型断连"):
            await publish_snapshot([changed], **kwargs)
    assert await alias_target(kwargs["client"], kwargs["alias_name"]) == first.collection_name
    factory = create_session_factory(kwargs["engine"])
    async with factory() as session:
        stored = await session.scalar(
            select(KBDocument).where(KBDocument.doc_code == metadata.doc_code)
        )
        assert stored.sha256_hash == original.sha256_hash
        assert stored.index_version == first.collection_name
    recovered = await publish_snapshot([changed], **kwargs)
    assert await alias_target(kwargs["client"], kwargs["alias_name"]) == recovered.collection_name


async def test_alias_publish_failure_preserves_reads_and_retry_repairs_publication(storage):
    kwargs, metadata = storage
    original = MarkdownParser().parse(_SOURCE, metadata)
    first = await publish_snapshot([original], **kwargs)
    changed = MarkdownParser().parse(_SOURCE.replace("800 W", "900 W"), metadata)
    with patch.object(
        kwargs["client"], "update_collection_aliases", side_effect=RuntimeError("模拟发布断连")
    ):
        with pytest.raises(RuntimeError, match="模拟发布断连"):
            await publish_snapshot([changed], **kwargs)
    assert await alias_target(kwargs["client"], kwargs["alias_name"]) == first.collection_name
    points, _ = await kwargs["client"].scroll(kwargs["alias_name"], limit=100)
    assert {point.id for point in points} == {child.child_id for child in original.children}
    recovered = await publish_snapshot([changed], **kwargs)
    assert await alias_target(kwargs["client"], kwargs["alias_name"]) == recovered.collection_name
