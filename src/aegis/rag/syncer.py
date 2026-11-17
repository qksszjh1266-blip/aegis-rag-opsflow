"""MySQL 父块落库与完整 Qdrant 快照发布，保留旧版本以支持在途查询。"""

import hashlib
import json
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from aegis.db.models import KBDocument, KBParentChunk
from aegis.llm.providers import EmbeddingProvider
from aegis.rag.indexer import QdrantIndexer
from aegis.rag.parser import ParsedDocument
from aegis.rag.sparse import SPARSE_VERSION


@dataclass(frozen=True)
class SyncResult:
    collection_name: str
    document_count: int
    child_count: int
    skipped: bool


async def publish_snapshot(
    documents: list[ParsedDocument],
    *,
    engine: AsyncEngine,
    client: AsyncQdrantClient,
    embedding_provider: EmbeddingProvider,
    embedding_fingerprint: str,
    alias_name: str = "kb_children_prod",
    batch_size: int = 32,
) -> SyncResult:
    """发布输入清单的完整快照；调用方必须提供该别名下的全部文档。

    MySQL 会话级锁覆盖构建与发布，进程中断时连接释放即解锁。跨库没有分布式
    事务：先提交父块再切别名，失败时线上仍可读旧快照，重跑能修复未发布状态。
    旧父块和旧集合保留，不在构建流程中执行可能影响在途请求的垃圾回收。
    """
    if not documents or len({doc.metadata.doc_code for doc in documents}) != len(documents):
        raise ValueError("快照文档不能为空且 doc_code 不得重复")
    signature = json.dumps(
        {
            "documents": sorted(
                (doc.metadata.doc_code, doc.sha256_hash, [c.child_id for c in doc.children])
                for doc in documents
            ),
            "embedding": embedding_fingerprint,
            "dimension": embedding_provider.dimension,
            "sparse": SPARSE_VERSION,
            "payload_version": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    # 名称长度不超过 kb_documents.index_version 的 32 字符限制。
    collection = "kb_children_" + hashlib.sha256(signature.encode()).hexdigest()[:20]
    indexer = QdrantIndexer(
        client, embedding_provider, collection, alias_name, batch_size=batch_size
    )
    expected = sum(len(doc.children) for doc in documents)
    lock_key = "kb_publish:" + hashlib.sha256(alias_name.encode()).hexdigest()[:40]
    async with engine.connect() as connection:
        acquired = await connection.scalar(text("SELECT GET_LOCK(:key, 0)"), {"key": lock_key})
        await connection.commit()
        if acquired != 1:
            raise RuntimeError("已有知识库发布任务执行中，请稍后重试")
        try:
            await indexer.ensure_collection()
            aliases = await client.get_aliases()
            current = next((a for a in aliases.aliases if a.alias_name == alias_name), None)
            count = await client.count(collection, exact=True)
            skipped = bool(
                current and current.collection_name == collection and count.count == expected
            )
            if current and current.collection_name == collection and not skipped:
                # 损坏的线上快照不能原地修复；另建物理集合保证读隔离。
                from uuid import uuid4

                collection = "kb_children_" + uuid4().hex[:20]
                indexer.collection_name = collection
                await indexer.ensure_collection()
            if not skipped:
                for doc in documents:
                    await indexer.index_document(doc)
                if (await client.count(collection, exact=True)).count != expected:
                    raise RuntimeError("快照子块数量不一致，拒绝发布")
            async with AsyncSession(bind=connection, expire_on_commit=False) as session:
                async with session.begin():
                    for doc in documents:
                        stored = await session.scalar(
                            select(KBDocument).where(KBDocument.doc_code == doc.metadata.doc_code)
                        )
                        values = {
                            "title": doc.title,
                            "doc_type": doc.metadata.doc_type,
                            "dept_scope": doc.metadata.dept_scope,
                            "min_role_level": doc.metadata.min_role_level,
                            "sha256_hash": doc.sha256_hash,
                            "index_version": collection,
                            "status": "ACTIVE",
                        }
                        if stored is None:
                            stored = KBDocument(doc_code=doc.metadata.doc_code, **values)
                            session.add(stored)
                        else:
                            for key, value in values.items():
                                setattr(stored, key, value)
                        await session.flush()
                        existing = set(
                            await session.scalars(
                                select(KBParentChunk.parent_id).where(
                                    KBParentChunk.doc_id == stored.id
                                )
                            )
                        )
                        for parent in doc.parents:
                            if parent.parent_id not in existing:
                                session.add(
                                    KBParentChunk(
                                        parent_id=parent.parent_id,
                                        doc_id=stored.id,
                                        header_path=parent.header_path,
                                        content_text=parent.content_text,
                                        is_sku_table=parent.is_sku_table,
                                        sku_rows_json=parent.sku_rows_json,
                                    )
                                )
            await indexer.switch_alias()
            return SyncResult(collection, len(documents), expected, skipped)
        finally:
            if connection.in_transaction():
                await connection.rollback()
            await connection.execute(text("SELECT RELEASE_LOCK(:key)"), {"key": lock_key})
            await connection.commit()
