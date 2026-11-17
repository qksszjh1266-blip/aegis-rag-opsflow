"""命名向量、ACL 索引及集合别名的发布管理。"""

import math

from qdrant_client import AsyncQdrantClient, models

from aegis.llm.providers import EmbeddingProvider
from aegis.rag.parser import ParsedDocument
from aegis.rag.sparse import encode_document


class QdrantIndexer:
    """只向指定物理集合写入，完整构建后由调用方显式发布线上别名。"""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "kb_children_v1",
        alias_name: str = "kb_children_prod",
        *,
        batch_size: int = 32,
    ) -> None:
        if not 1 <= batch_size <= 2048:
            raise ValueError("向量写入批次必须介于 1 和 2048")
        if not collection_name or collection_name == alias_name:
            raise ValueError("物理集合名称不能为空或等于线上别名")
        self.client = client
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.alias_name = alias_name
        self.batch_size = batch_size

    async def ensure_collection(self) -> None:
        """幂等建库并校验维度；维度或稀疏算法不兼容时要求重建。"""
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.embedding_provider.dimension, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        info = await self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        if (
            not isinstance(vectors, dict)
            or "dense" not in vectors
            or vectors["dense"].size != self.embedding_provider.dimension
            or vectors["dense"].distance != models.Distance.COSINE
            or "bm25_sparse" not in sparse
            or sparse["bm25_sparse"].modifier != models.Modifier.IDF
        ):
            raise ValueError("现有集合的命名向量配置不兼容，请使用新的物理集合")
        for field, schema in (
            ("dept_scope", models.PayloadSchemaType.KEYWORD),
            ("min_role_level", models.PayloadSchemaType.INTEGER),
            ("parent_id", models.PayloadSchemaType.KEYWORD),
            ("doc_code", models.PayloadSchemaType.KEYWORD),
        ):
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
                wait=True,
            )

    async def switch_alias(self) -> None:
        """在一次 Qdrant 原子操作中删除旧映射并创建新映射；不删除旧集合。"""
        await self.ensure_collection()
        aliases = await self.client.get_aliases()
        current = next((a for a in aliases.aliases if a.alias_name == self.alias_name), None)
        if current and current.collection_name == self.collection_name:
            return
        operations = []
        if current:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.alias_name)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    alias_name=self.alias_name, collection_name=self.collection_name
                )
            )
        )
        await self.client.update_collection_aliases(change_aliases_operations=operations)

    async def index_document(self, document: ParsedDocument) -> int:
        """批量生成双向量并写入不可变父上下文；重复写入同一版本不会生成重复点。"""
        parents = {parent.parent_id: parent for parent in document.parents}
        metadata = document.metadata
        for start in range(0, len(document.children), self.batch_size):
            children = document.children[start : start + self.batch_size]
            vectors = await self.embedding_provider.embed(
                [child.content_text for child in children]
            )
            if len(vectors) != len(children) or any(
                len(vector) != self.embedding_provider.dimension
                or not all(math.isfinite(value) for value in vector)
                for vector in vectors
            ):
                raise ValueError("嵌入结果数量、维度或数值无效，拒绝写入")
            points = []
            for child, vector in zip(children, vectors, strict=True):
                parent = parents[child.parent_id]
                points.append(
                    models.PointStruct(
                        id=child.child_id,
                        vector={
                            "dense": vector,
                            "bm25_sparse": encode_document(child.content_text),
                        },
                        payload={
                            "dept_scope": metadata.dept_scope,
                            "min_role_level": metadata.min_role_level,
                            "parent_id": child.parent_id,
                            "doc_code": metadata.doc_code,
                            "doc_type": metadata.doc_type.value,
                            "sha256_hash": document.sha256_hash,
                            "header_path": child.header_path,
                            "content_text": child.content_text,
                            "parent_content": parent.content_text,
                            "sku_rows_json": child.sku_rows_json,
                            "parent_sku_rows_json": parent.sku_rows_json,
                            "is_sku_table": child.is_sku_table,
                        },
                    )
                )
            await self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(document.children)
