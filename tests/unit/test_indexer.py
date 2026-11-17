"""索引幂等、原子别名与 BM25 编码的独立契约。"""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client import AsyncQdrantClient, models

from aegis.llm.providers import MockEmbeddingProvider
from aegis.rag.indexer import QdrantIndexer
from aegis.rag.parser import DocumentMetadata, MarkdownParser
from aegis.rag.sparse import encode_document, encode_query, tokenize

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")


@pytest.fixture
async def client():
    client = AsyncQdrantClient(":memory:")
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def document():
    return MarkdownParser().parse(
        "# SKU 参数\n## 恒温系列\n| SKU | 净重 | 功率 |\n| --- | --- | --- |\n"
        "| HEAT-01 | 2.5 kg | 800 W |\n| HEAT-01-PRO | 3 kg | 1200 W |\n",
        DocumentMetadata(
            doc_code="sku-indexer-test",
            doc_type="SKU_SPEC",
            dept_scope="warehouse",
            min_role_level=2,
        ),
    )


def test_sparse_tokens_preserve_identifiers_units_and_chinese_bigrams():
    tokens = tokenize("HEAT-01-PRO 净重 2.5 kg，功率 1200 W，温度 80 ℃")
    assert {"heat-01-pro", "净重", "功率", "温度", "2.5", "kg", "w", "℃"} <= set(tokens)
    query = encode_query("HEAT-01-PRO HEAT-01-PRO")
    assert query.values == [1.0]
    assert query.indices == [int.from_bytes(hashlib.sha256(b"heat-01-pro").digest()[:4], "big")]
    assert encode_query("！").indices == []
    assert encode_document("！").values == []


def test_bm25_term_frequency_saturates_and_length_is_normalized():
    one = encode_document("sku")
    repeated = encode_document("sku sku")
    longer = encode_document("sku " + "unrelated " * 100)
    long_weight = dict(zip(longer.indices, longer.values, strict=True))[one.indices[0]]
    assert one.values[0] < repeated.values[0] < 2 * one.values[0]
    assert long_weight < one.values[0]
    assert one.values[0] == pytest.approx(2.2 / (1 + 1.2 * (0.25 + 0.75 / 200)))
    assert longer.indices == sorted(set(longer.indices))


async def test_collection_schema_acl_indexes_and_idempotent_points(client, document):
    indexer = QdrantIndexer(client, MockEmbeddingProvider(32), batch_size=1)
    with patch.object(client, "create_payload_index", wraps=client.create_payload_index) as create:
        await indexer.ensure_collection()
        schemas = {
            call.kwargs["field_name"]: call.kwargs["field_schema"] for call in create.call_args_list
        }
    assert schemas == {
        "dept_scope": models.PayloadSchemaType.KEYWORD,
        "min_role_level": models.PayloadSchemaType.INTEGER,
        "parent_id": models.PayloadSchemaType.KEYWORD,
        "doc_code": models.PayloadSchemaType.KEYWORD,
    }
    info = await client.get_collection(indexer.collection_name)
    assert info.config.params.vectors["dense"].size == 32
    assert info.config.params.sparse_vectors["bm25_sparse"].modifier == models.Modifier.IDF
    assert await indexer.index_document(document) == len(document.children)
    await indexer.index_document(document)
    assert (await client.count(indexer.collection_name, exact=True)).count == len(document.children)
    points, _ = await client.scroll(indexer.collection_name, with_vectors=True, limit=100)
    table_points = [point for point in points if point.payload["is_sku_table"]]
    assert table_points
    parents = {parent.parent_id: parent for parent in document.parents}
    for point in points:
        payload = point.payload
        assert payload["dept_scope"] == "warehouse"
        assert payload["min_role_level"] == 2
        assert payload["doc_code"] == "sku-indexer-test"
        assert payload["parent_content"] == parents[payload["parent_id"]].content_text
        assert payload["sha256_hash"] == document.sha256_hash
        assert set(point.vector) == {"dense", "bm25_sparse"}
    assert table_points[0].payload["sku_rows_json"]["rows"][0]["净重"] == "2.5 kg"


async def test_alias_replacement_uses_one_atomic_operation_and_preserves_old_collection(client):
    old = QdrantIndexer(client, MockEmbeddingProvider(16), "old", "production")
    new = QdrantIndexer(client, MockEmbeddingProvider(16), "new", "production")
    await old.switch_alias()
    with patch.object(
        client, "update_collection_aliases", wraps=client.update_collection_aliases
    ) as switch:
        await new.switch_alias()
        switch.assert_awaited_once()
        operations = switch.call_args.kwargs["change_aliases_operations"]
        assert len(operations) == 2
        assert operations[0].delete_alias.alias_name == "production"
        assert operations[1].create_alias.collection_name == "new"
        await new.switch_alias()
        switch.assert_awaited_once()
    assert await client.collection_exists("old")
    aliases = await client.get_aliases()
    assert [(a.alias_name, a.collection_name) for a in aliases.aliases] == [("production", "new")]


async def test_incompatible_dense_dimension_or_sparse_schema_requires_new_collection(client):
    await QdrantIndexer(client, MockEmbeddingProvider(16), "dense").ensure_collection()
    with pytest.raises(ValueError, match="不兼容"):
        await QdrantIndexer(client, MockEmbeddingProvider(32), "dense").ensure_collection()
    await client.create_collection(
        "sparse",
        vectors_config={"dense": models.VectorParams(size=16, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25_sparse": models.SparseVectorParams()},
    )
    with pytest.raises(ValueError, match="不兼容"):
        await QdrantIndexer(client, MockEmbeddingProvider(16), "sparse").ensure_collection()


@pytest.mark.parametrize("bad_result", [[], [[1.0]], [[float("nan")] * 8]])
async def test_invalid_embedding_never_writes_points(client, document, bad_result):
    provider = MockEmbeddingProvider(8)
    provider.embed = AsyncMock(return_value=bad_result)
    indexer = QdrantIndexer(client, provider, batch_size=1)
    await indexer.ensure_collection()
    with pytest.raises(ValueError, match="嵌入结果"):
        await indexer.index_document(document)
    assert (await client.count(indexer.collection_name, exact=True)).count == 0


def test_batch_size_accepts_configured_provider_upper_bound():
    indexer = QdrantIndexer(AsyncMock(), MockEmbeddingProvider(8), batch_size=2048)
    assert indexer.batch_size == 2048
    with pytest.raises(ValueError, match="批次"):
        QdrantIndexer(AsyncMock(), MockEmbeddingProvider(8), batch_size=2049)
