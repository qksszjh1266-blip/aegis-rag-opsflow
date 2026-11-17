"""验证 DashScope HTTP 契约和畸形响应防护，不依赖真实模型服务。"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from aegis.config import Settings
from aegis.rag.parser import DocumentMetadata, MarkdownParser
from aegis.rag.reranker import (
    DashScopeReranker,
    MockReranker,
    RerankScore,
    create_reranker,
    validate_scores,
)

_SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "seed_docs"


async def test_mock_reranker_is_deterministic():
    reranker = MockReranker()
    documents = ["包装重量 2 kg", "SKU-A 功率 1200 W"]
    scores = await reranker.rerank("SKU-A 功率", documents)
    assert scores == await reranker.rerank("SKU-A 功率", documents)
    assert scores[0].index == 1
    assert isinstance(create_reranker(Settings(_env_file=None)), MockReranker)


async def test_long_seed_sku_query_keeps_exact_specification_table_in_top_five():
    parser = MarkdownParser()
    parents = []
    for name, doc_type in (
        ("ec_sku_specs.md", "SKU_SPEC"),
        ("ec_warehouse_sop.md", "WAREHOUSE_SOP"),
    ):
        parsed = parser.parse(
            await asyncio.to_thread((_SEED_DIR / name).read_text, encoding="utf-8"),
            DocumentMetadata(doc_code=name, doc_type=doc_type, dept_scope="warehouse"),
        )
        parents.extend(parsed.parents)
    query = "EC-ih200-Std 的净重是多少？请仅回答这个 SKU 的净重。"
    scores = await MockReranker().rerank(query, [parent.content_text for parent in parents])
    top = [parents[item.index] for item in scores[:5]]
    assert any("| EC-ih200-Std |" in parent.content_text for parent in top)


async def test_exact_identifier_weight_does_not_match_a_longer_sku_prefix():
    scores = await MockReranker().rerank(
        "EC-ih200-Std 的净重是多少？请仅回答这个 SKU 的净重。",
        ["EC-ih200-Std 净重 1.8 kg", "EC-ih200-Std-Pro 的净重是多少？请仅回答这个 SKU 的净重。"],
    )
    assert scores[0].index == 0


async def test_dashscope_request_and_response_contract():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "gte-rerank-v2"
        assert body["input"] == {"query": "功率", "documents": ["2 kg", "1200 W"]}
        assert body["parameters"] == {"top_n": 2, "return_documents": False}
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.1},
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = DashScopeReranker("test-key", http_client=client)
        scores = await reranker.rerank("功率", ["2 kg", "1200 W"])
        assert scores == [RerankScore(1, 0.9), RerankScore(0, 0.1)]
        await reranker.aclose()
        assert not client.is_closed


@pytest.mark.parametrize(
    "scores",
    [
        [RerankScore(0, 0.2), RerankScore(0, 0.3)],
        [RerankScore(0, 0.2)],
        [RerankScore(0, float("nan")), RerankScore(1, 0.5)],
        [RerankScore(0, 1.1), RerankScore(1, 0.5)],
        [RerankScore(False, 0.5), RerankScore(1, 0.5)],
    ],
)
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        validate_scores(scores, 2)


async def test_dashscope_business_error_raises():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"code": "InvalidApiKey", "message": "错误"})
        )
    ) as client:
        reranker = DashScopeReranker("test-key", http_client=client)
        with pytest.raises(ValueError, match="业务错误"):
            await reranker.rerank("功率", ["1200 W"])
