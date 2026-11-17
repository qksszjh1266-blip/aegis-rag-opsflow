"""向量供应商的离线可重现性、协议校验和批次边界。"""

import asyncio
import json
import math

import httpx
import pytest
from pydantic import ValidationError

from aegis.config import Settings
from aegis.llm.providers import (
    MockEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    create_embedding_provider,
)


async def test_mock_is_normalized_deterministic_and_content_sensitive():
    provider = MockEmbeddingProvider(dimension=64)
    texts = ["SKU-001 净重 2.3 kg", "仓储扫码失败重试"]
    vectors = await provider.embed(texts)
    assert vectors == await MockEmbeddingProvider(64).embed(texts)
    assert vectors[0] != vectors[1]
    for vector in vectors:
        assert len(vector) == 64
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1)
    assert await provider.embed([]) == []
    await provider.aclose()


@pytest.mark.parametrize("texts", [[""], ["   "], [None]])
async def test_mock_rejects_empty_text(texts):
    with pytest.raises(ValueError, match="非空文本"):
        await MockEmbeddingProvider().embed(texts)


def make_http_provider(client, **kwargs):
    return OpenAICompatibleEmbeddingProvider(
        dimension=2,
        model="测试模型",
        base_url="https://example.test/v1",
        api_key="secret",
        http_client=client,
        **kwargs,
    )


async def test_http_batches_restore_response_order_and_keep_external_client_open():
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(text), 1.0]}
                    for i, text in reversed(list(enumerate(payload["input"])))
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = make_http_provider(client, batch_size=2)
        assert await provider.embed(["1", "2", "3"]) == [[1, 1], [2, 1], [3, 1]]
        assert [request["input"] for request in requests] == [["1", "2"], ["3"]]
        assert requests[0]["dimensions"] == 2
        assert requests[0]["encoding_format"] == "float"
        assert await provider.embed([]) == []
        await provider.aclose()
        assert not client.is_closed


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": []},
        {"data": [None]},
        {"data": [{"index": -1, "embedding": [1, 2]}]},
        {"data": [{"index": True, "embedding": [1, 2]}]},
        {"data": [{"index": 0, "embedding": [1]}]},
        {"data": [{"index": 0, "embedding": [1, "2"]}]},
        {"data": [{"index": 0, "embedding": [True, 2]}]},
        {"data": [{"index": 0, "embedding": [0, 0]}]},
    ],
)
async def test_invalid_response_is_rejected(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ValueError):
            await make_http_provider(client).embed(["测试"])


async def test_duplicate_indices_and_nonfinite_values_are_rejected():
    async with httpx.AsyncClient() as client:
        provider = make_http_provider(client)
        with pytest.raises(ValueError, match="重复"):
            provider._parse_vectors({"data": [{"index": 0, "embedding": [1, 2]}] * 2}, 2)
        with pytest.raises(ValueError, match="有限"):
            provider._parse_vectors({"data": [{"index": 0, "embedding": [math.inf, 2]}]}, 1)


async def test_http_error_and_cancellation_are_propagated():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(429))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await make_http_provider(client).embed(["测试"])

    async def cancelled(request):
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelled)) as client:
        with pytest.raises(asyncio.CancelledError):
            await make_http_provider(client).embed(["测试"])


async def test_factory_selects_model_and_client_lifecycle():
    assert isinstance(create_embedding_provider(Settings(_env_file=None)), MockEmbeddingProvider)
    for mode, model, host in (
        ("dashscope", "text-embedding-v4", "dashscope.aliyuncs.com"),
        ("openai", "text-embedding-3-small", "api.openai.com"),
    ):
        with pytest.raises(ValueError, match="PROVIDER_API_KEY"):
            create_embedding_provider(Settings(_env_file=None, provider_mode=mode))
        provider = create_embedding_provider(
            Settings(_env_file=None, provider_mode=mode, provider_api_key="secret")
        )
        assert provider.model == model
        assert host in provider.base_url
        await provider.aclose()
        assert provider._client.is_closed


@pytest.mark.parametrize(
    "override",
    [
        {"embedding_dimension": 0},
        {"embedding_batch_size": 0},
        {"embedding_base_url": "file:///tmp/provider"},
        {"embedding_base_url": "https://user:secret@example.test/v1"},
        {"rerank_url": "https://example.test/rerank?api_key=secret"},
    ],
)
def test_provider_settings_reject_invalid_values(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)
