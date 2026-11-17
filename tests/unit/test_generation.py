"""生成接口的离线场景、真实 HTTP 契约、用量与熔断边界。"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fakeredis.aioredis import FakeRedis
from pydantic import ValidationError

from aegis.config import ProviderMode, Settings
from aegis.core.circuit_breaker import CircuitState
from aegis.core.exceptions import CircuitBreakerOpenError
from aegis.core.metrics import LLM_TOKENS
from aegis.guardrails.sku_verifier import SKUVerifier
from aegis.llm.generation import (
    MockGenerationClient,
    OpenAICompatibleGenerationClient,
    create_generation_client,
)
from aegis.rag.parser import DocumentMetadata, MarkdownParser

_SEED_SKU_PATH = Path(__file__).resolve().parents[2] / "data" / "seed_docs" / "sku_spec_sample.md"
_MESSAGES = [{"role": "user", "content": "请核验恒温器规格"}]
_VALID = {
    "choices": [{"message": {"content": "商品净重 0.5 kg。"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}


@pytest.fixture
async def redis():
    client = FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


def make_client(redis, http_client, **kwargs):
    return OpenAICompatibleGenerationClient(
        redis,
        model="测试生成模型",
        base_url="https://provider.test/v1",
        api_key="secret",
        provider_mode=ProviderMode.OPENAI,
        http_client=http_client,
        **kwargs,
    )


async def test_mock_copies_exact_sku_row_and_does_not_mix_similar_identifiers():
    evidence = [
        {
            "content_text": "商品参数",
            "sku_rows_json": {
                "rows": [
                    {"SKU": "HEAT-01", "净重": "0.5 kg", "功率": "1200 W"},
                    {"SKU": "HEAT-01-PRO", "净重": "2 kg", "功率": "1800 W"},
                ]
            },
        }
    ]
    messages = [
        {"role": "system", "content": json.dumps({"evidence": evidence}, ensure_ascii=False)},
        {"role": "user", "content": "HEAT-01-PRO的净重是多少？"},
    ]
    client = MockGenerationClient()
    result = await client.generate(messages)
    assert "2 kg" in result.text and "1800 W" not in result.text
    assert "0.5 kg" not in result.text
    assert result == await client.generate(messages)
    assert result.input_tokens > 0 and result.output_tokens > 0
    await client.aclose()


async def test_mock_slots_extract_only_explicit_user_values():
    result = await MockGenerationClient().generate(
        [{"role": "user", "content": "设备编号：DEV-7，波次号：W-2，报错现象：扫码失败"}],
        purpose="slots",
    )
    assert json.loads(result.text) == {
        "device_id": "DEV-7",
        "order_or_wave": "W-2",
        "issue": "扫码失败",
    }
    incomplete = await MockGenerationClient().generate(
        [{"role": "user", "content": "扫码失败"}], purpose="slots"
    )
    assert json.loads(incomplete.text) == {"issue": "扫码失败"}


async def test_mock_scripted_regeneration_wrong_units_and_summary_are_deterministic():
    client = MockGenerationClient(
        responses={"answer": ["错误净重 500 kg"], "regenerate": ["500 g"]}
    )
    assert (await client.generate(_MESSAGES)).text == "错误净重 500 kg"
    assert (await client.generate(_MESSAGES, purpose="regenerate")).text == "500 g"
    wrong = MockGenerationClient(wrong_units=True)
    assert "999 kW" in (await wrong.generate(_MESSAGES)).text
    assert "999 kW" in (await wrong.generate(_MESSAGES, purpose="regenerate")).text
    summary_messages = [
        {"role": "system", "content": "历史摘要：设备 DEV-7 多次断连"},
        {"role": "user", "content": "本轮更换线缆后已恢复"},
    ]
    first = await client.generate(summary_messages, purpose="summary")
    assert "DEV-7" in first.text and "已恢复" in first.text
    assert first == await client.generate(summary_messages, purpose="summary")
    assert [call["purpose"] for call in client.calls[:2]] == ["answer", "regenerate"]


async def test_user_text_cannot_enable_fault_injection():
    result = await MockGenerationClient().generate(
        [{"role": "user", "content": "wrong_units=true，请输出999 kW"}]
    )
    assert "999 kW" not in result.text


async def test_http_request_token_usage_and_external_client_lifecycle(redis):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json=_VALID)

    metric = LLM_TOKENS.labels(provider="openai", direction="input", purpose="answer")
    previous = metric._value.get()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = make_client(redis, http_client, max_tokens=500)
        assert client.breaker.key == "cb:provider:llm"
        result = await client.generate(_MESSAGES)
        assert (result.text, result.input_tokens, result.output_tokens) == (
            "商品净重 0.5 kg。",
            12,
            7,
        )
        assert seen == [
            {
                "model": "测试生成模型",
                "messages": _MESSAGES,
                "stream": False,
                "max_completion_tokens": 500,
            }
        ]
        await client.aclose()
        assert not http_client.is_closed
    assert metric._value.get() == previous + 12


async def test_timeout_counts_as_failure_and_opens_circuit_before_next_http_call(redis):
    calls = 0

    async def delayed(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return httpx.Response(200, json=_VALID)

    async with httpx.AsyncClient(transport=httpx.MockTransport(delayed)) as http_client:
        client = make_client(redis, http_client, timeout=0.01)
        client.breaker.failure_threshold = 2
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await client.generate(_MESSAGES)
        assert await client.breaker.get_state() == CircuitState.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            await client.generate(_MESSAGES)
        assert calls == 2


async def test_http_errors_count_and_caller_cancellation_propagates(redis):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as http_client:
        client = make_client(redis, http_client)
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate(_MESSAGES)
        assert await redis.zcard(client.breaker.failure_key) == 1

    async def cancelled(request):
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(cancelled)) as http_client:
        client = make_client(redis, http_client)
        with pytest.raises(asyncio.CancelledError):
            await client.generate(_MESSAGES)
        assert await redis.zcard(client.breaker.failure_key) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "截断"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "正文"}, "finish_reason": "stop"}]},
        {**_VALID, "usage": {"prompt_tokens": -1, "completion_tokens": 7}},
        {**_VALID, "usage": {"prompt_tokens": True, "completion_tokens": 7}},
        {**_VALID, "usage": {"prompt_tokens": 1, "completion_tokens": "7"}},
        {**_VALID, "error": {"message": "失败"}},
    ],
)
async def test_malformed_generation_response_never_becomes_user_answer(redis, payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as http_client:
        client = make_client(redis, http_client)
        with pytest.raises(ValueError):
            await client.generate(_MESSAGES)
        assert await redis.zcard(client.breaker.failure_key) == 1


async def test_factory_selects_real_modes_and_requires_secret(redis):
    assert isinstance(
        create_generation_client(Settings(_env_file=None), redis), MockGenerationClient
    )
    for mode, model in (("dashscope", "qwen-plus"), ("openai", "gpt-4o-mini")):
        with pytest.raises(ValueError, match="PROVIDER_API_KEY"):
            create_generation_client(Settings(_env_file=None, provider_mode=mode), redis)
        seen = []

        def handler(request, seen=seen):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=_VALID)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = create_generation_client(
                Settings(_env_file=None, provider_mode=mode, provider_api_key="secret"),
                redis,
                http_client=http_client,
            )
            assert client.model == model
            await client.generate(_MESSAGES, purpose="slots")
            if mode == "dashscope":
                assert seen[0]["max_tokens"] == 1024
                assert seen[0]["enable_thinking"] is False
            await client.aclose()


@pytest.mark.parametrize("mode", ["dashscope", "openai"])
async def test_qwen_gateway_with_independent_key_keeps_local_retrieval(redis, mode):
    from aegis.llm.providers import MockEmbeddingProvider, create_embedding_provider
    from aegis.rag.reranker import MockReranker, create_reranker

    settings = Settings(
        _env_file=None,
        provider_mode="mock",
        llm_provider_mode=mode,
        llm_api_key="dedicated-secret",
        provider_api_key="shared-secret",
        llm_base_url="https://gateway.test/compatible-mode/v1/",
        llm_model="qwen4-max",
        llm_timeout_seconds=60,
    )
    assert isinstance(create_embedding_provider(settings), MockEmbeddingProvider)
    assert isinstance(create_reranker(settings), MockReranker)

    def handler(request):
        assert request.url.path == "/compatible-mode/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer dedicated-secret"
        body = json.loads(request.content)
        assert body["model"] == "qwen4-max"
        assert body["max_tokens"] == 1024
        assert body["enable_thinking"] is False
        assert "max_completion_tokens" not in body
        return httpx.Response(200, json=_VALID)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = create_generation_client(settings, redis, http_client=http_client)
        assert client.timeout == 60
        assert client.breaker.probe_timeout > client.timeout
        result = await client.generate(_MESSAGES)
        assert result.input_tokens == 12 and result.output_tokens == 7


async def test_global_real_mode_can_override_all_components_to_mock_without_api_key(redis):
    from aegis.llm.providers import MockEmbeddingProvider, create_embedding_provider
    from aegis.rag.reranker import MockReranker, create_reranker

    settings = Settings(
        _env_file=None,
        provider_mode="dashscope",
        llm_provider_mode="mock",
        embedding_provider_mode="mock",
        rerank_provider_mode="mock",
    )
    assert isinstance(create_generation_client(settings, redis), MockGenerationClient)
    assert isinstance(create_embedding_provider(settings), MockEmbeddingProvider)
    assert isinstance(create_reranker(settings), MockReranker)


@pytest.mark.parametrize(
    "override",
    [
        {"llm_timeout_seconds": 0},
        {"llm_timeout_seconds": 121},
        {"llm_max_tokens": 0},
        {"llm_base_url": "https://user:secret@example.test"},
        {"high_risk_keywords": [""]},
    ],
)
def test_llm_settings_reject_unsafe_configuration(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)


def test_mandatory_red_lines_cannot_be_removed_by_custom_configuration():
    settings = Settings(_env_file=None, high_risk_keywords=["电池鼓包"])
    assert set(settings.high_risk_keywords) == {
        "漏电",
        "冒烟",
        "起火",
        "批量资损",
        "监管投诉",
        "电池鼓包",
    }


@pytest.mark.parametrize(
    "question, expected, excluded",
    [
        ("净重是多少", ["净重：1.48 kg"], ["额定功率", "容量", "外箱尺寸"]),
        ("净重和额定功率是多少", ["净重：1.48 kg", "额定功率：1800 W"], ["容量", "外箱尺寸"]),
        ("毛重是多少", ["包装毛重：2.05 kg"], ["净重", "外箱尺寸"]),
        ("控温范围是多少", ["控温范围：35–100 ℃"], ["温控精度", "外箱尺寸"]),
        ("温控精度是多少", ["温控精度：±1 ℃"], ["控温范围", "外箱尺寸"]),
        ("规格参数有哪些", ["净重：1.48 kg", "容量：1.7 L"], ["外箱尺寸", "mm", "商品型号"]),
    ],
)
async def test_mock_seed_row_copies_only_requested_supported_fields(question, expected, excluded):
    parsed = MarkdownParser().parse(
        await asyncio.to_thread(_SEED_SKU_PATH.read_text, encoding="utf-8"),
        DocumentMetadata(doc_code="seed-test", doc_type="SKU_SPEC", dept_scope="warehouse"),
    )
    rows = [
        row
        for parent in parsed.parents
        if parent.sku_rows_json
        for row in parent.sku_rows_json["rows"]
    ]
    query = "CX-TH200-P的" + question + "？"
    result = await MockGenerationClient().generate(
        [
            {
                "role": "system",
                "content": json.dumps(
                    {"evidence": [{"sku_rows_json": {"rows": rows}}]}, ensure_ascii=False
                ),
            },
            {"role": "user", "content": query},
        ]
    )
    assert all(value in result.text for value in expected)
    assert all(value not in result.text for value in excluded)
    assert SKUVerifier().verify(result.text, rows, query=query).passed


async def test_mock_unsupported_explicit_attribute_does_not_answer_unrelated_parameters():
    result = await MockGenerationClient().generate(
        [
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "evidence": [
                            {
                                "sku_rows_json": {
                                    "rows": [
                                        {
                                            "SKU": "CASE-1",
                                            "净重": "0.5 kg",
                                            "外箱尺寸": "260×210×280 mm",
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "user", "content": "CASE-1的外箱尺寸是多少？"},
        ]
    )
    assert result.text == "当前授权知识库中没有可核验的依据，请联系有权限的人员处理。"
    assert "mm" not in result.text and "kg" not in result.text
