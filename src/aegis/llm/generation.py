"""统一异步文本生成：确定性离线模拟、受熔断保护的真实兼容接口。"""

import asyncio
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import httpx
from redis.asyncio import Redis

from aegis.config import ProviderMode, Settings
from aegis.core.circuit_breaker import RedisCircuitBreaker
from aegis.core.metrics import LLM_TOKENS
from aegis.llm.evidence import NO_EVIDENCE, synthesize_evidence

_PURPOSES = frozenset({"answer", "slots", "summary", "regenerate"})


@dataclass(frozen=True)
class GenerationResult:
    """真实服务使用响应中的用量；离线模式使用字符计数作为可复现估算。"""

    text: str
    input_tokens: int
    output_tokens: int


class GenerationClient(Protocol):
    async def generate(
        self, messages: list[dict[str, str]], *, purpose: str = "answer"
    ) -> GenerationResult:
        """只返回经过协议校验的完整文本，不把半截响应当作正常结果。"""
        ...

    async def aclose(self) -> None:
        """关闭当前实例持有的外部连接。"""
        ...


def _validate_request(messages: list[dict[str, str]], purpose: str) -> None:
    if purpose not in _PURPOSES:
        raise ValueError("生成用途必须为 answer、slots、summary 或 regenerate")
    if not messages or any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or message["role"] not in {"system", "user", "assistant"}
        or not isinstance(message["content"], str)
        or not message["content"].strip()
        for message in messages
    ):
        raise ValueError("生成消息必须包含合法角色与非空文本")


def _observe(result: GenerationResult, provider: str, purpose: str) -> GenerationResult:
    for direction, count in (
        ("input", result.input_tokens),
        ("output", result.output_tokens),
    ):
        LLM_TOKENS.labels(provider=provider, direction=direction, purpose=purpose).inc(count)
    return result


def _extract_slots(text: str) -> str:
    """离线槽位只识别显式字段，不凭上下文臆造设备号、订单号或故障。"""
    patterns = {
        "device_id": r"(?:设备(?:编号|号|ID)?|device_id)\s*[:：=]\s*([^，,；;\n\s]+)",
        "order_or_wave": (
            r"(?:订单(?:编号|号)?|波次(?:编号|号)?|order_or_wave)\s*[:：=]\s*([^，,；;\n\s]+)"
        ),
        "issue": r"(?:报错现象|故障现象|报错|故障|现象|issue)\s*[:：=]\s*([^，,；;\n]+)",
    }
    slots = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            slots[name] = match.group(1).strip()
    if "issue" not in slots and any(
        word in text for word in ("失败", "报错", "卡纸", "无法", "异常")
    ):
        slots["issue"] = text[:512]
    return json.dumps(slots, ensure_ascii=False)


def _evidence_answer(messages: list[dict[str, str]], query: str) -> str:
    """只从系统提供的授权证据合成，用户消息不能注入新的证据来源。"""
    for message in messages:
        if message["role"] != "system":
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
            return synthesize_evidence(query, payload["evidence"])
    return NO_EVIDENCE


class MockGenerationClient:
    """无网络的确定性模拟；错误参数与脚本响应只能由构造参数启用。

    responses 可传全局序列或按用途分组的序列，便于精确验证重生成次数。
    wrong_units 始终返回错误量纲，验证二次核验失败后的安全拒答路径。
    """

    def __init__(
        self,
        responses: list[str] | dict[str, list[str]] | None = None,
        *,
        wrong_units: bool = False,
    ) -> None:
        self._responses = deque(responses if isinstance(responses, list) else [])
        self._by_purpose = (
            {purpose: deque(items) for purpose, items in responses.items()}
            if isinstance(responses, dict)
            else {}
        )
        self.wrong_units = wrong_units
        self.calls: list[dict[str, object]] = []

    async def generate(
        self, messages: list[dict[str, str]], *, purpose: str = "answer"
    ) -> GenerationResult:
        _validate_request(messages, purpose)
        self.calls.append({"messages": [dict(message) for message in messages], "purpose": purpose})
        user_text = next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"),
            "",
        )
        queue = self._by_purpose.get(purpose)
        if queue:
            answer = queue.popleft()
        elif self._responses:
            answer = self._responses.popleft()
        elif purpose == "slots":
            answer = _extract_slots(user_text)
        elif purpose == "summary":
            # 模拟摘要不再引入量纲换算或新事实，长度有界以支持滚动窗口。
            prior = "\n".join(
                message["content"] for message in messages if message["role"] == "system"
            )[-400:]
            recent = "\n".join(
                message["content"] for message in messages if message["role"] != "system"
            )[-800:]
            answer = "会话摘要：" + prior + "\n" + recent
        elif self.wrong_units:
            answer = "该商品净重为 999 kW。"
        else:
            answer = _evidence_answer(messages, user_text)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("模拟响应必须为非空文本")
        return _observe(
            GenerationResult(
                answer,
                sum(len(message["content"]) for message in messages),
                len(answer),
            ),
            "mock",
            purpose,
        )

    async def aclose(self) -> None:
        """模拟客户端没有需要释放的连接。"""


class OpenAICompatibleGenerationClient:
    """所有真实调用必须经过固定 llm 熔断器，超时在熔断器内部转成失败。"""

    def __init__(
        self,
        redis: Redis,
        *,
        model: str,
        base_url: str,
        api_key: str,
        provider_mode: ProviderMode,
        timeout: float = 30.0,
        max_tokens: int = 1024,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip() or not api_key.strip():
            raise ValueError("真实生成服务必须配置模型和密钥")
        if provider_mode not in {ProviderMode.OPENAI, ProviderMode.DASHSCOPE}:
            raise ValueError("真实生成服务只支持 OpenAI 或 DashScope")
        if (
            isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 120
            or type(max_tokens) is not int
            or max_tokens < 1
        ):
            raise ValueError("生成超时须为不超过 120 秒的有限正数，输出 Token 上限必须为正整数")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider_mode = provider_mode
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.breaker = RedisCircuitBreaker(redis, "llm", probe_timeout=timeout + 5)
        self._api_key = api_key
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _parse_response(payload: object) -> GenerationResult:
        if not isinstance(payload, dict) or payload.get("error") or payload.get("code"):
            raise ValueError("生成服务返回错误响应")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("生成响应必须包含唯一候选")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError("生成响应不完整或需要未授权的工具调用")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("生成响应缺少消息")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or message.get("tool_calls"):
            raise ValueError("生成响应必须为非空纯文本")
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("生成响应缺少 Token 用量")
        counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Token 用量必须为非负整数")
        return GenerationResult(content, counts[0], counts[1])

    async def _request(self, messages: list[dict[str, str]]) -> GenerationResult:
        body = {"model": self.model, "messages": messages, "stream": False}
        # 千问兼容网关沿用 max_tokens，包括显式选择 openai 协议的套餐端点。
        is_qwen = self.provider_mode is ProviderMode.DASHSCOPE or "qwen" in self.model.lower()
        token_key = "max_tokens" if is_qwen else "max_completion_tokens"
        body[token_key] = self.max_tokens
        if is_qwen:
            # 排障流程只消费最终回答，不等待推理过程；模型须支持非思考模式。
            body["enable_thinking"] = False
        async with asyncio.timeout(self.timeout):
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return self._parse_response(response.json())

    async def generate(
        self, messages: list[dict[str, str]], *, purpose: str = "answer"
    ) -> GenerationResult:
        _validate_request(messages, purpose)
        result = await self.breaker.call(self._request, messages)
        return _observe(result, self.provider_mode.value, purpose)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def create_generation_client(
    settings: Settings,
    redis: Redis,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> GenerationClient:
    """真实模式必须提供密钥；外部注入的 HTTP 客户端由调用方管理生命周期。"""
    if settings.llm_provider_mode is ProviderMode.MOCK:
        return MockGenerationClient()
    api_key = settings.llm_api_key or settings.provider_api_key
    if api_key is None:
        raise ValueError("真实生成服务必须配置 LLM_API_KEY 或 PROVIDER_API_KEY")
    dashscope = settings.llm_provider_mode is ProviderMode.DASHSCOPE
    return OpenAICompatibleGenerationClient(
        redis,
        model=settings.llm_model or ("qwen-plus" if dashscope else "gpt-4o-mini"),
        base_url=settings.llm_base_url
        or (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if dashscope
            else "https://api.openai.com/v1"
        ),
        api_key=api_key.get_secret_value(),
        provider_mode=settings.llm_provider_mode,
        timeout=settings.llm_timeout_seconds,
        max_tokens=settings.llm_max_tokens,
        http_client=http_client,
    )
