"""可替换的精排供应商；返回索引而非模型生成文本，避免改写证据。"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from aegis.config import ProviderMode, Settings


@dataclass(frozen=True)
class RerankScore:
    """索引始终指向调用方传入的候选数组。"""

    index: int
    score: float


class Reranker(Protocol):
    async def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        """返回所有候选的相关度，网络实现必须支持协程取消。"""
        ...


def validate_scores(scores: Sequence[RerankScore], count: int) -> None:
    """完整排列与有限分数校验防止异常供应商响应丢失或串接证据。"""
    if len(scores) != count or {item.index for item in scores} != set(range(count)):
        raise ValueError("精排结果必须完整覆盖候选且索引不能重复")
    for item in scores:
        if type(item.index) is not int or not math.isfinite(item.score) or not 0 <= item.score <= 1:
            raise ValueError("精排结果索引或分数非法")


class MockReranker:
    """离线词面重合度模拟器，用于可重复验收，不替代真实语义模型。"""

    async def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        query_terms = set(re.findall(r"[a-z0-9_-]+|[\u4e00-\u9fff]", query.lower()))
        # 显式型号和故障码比礼貌用语、常见汉字更有判别力，防止长问句挤掉规格表。
        # 权重只依赖查询标识，仍按完整词匹配，不能将 Std 与 Std-Pro 当作同一型号。
        weights = {
            term: (
                len(query_terms) + 1
                if re.search(r"[a-z]", term) and re.search(r"[-_]", term)
                else 1
            )
            for term in query_terms
        }
        total_weight = max(sum(weights.values()), 1)
        scores = []
        for index, document in enumerate(documents):
            terms = set(re.findall(r"[a-z0-9_-]+|[\u4e00-\u9fff]", document.lower()))
            score = sum(weights[term] for term in query_terms & terms) / total_weight
            scores.append(RerankScore(index, score))
        return sorted(scores, key=lambda item: (-item.score, item.index))


class DashScopeReranker:
    """异步 HTTP 精排；生命周期归属明确，注入的客户端由调用方关闭。"""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gte-rerank-v2",
        url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DashScope 精排必须配置 API 密钥")
        self._api_key = api_key
        self.model = model
        self.url = url
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=0.6)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        if not documents:
            return []
        response = await self._client.post(
            self.url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {"top_n": len(documents), "return_documents": False},
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code"):
            raise ValueError("DashScope 精排返回业务错误")
        scores = [
            RerankScore(index=item["index"], score=item["relevance_score"])
            for item in payload["output"]["results"]
        ]
        validate_scores(scores, len(documents))
        return sorted(scores, key=lambda item: (-item.score, item.index))


def create_reranker(
    settings: Settings, *, http_client: httpx.AsyncClient | None = None
) -> Reranker:
    """显式选择供应商，避免把未配置的真实服务静默替换为模拟模型。"""
    if settings.rerank_provider_mode == ProviderMode.MOCK:
        return MockReranker()
    if settings.rerank_provider_mode != ProviderMode.DASHSCOPE:
        raise ValueError("当前真实精排仅支持 DashScope；OpenAI 模式需显式注入精排实现")
    if settings.provider_api_key is None:
        raise ValueError("DashScope 精排必须配置 PROVIDER_API_KEY")
    return DashScopeReranker(
        settings.provider_api_key.get_secret_value(),
        model=settings.rerank_model,
        url=settings.rerank_url,
        http_client=http_client,
    )
