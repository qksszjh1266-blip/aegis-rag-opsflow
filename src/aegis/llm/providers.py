"""异步向量接口：离线确定性特征与 OpenAI 兼容的真实模型服务。"""

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import httpx

from aegis.config import ProviderMode, Settings


class EmbeddingProvider(Protocol):
    """索引与查询共用同一模型、维度和预处理方式。"""

    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序返回向量，失败时抛出异常且不返回部分结果。"""
        ...

    async def aclose(self) -> None:
        """释放由当前供应商创建的资源。"""
        ...


def _validate_texts(texts: list[str]) -> None:
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("向量输入必须是非空文本")


class MockEmbeddingProvider:
    """稳定散列词频向量用于离线联调，不替代真实语义模型的质量评测。"""

    def __init__(self, dimension: int = 1536) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("向量维度必须为正整数")
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        _validate_texts(texts)
        result = []
        for text in texts:
            # 中文保留单字和相邻双字，英文及 SKU 保留完整标识符。
            tokens = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*|[\u3400-\u9fff]", text.lower())
            tokens += re.findall(r"(?=([\u3400-\u9fff]{2}))", text)
            counts = Counter(tokens or [text])
            vector = [0.0] * self.dimension
            for token, count in counts.items():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                vector[index] += 1.0 + math.log(count)
            norm = math.sqrt(sum(value * value for value in vector))
            result.append([value / norm for value in vector])
        return result

    async def aclose(self) -> None:
        """离线供应商没有外部连接。"""


class OpenAICompatibleEmbeddingProvider:
    """通过异步 HTTP 调用兼容接口，严格校验响应后才允许写入索引。"""

    def __init__(
        self,
        *,
        dimension: int,
        model: str,
        base_url: str,
        api_key: str,
        batch_size: int = 10,
        timeout: float = 3.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if dimension < 1 or batch_size < 1 or timeout <= 0:
            raise ValueError("维度、批量大小和超时必须为正数")
        if not api_key.strip() or not model.strip():
            raise ValueError("真实向量服务必须配置模型与密钥")
        self.dimension = dimension
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self._api_key = api_key
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    def _parse_vectors(self, payload: object, count: int) -> list[list[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("向量响应缺少 data 数组")
        records = payload["data"]
        if len(records) != count:
            raise ValueError("向量响应数量与输入不一致")
        ordered: dict[int, list[float]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("向量响应记录必须为对象")
            index = record.get("index")
            if type(index) is not int or not 0 <= index < count or index in ordered:
                raise ValueError("向量响应索引缺失、重复或越界")
            vector = record.get("embedding")
            if not isinstance(vector, list) or len(vector) != self.dimension:
                raise ValueError("向量维度与集合配置不一致")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError("向量元素必须为有限数值")
            if not any(vector):
                raise ValueError("向量响应不得为全零向量")
            ordered[index] = [float(value) for value in vector]
        return [ordered[index] for index in range(count)]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        _validate_texts(texts)
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.model,
                    "input": batch,
                    "dimensions": self.dimension,
                    "encoding_format": "float",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            vectors.extend(self._parse_vectors(response.json(), len(batch)))
        return vectors

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def create_embedding_provider(
    settings: Settings, *, http_client: httpx.AsyncClient | None = None
) -> EmbeddingProvider:
    """按运行模式选择供应商；外部传入的 HTTP 客户端由调用者关闭。"""
    if settings.embedding_provider_mode is ProviderMode.MOCK:
        return MockEmbeddingProvider(settings.embedding_dimension)
    if settings.provider_api_key is None:
        raise ValueError("真实向量服务必须配置 PROVIDER_API_KEY")
    dashscope = settings.embedding_provider_mode is ProviderMode.DASHSCOPE
    base_url = settings.embedding_base_url or (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if dashscope
        else "https://api.openai.com/v1"
    )
    model = settings.embedding_model or (
        "text-embedding-v4" if dashscope else "text-embedding-3-small"
    )
    # DashScope 默认模型每批最多十条，避免全局批量配置超过供应商上限。
    batch_size = (
        min(settings.embedding_batch_size, 10) if dashscope else settings.embedding_batch_size
    )
    return OpenAICompatibleEmbeddingProvider(
        dimension=settings.embedding_dimension,
        model=model,
        base_url=base_url,
        api_key=settings.provider_api_key.get_secret_value(),
        batch_size=batch_size,
        timeout=settings.dependency_timeout_seconds,
        http_client=http_client,
    )
