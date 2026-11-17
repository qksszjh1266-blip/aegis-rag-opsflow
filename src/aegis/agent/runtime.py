"""装配 M1/M2/M3 依赖；所有连接均由应用生命周期统一关闭。"""

from contextlib import AsyncExitStack

from aegis.agent.workflow import ControlledWorkflow
from aegis.config import ProviderMode
from aegis.core.circuit_breaker import RedisCircuitBreaker
from aegis.llm.generation import create_generation_client
from aegis.llm.providers import create_embedding_provider
from aegis.rag.reranker import create_reranker
from aegis.rag.retriever import HybridRetriever
from aegis.workers.tickets import TicketService


class UnavailableReranker:
    async def rerank(self, query, documents):
        """OpenAI 没有兼容的精排接口，明确进入 M2 可审计的 RRF 降级。"""
        raise RuntimeError("未配置可用的精排供应商")


def create_workflow(settings, redis, qdrant, session_factory, stack: AsyncExitStack):
    embeddings = create_embedding_provider(settings)
    stack.push_async_callback(embeddings.aclose)
    llm = create_generation_client(settings, redis)
    stack.push_async_callback(llm.aclose)
    reranker = (
        UnavailableReranker()
        if settings.rerank_provider_mode == ProviderMode.OPENAI
        else create_reranker(settings)
    )
    if hasattr(reranker, "aclose"):
        stack.push_async_callback(reranker.aclose)
    retriever = HybridRetriever(
        qdrant,
        embeddings,
        reranker,
        session_factory,
        circuit_breaker=RedisCircuitBreaker(redis, "rerank"),
    )
    return ControlledWorkflow(
        redis,
        session_factory,
        retriever,
        llm,
        TicketService(redis, session_factory),
        high_risk_keywords=settings.high_risk_keywords,
        # 一轮最多依次调用槽位、摘要、回答及一次重生成，固定租期覆盖全部预算。
        execution_timeout_seconds=(
            10
            if settings.llm_provider_mode == ProviderMode.MOCK
            else 4 * settings.llm_timeout_seconds + 10
        ),
    )
