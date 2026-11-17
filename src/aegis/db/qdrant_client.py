"""Qdrant 异步客户端工厂。"""

from qdrant_client import AsyncQdrantClient

from aegis.config import Settings


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=settings.dependency_timeout_seconds,
        # 启动时仅创建客户端，连通性由就绪探针统一判断。
        check_compatibility=False,
    )
