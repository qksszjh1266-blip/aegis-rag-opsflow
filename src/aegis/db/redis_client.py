"""Redis 异步连接池工厂，关闭责任归应用生命周期。"""

from redis.asyncio import Redis

from aegis.config import Settings


def create_redis_client(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=settings.dependency_timeout_seconds,
        socket_timeout=settings.dependency_timeout_seconds,
        health_check_interval=30,
        max_connections=50,
    )
