"""MySQL 异步引擎与事务会话工厂。"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from aegis.config import Settings


def create_mysql_engine(settings: Settings) -> AsyncEngine:
    """延迟连接、失效连接探测与回收，避免重启后复用断开的连接。"""
    return create_async_engine(
        settings.mysql_dsn,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        pool_timeout=settings.dependency_timeout_seconds,
        connect_args={
            "connect_timeout": settings.dependency_timeout_seconds,
            "init_command": "SET time_zone = '+00:00'",
            "charset": "utf8mb4",
        },
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """业务层使用 factory.begin() 明确提交或回滚事务。"""
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
