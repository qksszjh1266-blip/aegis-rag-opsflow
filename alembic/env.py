"""通过 asyncmy 执行在线迁移，同时支持离线 SQL 审核。"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from aegis.config import Settings
from aegis.db import models  # noqa: F401
from aegis.db.base import Base
from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=Settings().mysql_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    settings = Settings()
    engine = create_async_engine(
        settings.mysql_dsn,
        poolclass=pool.NullPool,
        connect_args={
            "connect_timeout": settings.dependency_timeout_seconds,
            "init_command": "SET time_zone = '+00:00'",
            "charset": "utf8mb4",
        },
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
