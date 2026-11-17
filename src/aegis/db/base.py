"""声明式基类及各核心表共用的审计字段。"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, FetchedValue, MetaData, text
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """统一约束命名，保证后续迁移可稳定定位约束。"""

    # MySQL 不支持 INSERT RETURNING，主动读取服务端时间戳以免异步属性访问触发隐式 IO。
    __mapper_args__ = {"eager_defaults": True}

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class TimestampMixin:
    """时间由数据库生成；连接会话统一使用 UTC。"""

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime().with_variant(DATETIME(fsp=6), "mysql"),
        server_default=text("CURRENT_TIMESTAMP(6)"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime().with_variant(DATETIME(fsp=6), "mysql"),
        server_default=text("CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)"),
        server_onupdate=FetchedValue(),
    )
