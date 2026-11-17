"""集中管理环境配置，启动时校验协议、枚举与边界。"""

from enum import StrEnum
from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from aegis.schemas.agent import AccessContext


class ProviderMode(StrEnum):
    """模型服务供应商契约。"""

    MOCK = "mock"
    DASHSCOPE = "dashscope"
    OPENAI = "openai"


class Settings(BaseSettings):
    """环境变量优先于本地 .env，忽略编排专用的环境变量。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_name: str = "aegis-rag-opsflow"
    public_base_url: str = "http://localhost:8000"
    debug: bool = False
    log_level: str = "INFO"
    mysql_dsn: str = Field(
        default="mysql+asyncmy://aegis:aegis_pass@127.0.0.1:3306/aegis_rag", repr=False
    )
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", repr=False)
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    provider_mode: ProviderMode = ProviderMode.MOCK
    llm_provider_mode: ProviderMode | None = None
    embedding_provider_mode: ProviderMode | None = None
    rerank_provider_mode: ProviderMode | None = None
    provider_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None
    gateway_api_tokens: dict[str, AccessContext] = Field(default_factory=dict, repr=False)
    embedding_dimension: int = Field(default=1536, ge=1, le=65536)
    embedding_model: str | None = Field(default=None, min_length=1)
    embedding_base_url: str | None = None
    embedding_batch_size: int = Field(default=10, ge=1, le=2048)
    rerank_model: str = Field(default="gte-rerank-v2", min_length=1)
    rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    llm_model: str | None = Field(default=None, min_length=1)
    llm_base_url: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    llm_max_tokens: int = Field(default=1024, ge=1, le=8192)
    high_risk_keywords: list[str] = Field(
        default_factory=lambda: ["漏电", "冒烟", "起火", "批量资损", "监管投诉"]
    )
    dependency_timeout_seconds: float = Field(default=3.0, gt=0, le=60)

    @model_validator(mode="after")
    def inherit_provider_modes(self) -> "Settings":
        """未显式设置的组件继承全局模式，独立模式不会改变其他组件。"""
        for name in ("llm_provider_mode", "embedding_provider_mode", "rerank_provider_mode"):
            if getattr(self, name) is None:
                setattr(self, name, self.provider_mode)
        return self

    @field_validator("gateway_api_tokens")
    @classmethod
    def validate_gateway_api_tokens(cls, value: dict[str, AccessContext]):
        """网关密钥必须显式配置，拒绝空白密钥和空白主体。"""
        if any(
            not token.strip() or not access.subject_id.strip() for token, access in value.items()
        ):
            raise ValueError("网关密钥与调用主体不能为空")
        return value

    @field_validator("high_risk_keywords")
    @classmethod
    def validate_high_risk_keywords(cls, value: list[str]) -> list[str]:
        if any(not keyword.strip() or len(keyword) > 64 for keyword in value):
            raise ValueError("高危词必须为 1 至 64 字符的非空文本")
        # 必备红线始终保留，配置只能增加自定义关键词。
        required = ["漏电", "冒烟", "起火", "批量资损", "监管投诉"]
        return list(dict.fromkeys(required + [keyword.strip() for keyword in value]))

    @field_validator("embedding_base_url", "rerank_url", "llm_base_url", "public_base_url")
    @classmethod
    def validate_provider_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("服务地址必须为不含凭据、查询参数与片段的 HTTP(S) 地址")
        return value.rstrip("/")

    @field_validator("mysql_dsn")
    @classmethod
    def validate_mysql_dsn(cls, value: str) -> str:
        try:
            url = make_url(value)
            valid = url.drivername == "mysql+asyncmy" and url.host and url.database
        except ArgumentError:
            valid = False
        if not valid:
            raise ValueError("MYSQL_DSN 必须使用 mysql+asyncmy 协议并包含主机和数据库")
        return value

    @field_validator("redis_url", "qdrant_url")
    @classmethod
    def validate_service_url(cls, value: str, info) -> str:
        parsed = urlsplit(value)
        schemes = {"redis", "rediss"} if info.field_name == "redis_url" else {"http", "https"}
        if parsed.scheme not in schemes or not parsed.hostname:
            raise ValueError("服务地址的协议或主机无效")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("日志等级无效")
        return level


@lru_cache
def get_settings() -> Settings:
    """每个进程复用一份经校验的配置。"""
    return Settings()
