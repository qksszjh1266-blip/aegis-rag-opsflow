"""配置在进程启动前拒绝非法供应商和存储协议。"""

import pytest
from pydantic import ValidationError

from aegis.config import ProviderMode, Settings


@pytest.mark.parametrize("mode", ["mock", "dashscope", "openai"])
def test_provider_modes(mode):
    assert Settings(_env_file=None, provider_mode=mode).provider_mode == ProviderMode(mode)


@pytest.mark.parametrize(
    "override",
    [
        {"provider_mode": "unknown"},
        {"mysql_dsn": "mysql+pymysql://localhost/aegis"},
        {"mysql_dsn": "not-a-url"},
        {"redis_url": "http://localhost:6379"},
        {"qdrant_url": "file:///tmp/qdrant"},
        {"dependency_timeout_seconds": 0},
        {"log_level": "unknown"},
    ],
)
def test_invalid_settings_are_rejected(override):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **override)


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("PROVIDER_MODE=mock\n", encoding="utf-8")
    monkeypatch.setenv("PROVIDER_MODE", "openai")
    assert Settings(_env_file=env_file).provider_mode is ProviderMode.OPENAI


@pytest.mark.parametrize("mode", list(ProviderMode))
def test_component_modes_inherit_global_provider(mode):
    settings = Settings(_env_file=None, provider_mode=mode)
    assert settings.llm_provider_mode is mode
    assert settings.embedding_provider_mode is mode
    assert settings.rerank_provider_mode is mode


def test_explicit_component_modes_override_only_selected_components(monkeypatch):
    monkeypatch.setenv("PROVIDER_MODE", "dashscope")
    monkeypatch.setenv("LLM_PROVIDER_MODE", "openai")
    monkeypatch.setenv("EMBEDDING_PROVIDER_MODE", "mock")
    settings = Settings(_env_file=None)
    assert settings.llm_provider_mode is ProviderMode.OPENAI
    assert settings.embedding_provider_mode is ProviderMode.MOCK
    assert settings.rerank_provider_mode is ProviderMode.DASHSCOPE


def test_gateway_token_mapping_is_validated_and_hidden_from_repr():
    settings = Settings(
        _env_file=None,
        gateway_api_tokens={
            "sensitive-test-token": {
                "subject_id": "dify",
                "dept_scope": ["customer_service"],
                "role_level": 1,
            }
        },
    )
    assert settings.gateway_api_tokens["sensitive-test-token"].role_level == 1
    assert "sensitive-test-token" not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            gateway_api_tokens={" ": settings.gateway_api_tokens["sensitive-test-token"]},
        )
