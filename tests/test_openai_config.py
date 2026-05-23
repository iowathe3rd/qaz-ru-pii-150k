from __future__ import annotations

from qaz_ru_pii_150k.config import OpenAIConfig
from qaz_ru_pii_150k.config import load_generation_config
from qaz_ru_pii_150k.llm import OpenAITemplateProvider


def test_openai_config_resolves_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CUSTOM_OPENAI_KEY", "env-key")
    monkeypatch.setenv("CUSTOM_OPENAI_BASE_URL", "https://env.example/v1")
    config = OpenAIConfig(
        api_key_env="CUSTOM_OPENAI_KEY",
        base_url_env="CUSTOM_OPENAI_BASE_URL",
    )

    assert config.resolved_api_key() == "env-key"
    assert config.resolved_base_url() == "https://env.example/v1"


def test_openai_config_prefers_explicit_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    config = OpenAIConfig(api_key="yaml-key", base_url="https://yaml.example/v1")

    assert config.resolved_api_key() == "yaml-key"
    assert config.resolved_base_url() == "https://yaml.example/v1"


def test_openai_provider_passes_client_options(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("qaz_ru_pii_150k.llm.OpenAI", FakeOpenAI)

    OpenAITemplateProvider(OpenAIConfig(api_key="yaml-key", base_url="https://yaml.example/v1"))

    assert captured == {"api_key": "yaml-key", "base_url": "https://yaml.example/v1"}


def test_generation_config_can_reference_runtime_configs(tmp_path) -> None:
    openai = tmp_path / "openai.yaml"
    temporal = tmp_path / "temporal.yaml"
    generation = tmp_path / "generation.yaml"
    openai.write_text(
        "api_key: test-key\nbase_url: https://example.test/v1\nmodel: test-model\n",
        encoding="utf-8",
    )
    temporal.write_text("address: localhost:7233\nnamespace: default\n", encoding="utf-8")
    generation.write_text(
        """
target_count: 100
seed: 1
shard_size: 20
openai_config_path: openai.yaml
temporal_config_path: temporal.yaml
languages: {ru: 1.0}
surfaces: {chat: 1.0}
pii_density: {low: 1.0}
scenarios: [identity_verification]
""".strip(),
        encoding="utf-8",
    )

    config = load_generation_config(generation)

    assert config.openai.api_key == "test-key"
    assert config.openai.base_url == "https://example.test/v1"
    assert config.openai.model == "test-model"
    assert config.temporal.address == "localhost:7233"
