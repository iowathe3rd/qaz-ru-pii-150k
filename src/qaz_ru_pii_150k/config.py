from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class OutputConfig(BaseModel):
    raw_llm_dir: Path = Path("data/raw_llm")
    canonical_raw: Path = Path("data/canonical/raw.jsonl")
    canonical_validated: Path = Path("data/canonical/validated.jsonl")
    reports_dir: Path = Path("reports")
    splits_dir: Path = Path("data/splits")
    opf_dir: Path = Path("data/opf")


class OpenAIConfig(BaseModel):
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "gpt-5.1"
    temperature: float = 0.9
    max_output_tokens: int = 6000
    cache_dir: Path = Path("data/raw_llm")

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.getenv(self.api_key_env)

    def resolved_base_url(self) -> str | None:
        return self.base_url or os.getenv(self.base_url_env)


class TemporalConfig(BaseModel):
    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "qaz-ru-pii-150k"
    shard_size: int = 100
    max_concurrent_activities: int = 4


class GenerationConfig(BaseModel):
    target_count: int = 150_000
    seed: int = 20260523
    shard_size: int = 100
    generator_version: str = "v1"
    languages: dict[str, float]
    surfaces: dict[str, float]
    pii_density: dict[str, float]
    scenarios: list[str]
    noise_types: list[str] = Field(default_factory=lambda: ["clean"])
    output: OutputConfig = Field(default_factory=OutputConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)


class OntologyEntry(BaseModel):
    opf_label: str
    marker_prefixes: list[str]
    validator: str | None = None


class OntologyConfig(BaseModel):
    labels: dict[str, OntologyEntry]


class PolicyConfig(BaseModel):
    public_entities: list[str] = Field(default_factory=list)
    forbidden_value_substrings: list[str] = Field(default_factory=list)
    hard_negative_rules: list[str] = Field(default_factory=list)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return payload


def load_generation_config(path: Path) -> GenerationConfig:
    payload = load_yaml(path)
    base_dir = path.parent
    openai_config_path = payload.pop("openai_config_path", None)
    if openai_config_path is not None:
        payload["openai"] = load_yaml(_resolve_config_path(base_dir, openai_config_path))
    temporal_config_path = payload.pop("temporal_config_path", None)
    if temporal_config_path is not None:
        payload["temporal"] = load_yaml(_resolve_config_path(base_dir, temporal_config_path))
    return GenerationConfig.model_validate(payload)


def _resolve_config_path(base_dir: Path, path: object) -> Path:
    config_path = Path(str(path))
    if config_path.is_absolute():
        return config_path
    if config_path.exists():
        return config_path
    return base_dir / config_path


def load_ontology(path: Path = Path("configs/ontology.yaml")) -> OntologyConfig:
    return OntologyConfig.model_validate(load_yaml(path))


def load_policy(path: Path = Path("configs/policy.yaml")) -> PolicyConfig:
    return PolicyConfig.model_validate(load_yaml(path))
