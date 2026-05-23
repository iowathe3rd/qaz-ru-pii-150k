from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Language(StrEnum):
    RU = "ru"
    KK = "kk"
    RU_KK = "ru_kk"
    RU_KK_EN = "ru_kk_en"


class Surface(StrEnum):
    CALL_CENTER_ASR = "call_center_asr"
    CHAT = "chat"
    CRM_NOTE = "crm_note"
    FORMAL_NOTICE = "formal_notice"
    HARD_NEGATIVE = "hard_negative"


class PiiDensity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TemplateMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Language
    surface: Surface
    scenario: str
    pii_density: PiiDensity
    noise_type: str = "clean"
    hard_negative: bool = False


class LlmValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: str

    @model_validator(mode="after")
    def non_empty_value(self) -> LlmValue:
        if not self.value:
            raise ValueError("LLM value must not be empty")
        return self


class LlmTemplateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str
    values: dict[str, LlmValue] = Field(default_factory=dict)
    meta: TemplateMeta


class Span(BaseModel):
    start: int
    end: int
    text: str
    label: str
    opf_label: str


class ExampleMeta(TemplateMeta):
    locale: str = "KZ"
    domain: str = "banking_call_center"
    source: str = "synthetic"
    generator_version: str = "v1"
    raw_record_id: str | None = None
    shard_id: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CanonicalExample(BaseModel):
    id: str
    text: str
    spans: list[Span]
    meta: ExampleMeta


class ValidationIssue(BaseModel):
    example_id: str | None = None
    code: str
    message: str


class ValidationReport(BaseModel):
    input_path: str
    total: int = 0
    valid: int = 0
    invalid: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)


class GenerationSlice(BaseModel):
    count: int
    language: Language
    surface: Surface
    scenario: str
    pii_density: PiiDensity
    noise_type: str = "clean"
    hard_negative: bool = False
