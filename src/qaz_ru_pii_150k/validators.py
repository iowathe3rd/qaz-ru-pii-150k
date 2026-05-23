from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from .config import PolicyConfig
from .constants import OPF_LABELS
from .io import read_jsonl, write_jsonl
from .models import CanonicalExample, ValidationIssue, ValidationReport
from .ontology import Ontology

VALIDATOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone_kz": re.compile(r"^(?:\+?7|8)?[\s\-\(\)]*(?:7\d{2}|700|701|702|705|707|708|747|775|776|777|778)[\d\s\-\(\)]{7,}$|[а-яәіңғүұқөһ\s]+$", re.IGNORECASE),
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "iin_kz": re.compile(r"^\d{12}$|^[а-яәіңғүұқөһ\s]+$", re.IGNORECASE),
    "bin_kz": re.compile(r"^\d{12}$|^[а-яәіңғүұқөһ\s]+$", re.IGNORECASE),
    "iban_kz": re.compile(r"^KZ[0-9A-Z\s]{18,}$", re.IGNORECASE),
    "card_pan": re.compile(r"^(?:\d[ -]?){13,19}$"),
    "card_last4": re.compile(r"^\d{4}$|^[а-яәіңғүұқөһ\s]+$", re.IGNORECASE),
    "sms_code": re.compile(r"^\d{4,8}$|^[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9\- ]{4,64}$"),
    "otp": re.compile(r"^\d{4,8}$|^[A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9\- ]{4,64}$"),
    "cvv": re.compile(r"^\d{3,4}$|^[а-яәіңғүұқөһ\s]+$", re.IGNORECASE),
    "pin": re.compile(r"^\d{4,6}$|^[а-яәіңғүұқөһ\s]+$", re.IGNORECASE),
    "private_url": re.compile(r"^https?://", re.IGNORECASE),
}


def validate_example(
    example: CanonicalExample,
    *,
    ontology: Ontology,
    policy: PolicyConfig,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    previous_end = -1
    for idx, span in enumerate(example.spans):
        prefix = f"span {idx}"
        if span.label not in ontology.fine_labels:
            issues.append(ValidationIssue(example_id=example.id, code="unknown_label", message=f"{prefix}: {span.label}"))
            continue
        expected_opf = ontology.opf_label_for(span.label)
        if span.opf_label != expected_opf:
            issues.append(
                ValidationIssue(
                    example_id=example.id,
                    code="opf_label_mismatch",
                    message=f"{prefix}: {span.label} maps to {expected_opf}, got {span.opf_label}",
                )
            )
        if span.opf_label not in OPF_LABELS:
            issues.append(ValidationIssue(example_id=example.id, code="unknown_opf_label", message=f"{prefix}: {span.opf_label}"))
        if not (0 <= span.start < span.end <= len(example.text)):
            issues.append(ValidationIssue(example_id=example.id, code="invalid_offsets", message=f"{prefix}: {span.start}:{span.end}"))
            continue
        if span.start < previous_end:
            issues.append(ValidationIssue(example_id=example.id, code="overlap", message=f"{prefix}: overlaps previous span"))
        previous_end = span.end
        actual = example.text[span.start : span.end]
        if actual != span.text:
            issues.append(
                ValidationIssue(
                    example_id=example.id,
                    code="span_text_mismatch",
                    message=f"{prefix}: expected {span.text!r}, got {actual!r}",
                )
            )
        validator_name = ontology.config.labels[span.label].validator
        pattern = VALIDATOR_PATTERNS.get(validator_name or "")
        if pattern and not pattern.match(span.text):
            issues.append(
                ValidationIssue(
                    example_id=example.id,
                    code="format_mismatch",
                    message=f"{prefix}: {span.label} value {span.text!r} failed {validator_name}",
                )
            )
        lowered = span.text.lower()
        for forbidden in policy.forbidden_value_substrings:
            if forbidden.lower() in lowered:
                issues.append(
                    ValidationIssue(
                        example_id=example.id,
                        code="forbidden_value",
                        message=f"{prefix}: value contains forbidden substring {forbidden!r}",
                    )
                )
    return issues


def build_distribution(examples: list[CanonicalExample]) -> dict[str, dict[str, int]]:
    distributions: dict[str, Counter[str]] = {
        "language": Counter(),
        "surface": Counter(),
        "pii_density": Counter(),
        "scenario": Counter(),
        "label": Counter(),
        "opf_label": Counter(),
    }
    for example in examples:
        distributions["language"][str(example.meta.language)] += 1
        distributions["surface"][str(example.meta.surface)] += 1
        distributions["pii_density"][str(example.meta.pii_density)] += 1
        distributions["scenario"][example.meta.scenario] += 1
        for span in example.spans:
            distributions["label"][span.label] += 1
            distributions["opf_label"][span.opf_label] += 1
    return {key: dict(value) for key, value in distributions.items()}


def validate_file(
    input_path: Path,
    output_path: Path | None,
    report_path: Path,
    *,
    ontology: Ontology,
    policy: PolicyConfig,
) -> ValidationReport:
    examples: list[CanonicalExample] = []
    valid_examples: list[CanonicalExample] = []
    issues: list[ValidationIssue] = []
    seen_text: dict[str, str] = {}

    for raw in read_jsonl(input_path):
        try:
            example = CanonicalExample.model_validate(raw)
        except ValidationError as exc:
            issues.append(ValidationIssue(code="schema_error", message=str(exc)))
            continue
        examples.append(example)
        example_issues = validate_example(example, ontology=ontology, policy=policy)
        if example.text in seen_text:
            example_issues.append(
                ValidationIssue(
                    example_id=example.id,
                    code="duplicate_text",
                    message=f"Duplicate of {seen_text[example.text]}",
                )
            )
        else:
            seen_text[example.text] = example.id
        if example_issues:
            issues.extend(example_issues)
        else:
            valid_examples.append(example)

    invalid_ids = {issue.example_id for issue in issues if issue.example_id}
    report = ValidationReport(
        input_path=str(input_path),
        total=len(examples),
        valid=len(valid_examples),
        invalid=len(invalid_ids) + sum(1 for issue in issues if issue.example_id is None),
        issues=issues,
        distributions=build_distribution(valid_examples),
    )
    if output_path:
        write_jsonl(output_path, valid_examples)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
