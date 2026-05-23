from __future__ import annotations

import re

from .constants import DEFAULT_GENERATOR_VERSION
from .models import CanonicalExample, ExampleMeta, LlmTemplateRecord, Span
from .ontology import Ontology

MARKER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*?)\}\}")


def marker_names(template: str) -> list[str]:
    return [match.group(1) for match in MARKER_RE.finditer(template)]


def inject_record(
    record: LlmTemplateRecord,
    *,
    ontology: Ontology,
    example_id: str,
    generator_version: str = DEFAULT_GENERATOR_VERSION,
    raw_record_id: str | None = None,
    shard_id: int | None = None,
) -> CanonicalExample:
    markers = marker_names(record.template)
    marker_set = set(markers)
    value_set = set(record.values)

    missing = marker_set - value_set
    if missing:
        raise ValueError(f"Template is missing values for markers: {sorted(missing)}")
    unused = value_set - marker_set
    if unused:
        raise ValueError(f"Values contain markers unused by template: {sorted(unused)}")
    duplicates = sorted({marker for marker in markers if markers.count(marker) > 1})
    if duplicates:
        raise ValueError(f"Duplicate exact markers are ambiguous: {duplicates}")

    parts: list[str] = []
    spans: list[Span] = []
    output_position = 0
    cursor = 0

    for match in MARKER_RE.finditer(record.template):
        literal = record.template[cursor : match.start()]
        parts.append(literal)
        output_position += len(literal)

        marker = match.group(1)
        llm_value = record.values[marker]
        ontology.validate_marker_label(marker, llm_value.label)
        opf_label = ontology.opf_label_for(llm_value.label)

        start = output_position
        value = llm_value.value
        parts.append(value)
        output_position += len(value)
        end = output_position
        spans.append(
            Span(
                start=start,
                end=end,
                text=value,
                label=llm_value.label,
                opf_label=opf_label,
            )
        )
        cursor = match.end()

    tail = record.template[cursor:]
    parts.append(tail)
    text = "".join(parts)

    meta = ExampleMeta(
        **record.meta.model_dump(mode="json"),
        generator_version=generator_version,
        raw_record_id=raw_record_id,
        shard_id=shard_id,
    )
    example = CanonicalExample(id=example_id, text=text, spans=spans, meta=meta)
    for span in example.spans:
        if example.text[span.start : span.end] != span.text:
            raise ValueError(f"Internal offset error for {example.id}: {span}")
    return example
