from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .io import ensure_parent, read_jsonl
from .models import CanonicalExample
from .ontology import Ontology


def to_opf_record(example: CanonicalExample, *, mode: str) -> dict[str, object]:
    if mode not in {"opf8", "custom"}:
        raise ValueError("mode must be 'opf8' or 'custom'")
    spans: dict[str, list[list[int]]] = defaultdict(list)
    for span in example.spans:
        label = span.opf_label if mode == "opf8" else span.label
        key = f"{label}: {span.text}"
        spans[key].append([span.start, span.end])
    return {
        "text": example.text,
        "spans": dict(spans),
        "info": {
            "id": example.id,
            "source": "qaz-ru-pii-150k",
            "language": str(example.meta.language),
            "surface": str(example.meta.surface),
            "scenario": example.meta.scenario,
            "mode": mode,
        },
    }


def export_opf(input_path: Path, output_path: Path, *, mode: str) -> int:
    ensure_parent(output_path)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for raw in read_jsonl(input_path):
            example = CanonicalExample.model_validate(raw)
            handle.write(json.dumps(to_opf_record(example, mode=mode), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def export_label_space(path: Path, *, ontology: Ontology) -> None:
    labels = ["O", *sorted(ontology.fine_labels)]
    payload = {"category_version": "kz_banking_v1", "span_class_names": labels}
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
