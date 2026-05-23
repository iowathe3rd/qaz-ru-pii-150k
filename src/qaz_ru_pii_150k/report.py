from __future__ import annotations

import json
from pathlib import Path


def write_dataset_card(validation_report_path: Path, output_path: Path) -> None:
    report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    lines = [
        "# KZ RU/KK Synthetic PII Dataset Card",
        "",
        f"- Total examples: {report.get('total', 0)}",
        f"- Valid examples: {report.get('valid', 0)}",
        f"- Invalid examples: {report.get('invalid', 0)}",
        "- Source: synthetic only",
        "- Intended use: OpenAI Privacy Filter local fine-tuning and evaluation",
        "- Warning: synthetic data is not a privacy or compliance guarantee",
        "",
        "## Distributions",
        "",
    ]
    for section, values in report.get("distributions", {}).items():
        lines.append(f"### {section}")
        for key, value in sorted(values.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
