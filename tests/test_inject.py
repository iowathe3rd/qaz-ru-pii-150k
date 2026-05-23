from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from qaz_ru_pii_150k.config import load_ontology
from qaz_ru_pii_150k.inject import inject_record
from qaz_ru_pii_150k.models import LlmTemplateRecord
from qaz_ru_pii_150k.ontology import Ontology


def ontology() -> Ontology:
    return Ontology(load_ontology(Path("configs/ontology.yaml")))


def test_injects_cyrillic_kazakh_offsets() -> None:
    record = LlmTemplateRecord.model_validate(
        {
            "template": "сәлеметсіз бе менің жсн {{IIN_1}} атым {{NAME_1}}",
            "values": {
                "IIN_1": {"label": "iin_kz", "value": "010203500123"},
                "NAME_1": {"label": "person_name", "value": "Нұрлан Әбдірахманұлы"},
            },
            "meta": {
                "language": "kk",
                "surface": "call_center_asr",
                "scenario": "identity_verification",
                "pii_density": "medium",
                "noise_type": "clean",
                "hard_negative": False,
            },
        }
    )

    example = inject_record(record, ontology=ontology(), example_id="ex1")

    assert example.text == "сәлеметсіз бе менің жсн 010203500123 атым Нұрлан Әбдірахманұлы"
    for span in example.spans:
        assert example.text[span.start : span.end] == span.text
    assert [span.label for span in example.spans] == ["iin_kz", "person_name"]


def test_rejects_unused_values() -> None:
    record = LlmTemplateRecord.model_validate(
        {
            "template": "алло {{NAME_1}}",
            "values": {
                "NAME_1": {"label": "person_name", "value": "Айжан Сейдахметова"},
                "IIN_1": {"label": "iin_kz", "value": "010203500123"},
            },
            "meta": {
                "language": "ru",
                "surface": "call_center_asr",
                "scenario": "identity_verification",
                "pii_density": "low",
                "noise_type": "clean",
                "hard_negative": False,
            },
        }
    )
    with pytest.raises(ValueError, match="unused"):
        inject_record(record, ontology=ontology(), example_id="ex2")


def test_rejects_marker_label_mismatch() -> None:
    record = LlmTemplateRecord.model_validate(
        {
            "template": "мой иин {{IIN_1}}",
            "values": {"IIN_1": {"label": "person_name", "value": "Айжан"}},
            "meta": {
                "language": "ru",
                "surface": "call_center_asr",
                "scenario": "identity_verification",
                "pii_density": "low",
                "noise_type": "clean",
                "hard_negative": False,
            },
        }
    )
    with pytest.raises(ValueError, match="implies label"):
        inject_record(record, ontology=ontology(), example_id="ex3")


def test_rejects_llm_offsets_in_values() -> None:
    with pytest.raises(ValidationError):
        LlmTemplateRecord.model_validate(
            {
                "template": "алло {{NAME_1}}",
                "values": {
                    "NAME_1": {
                        "label": "person_name",
                        "value": "Айжан Сейдахметова",
                        "start": 5,
                        "end": 23,
                    }
                },
                "meta": {
                    "language": "ru",
                    "surface": "call_center_asr",
                    "scenario": "identity_verification",
                    "pii_density": "low",
                    "noise_type": "clean",
                    "hard_negative": False,
                },
            }
        )
