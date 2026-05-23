from __future__ import annotations

from pathlib import Path

from qaz_ru_pii_150k.config import load_ontology
from qaz_ru_pii_150k.export import to_opf_record
from qaz_ru_pii_150k.inject import inject_record
from qaz_ru_pii_150k.models import LlmTemplateRecord
from qaz_ru_pii_150k.ontology import Ontology


def test_opf_export_mapping_shape() -> None:
    record = LlmTemplateRecord.model_validate(
        {
            "template": "меня зовут {{NAME_1}} мой иин {{IIN_1}}",
            "values": {
                "NAME_1": {"label": "person_name", "value": "Айжан Сейдахметова"},
                "IIN_1": {"label": "iin_kz", "value": "010203500123"},
            },
            "meta": {
                "language": "ru",
                "surface": "call_center_asr",
                "scenario": "identity_verification",
                "pii_density": "medium",
                "noise_type": "clean",
                "hard_negative": False,
            },
        }
    )
    example = inject_record(record, ontology=Ontology(load_ontology(Path("configs/ontology.yaml"))), example_id="ex1")

    opf = to_opf_record(example, mode="opf8")

    assert opf["text"] == example.text
    assert opf["spans"] == {
        "private_person: Айжан Сейдахметова": [[11, 29]],
        "account_number: 010203500123": [[38, 50]],
    }
