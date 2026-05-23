from __future__ import annotations

from pathlib import Path

from qaz_ru_pii_150k.config import load_generation_config, load_ontology, load_policy
from qaz_ru_pii_150k.export import export_opf
from qaz_ru_pii_150k.generate import generate_dataset
from qaz_ru_pii_150k.llm import FakeTemplateProvider
from qaz_ru_pii_150k.ontology import Ontology
from qaz_ru_pii_150k.split import split_dataset
from qaz_ru_pii_150k.validators import validate_file


def test_tiny_pipeline(tmp_path: Path) -> None:
    config = load_generation_config(Path("configs/generation_mix_tiny.yaml"))
    config.output.raw_llm_dir = tmp_path / "raw_llm"
    raw = tmp_path / "raw.jsonl"
    validated = tmp_path / "validated.jsonl"
    report = tmp_path / "report.json"
    splits = tmp_path / "splits"
    opf = tmp_path / "opf_train.jsonl"
    ontology = Ontology(load_ontology(Path("configs/ontology.yaml")))

    count = generate_dataset(config=config, provider=FakeTemplateProvider(), ontology=ontology, output_path=raw)
    assert count == 12

    validation_report = validate_file(
        raw,
        validated,
        report,
        ontology=ontology,
        policy=load_policy(Path("configs/policy.yaml")),
    )
    assert validation_report.valid == 12
    assert not validation_report.issues

    split_counts = split_dataset(validated, splits, seed=1)
    assert split_counts == {"train": 9, "val": 1, "test": 2}

    exported = export_opf(splits / "train.jsonl", opf, mode="opf8")
    assert exported == 9
