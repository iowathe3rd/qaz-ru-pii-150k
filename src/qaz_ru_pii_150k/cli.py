from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .config import load_generation_config, load_ontology, load_policy
from .export import export_label_space, export_opf
from .generate import generate_dataset
from .llm import provider_from_name
from .ontology import Ontology
from .report import write_dataset_card
from .split import split_dataset
from .temporal_app.worker import run_worker, start_generation_workflow
from .validators import validate_file

app = typer.Typer(help="Generate and export synthetic KZ RU/KK PII data for OpenAI Privacy Filter.")
console = Console()


ConfigArg = Annotated[Path, typer.Option("--config", exists=True, file_okay=True, dir_okay=False)]


@app.command("generate")
def generate_command(
    config: ConfigArg = Path("configs/generation_mix_150k.yaml"),
    out: Annotated[Path | None, typer.Option("--out")] = None,
    provider: Annotated[str, typer.Option("--provider", help="'openai' or 'fake'")] = "openai",
) -> None:
    generation_config = load_generation_config(config)
    ontology = Ontology(load_ontology())
    template_provider = provider_from_name(provider, generation_config.openai)
    output_path = out or generation_config.output.canonical_raw
    count = generate_dataset(
        config=generation_config,
        provider=template_provider,
        ontology=ontology,
        output_path=output_path,
    )
    console.print(f"Generated {count} canonical examples -> {output_path}")


@app.command("temporal-worker")
def temporal_worker_command(config: ConfigArg = Path("configs/generation_mix_150k.yaml")) -> None:
    asyncio.run(run_worker(config))


@app.command("generate-temporal")
def generate_temporal_command(
    config: ConfigArg = Path("configs/generation_mix_150k.yaml"),
    provider: Annotated[str, typer.Option("--provider", help="'openai' or 'fake'")] = "openai",
) -> None:
    workflow_id = asyncio.run(start_generation_workflow(config, provider=provider))
    console.print(f"Started Temporal workflow {workflow_id}")


@app.command("validate")
def validate_command(
    input: Annotated[Path, typer.Option("--input", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    report: Annotated[Path, typer.Option("--report")] = Path("reports/validation_v1.json"),
) -> None:
    ontology = Ontology(load_ontology())
    policy = load_policy()
    validation_report = validate_file(input, out, report, ontology=ontology, policy=policy)
    console.print(
        f"Validated {validation_report.total} examples: "
        f"{validation_report.valid} valid, {len(validation_report.issues)} issues -> {report}"
    )
    if validation_report.issues:
        raise typer.Exit(code=1)


@app.command("split")
def split_command(
    input: Annotated[Path, typer.Option("--input", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out")] = Path("data/splits"),
    seed: Annotated[int, typer.Option("--seed")] = 20260523,
) -> None:
    counts = split_dataset(input, out, seed=seed)
    console.print(f"Wrote splits to {out}: {counts}")


@app.command("export-opf")
def export_opf_command(
    input: Annotated[Path, typer.Option("--input", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out")],
    mode: Annotated[str, typer.Option("--mode")] = "opf8",
) -> None:
    count = export_opf(input, out, mode=mode)
    console.print(f"Exported {count} {mode} records -> {out}")


@app.command("export-label-space")
def export_label_space_command(out: Annotated[Path, typer.Option("--out")] = Path("configs/label_space_kz_custom.json")) -> None:
    ontology = Ontology(load_ontology())
    export_label_space(out, ontology=ontology)
    console.print(f"Wrote custom label-space -> {out}")


@app.command("opf-train-command")
def opf_train_command(
    train: Annotated[Path, typer.Option("--train")],
    val: Annotated[Path, typer.Option("--val")],
    out: Annotated[Path, typer.Option("--out")],
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
) -> None:
    pieces = ["opf", "train", str(train), "--validation-dataset", str(val), "--output-dir", str(out)]
    if checkpoint:
        pieces.extend(["--checkpoint", str(checkpoint)])
    console.print(" ".join(pieces))


@app.command("report")
def report_command(
    validation: Annotated[Path, typer.Option("--validation", exists=True, file_okay=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out")] = Path("reports/dataset_card_v1.md"),
) -> None:
    write_dataset_card(validation, out)
    console.print(f"Wrote dataset card -> {out}")
