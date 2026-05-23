from __future__ import annotations

import random
from pathlib import Path

from .config import GenerationConfig
from .inject import inject_record
from .io import read_jsonl, write_jsonl
from .llm import TemplateProvider
from .models import GenerationSlice, LlmTemplateRecord
from .ontology import Ontology


def choose_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    cursor = rng.random() * total
    upto = 0.0
    for key, weight in weights.items():
        upto += weight
        if upto >= cursor:
            return key
    return next(reversed(weights))


def slices_for_config(config: GenerationConfig) -> list[GenerationSlice]:
    rng = random.Random(config.seed)
    slices: list[GenerationSlice] = []
    remaining = config.target_count
    while remaining > 0:
        count = min(config.shard_size, remaining)
        surface = choose_weighted(rng, config.surfaces)
        density = choose_weighted(rng, config.pii_density)
        hard_negative = surface == "hard_negative"
        if hard_negative:
            density = "none"
        slices.append(
            GenerationSlice.model_validate(
                {
                    "count": count,
                    "language": choose_weighted(rng, config.languages),
                    "surface": surface,
                    "scenario": rng.choice(config.scenarios),
                    "pii_density": density,
                    "noise_type": rng.choice(config.noise_types),
                    "hard_negative": hard_negative,
                }
            )
        )
        remaining -= count
    return slices


def generate_shard(
    *,
    config: GenerationConfig,
    generation_slice: GenerationSlice,
    shard_id: int,
    provider: TemplateProvider,
    ontology: Ontology,
    raw_dir: Path,
) -> list[dict[str, object]]:
    raw_path = raw_dir / f"raw_llm_{shard_id:06d}.jsonl"
    if raw_path.exists():
        records = [LlmTemplateRecord.model_validate(raw) for raw in read_jsonl(raw_path)]
    else:
        records = provider.generate(generation_slice, seed=config.seed + shard_id)
        write_jsonl(raw_path, records)

    examples = []
    for idx, record in enumerate(records):
        example = inject_record(
            record,
            ontology=ontology,
            example_id=f"kz_bank_{shard_id:06d}_{idx:04d}",
            generator_version=config.generator_version,
            raw_record_id=f"raw_llm_{shard_id:06d}:{idx}",
            shard_id=shard_id,
        )
        examples.append(example.model_dump(mode="json"))
    return examples


def generate_dataset(
    *,
    config: GenerationConfig,
    provider: TemplateProvider,
    ontology: Ontology,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = config.output.raw_llm_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for shard_id, generation_slice in enumerate(slices_for_config(config)):
            examples = generate_shard(
                config=config,
                generation_slice=generation_slice,
                shard_id=shard_id,
                provider=provider,
                ontology=ontology,
                raw_dir=raw_dir,
            )
            for example in examples:
                import json

                handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
    return count
