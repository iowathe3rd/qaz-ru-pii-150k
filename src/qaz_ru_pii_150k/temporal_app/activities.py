from __future__ import annotations

from pathlib import Path

from temporalio import activity

from ..config import load_generation_config, load_ontology
from ..generate import generate_shard, slices_for_config
from ..io import write_jsonl
from ..llm import provider_from_name
from ..ontology import Ontology


@activity.defn
async def generate_shard_activity(
    config_path: str,
    shard_id: int,
    provider_name: str,
    canonical_shard_dir: str,
) -> str:
    config = load_generation_config(Path(config_path))
    ontology = Ontology(load_ontology())
    generation_slice = slices_for_config(config)[shard_id]
    provider = provider_from_name(provider_name, config.openai)
    records = generate_shard(
        config=config,
        generation_slice=generation_slice,
        shard_id=shard_id,
        provider=provider,
        ontology=ontology,
        raw_dir=config.output.raw_llm_dir,
    )
    shard_path = Path(canonical_shard_dir) / f"canonical_{shard_id:06d}.jsonl"
    write_jsonl(shard_path, records)
    return str(shard_path)
