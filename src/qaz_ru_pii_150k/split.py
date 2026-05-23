from __future__ import annotations

import random
from pathlib import Path

from .io import read_jsonl, write_jsonl


def split_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, int]:
    records = list(read_jsonl(input_path))
    random.Random(seed).shuffle(records)
    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    return {name: write_jsonl(output_dir / f"{name}.jsonl", split_records) for name, split_records in splits.items()}
