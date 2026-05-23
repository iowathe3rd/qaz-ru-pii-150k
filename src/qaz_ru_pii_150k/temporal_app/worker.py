from __future__ import annotations

from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker

from ..config import load_generation_config
from .activities import generate_shard_activity
from .workflows import GenerateShardListWorkflow


async def run_worker(config_path: Path) -> None:
    config = load_generation_config(config_path)
    client = await Client.connect(config.temporal.address, namespace=config.temporal.namespace)
    worker = Worker(
        client,
        task_queue=config.temporal.task_queue,
        workflows=[GenerateShardListWorkflow],
        activities=[generate_shard_activity],
        max_concurrent_activities=config.temporal.max_concurrent_activities,
    )
    await worker.run()


async def start_generation_workflow(config_path: Path, *, provider: str) -> str:
    config = load_generation_config(config_path)
    client = await Client.connect(config.temporal.address, namespace=config.temporal.namespace)
    shard_count = (config.target_count + config.shard_size - 1) // config.shard_size
    handle = await client.start_workflow(
        GenerateShardListWorkflow.run,
        args=[shard_count, str(config_path), provider, "data/work/shards"],
        id=f"qaz-ru-pii-150k-{config.seed}",
        task_queue=config.temporal.task_queue,
    )
    return handle.id
