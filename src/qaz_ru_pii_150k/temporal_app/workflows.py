from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .activities import generate_shard_activity


@workflow.defn
class GenerateShardListWorkflow:
    @workflow.run
    async def run(
        self,
        shard_count: int,
        config_path: str,
        provider_name: str = "openai",
        canonical_shard_dir: str = "data/work/shards",
    ) -> list[str]:
        paths: list[str] = []
        for shard_id in range(shard_count):
            path = await workflow.execute_activity(
                generate_shard_activity,
                args=[config_path, shard_id, provider_name, canonical_shard_dir],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=None,
            )
            paths.append(path)
        return paths
