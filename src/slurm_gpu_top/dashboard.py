from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple

from .commands import CommandRunner, run_command
from .gpu import poll_node
from .models import ClusterSnapshot, HostStats, NodeSnapshot, SlurmJob, SnapshotBuilderConfig
from .slurm import SlurmError, discover_gpu_jobs


def build_snapshot(
    *,
    config: SnapshotBuilderConfig,
    runner: CommandRunner = run_command,
) -> ClusterSnapshot:
    now = config.now or time.time()
    try:
        jobs = discover_gpu_jobs(
            user=config.user,
            all_users=config.all_users,
            runner=runner,
            timeout=config.command_timeout_s,
        )
    except SlurmError as exc:
        return ClusterSnapshot(errors=(str(exc),), generated_at=now, user_filter=config.user)

    jobs_by_node = _jobs_by_node(jobs)
    if not jobs_by_node:
        return ClusterSnapshot(generated_at=now, user_filter=config.user)

    snapshots: Dict[str, NodeSnapshot] = {}
    workers = max(1, min(config.max_workers, len(jobs_by_node)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                poll_node,
                node,
                ssh_options=config.ssh_options,
                runner=runner,
                timeout=config.command_timeout_s,
            ): node
            for node in jobs_by_node
        }
        for future in as_completed(futures):
            node = futures[future]
            try:
                gpus, host, error = future.result()
            except Exception as exc:  # defensive: keep the dashboard alive
                gpus, host, error = (), HostStats(hostname=node), str(exc)
            snapshots[node] = NodeSnapshot(
                node=node,
                jobs=tuple(sorted(jobs_by_node[node], key=lambda job: job.job_id)),
                gpus=gpus,
                host=host,
                error=error,
            )

    ordered = tuple(snapshots[node] for node in sorted(snapshots))
    return ClusterSnapshot(nodes=ordered, generated_at=now, user_filter=config.user)


def _jobs_by_node(jobs: Iterable[SlurmJob]) -> Dict[str, List[SlurmJob]]:
    grouped: Dict[str, List[SlurmJob]] = {}
    for job in jobs:
        for node in job.nodes:
            grouped.setdefault(node, []).append(job)
    return grouped
