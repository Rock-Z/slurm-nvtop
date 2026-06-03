from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple

from .commands import CommandRunner, run_command
from .gpu import poll_job
from .models import ClusterSnapshot, GPUDevice, HostStats, NodeSnapshot, SlurmJob, SnapshotBuilderConfig
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

    # The probe unit is the job, not the node: each job's GPUs are only visible from
    # inside that job's cgroup, so a node hosting several of the user's jobs needs one
    # probe per job to be seen in full.
    probes = [(job, node) for job in jobs for node in job.nodes]
    if not probes:
        return ClusterSnapshot(generated_at=now, user_filter=config.user)

    builders: Dict[str, _NodeBuilder] = {}
    for job in jobs:
        for node in job.nodes:
            builders.setdefault(node, _NodeBuilder(node)).jobs.append(job)

    workers = max(1, min(config.max_workers, len(probes)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                poll_job,
                job.job_id,
                node,
                runner=runner,
                timeout=config.command_timeout_s,
            ): (job, node)
            for job, node in probes
        }
        for future in as_completed(futures):
            job, node = futures[future]
            try:
                gpus, host, error = future.result()
            except Exception as exc:  # defensive: keep the dashboard alive
                gpus, host, error = (), HostStats(hostname=node), str(exc)
            builders[node].add(job, gpus, host, error)

    ordered = tuple(builders[node].build() for node in sorted(builders))
    return ClusterSnapshot(nodes=ordered, generated_at=now, user_filter=config.user)


class _NodeBuilder:
    """Accumulates the per-job probe results that land on a single node."""

    def __init__(self, node: str) -> None:
        self.node = node
        self.jobs: List[SlurmJob] = []
        self._gpus: Dict[str, GPUDevice] = {}
        self._host = HostStats(hostname=node)
        self._failures: List[Tuple[str, str]] = []

    def add(
        self,
        job: SlurmJob,
        gpus: Iterable[GPUDevice],
        host: HostStats,
        error: object,
    ) -> None:
        added = False
        for gpu in gpus:
            # Each job sees a disjoint set of GPUs; key by uuid so a GPU shared by
            # two jobs (MPS/sharding) is merged rather than duplicated.
            self._gpus.setdefault(gpu.uuid, gpu)
            added = True
        if host.driver_version or host.cuda_version or host.cpu_percent is not None:
            self._host = host
        if error and not added:
            self._failures.append((job.job_id, str(error)))

    def build(self) -> NodeSnapshot:
        gpus = tuple(sorted(self._gpus.values(), key=lambda gpu: gpu.index))
        return NodeSnapshot(
            node=self.node,
            jobs=tuple(sorted(self.jobs, key=lambda job: job.job_id)),
            gpus=gpus,
            host=self._host,
            error=self._node_error(bool(gpus)),
        )

    def _node_error(self, has_gpus: bool) -> object:
        # Surface a node-level error only when nothing was observed; otherwise the
        # renderer would hide the GPUs we did manage to probe. With a lone failing
        # job the bare message reads best; name the jobs only when several failed.
        if has_gpus or not self._failures:
            return None
        if len(self._failures) == 1:
            return self._failures[0][1]
        return "; ".join(f"job {job_id}: {message}" for job_id, message in self._failures)
