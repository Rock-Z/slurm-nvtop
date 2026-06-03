from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class SlurmJob:
    job_id: str
    name: str
    user: str
    state: str
    elapsed: str
    node_count: Optional[int]
    nodelist: str
    gres: str = ""
    tres: str = ""
    nodes: Tuple[str, ...] = ()

    @property
    def gpu_hint(self) -> bool:
        haystack = f"{self.gres} {self.tres}".lower()
        return "gpu" in haystack


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    name: str
    used_memory_mib: Optional[int]
    gpu_uuid: str
    slurm_job_id: str = ""
    type: str = "C"
    user: str = ""
    cpu_percent: Optional[float] = None
    mem_percent: Optional[float] = None
    elapsed: str = ""
    command: str = ""
    sm_util_percent: Optional[int] = None
    mem_bw_util_percent: Optional[int] = None


@dataclass(frozen=True)
class GPUDevice:
    node: str
    index: int
    uuid: str
    name: str
    gpu_util_percent: Optional[int]
    mem_util_percent: Optional[int]
    mem_used_mib: Optional[int]
    mem_total_mib: Optional[int]
    temperature_c: Optional[int]
    power_draw_w: Optional[float]
    power_limit_w: Optional[float]
    slurm_job_id: str = ""
    persistence_mode: str = ""
    pci_bus_id: str = ""
    display_active: str = ""
    mig_mode: str = ""
    ecc_errors: Optional[int] = None
    fan_speed_percent: Optional[int] = None
    performance_state: str = ""
    compute_mode: str = ""
    sm_clock_mhz: Optional[int] = None
    processes: Tuple[GPUProcess, ...] = ()


@dataclass(frozen=True)
class HostStats:
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_mib: Optional[int] = None
    memory_total_mib: Optional[int] = None
    swap_percent: Optional[float] = None
    swap_used_mib: Optional[int] = None
    swap_total_mib: Optional[int] = None
    load_average: Tuple[float, float, float] = ()
    uptime_seconds: Optional[float] = None
    hostname: str = ""
    driver_version: str = ""
    cuda_version: str = ""


@dataclass(frozen=True)
class NodeSnapshot:
    node: str
    jobs: Tuple[SlurmJob, ...] = ()
    gpus: Tuple[GPUDevice, ...] = ()
    host: HostStats = field(default_factory=HostStats)
    error: Optional[str] = None


@dataclass(frozen=True)
class ClusterSnapshot:
    nodes: Tuple[NodeSnapshot, ...] = ()
    errors: Tuple[str, ...] = ()
    generated_at: float = 0.0
    user_filter: Optional[str] = None

    @property
    def gpu_count(self) -> int:
        return sum(len(node.gpus) for node in self.nodes)

    @property
    def job_count(self) -> int:
        seen = set()
        for node in self.nodes:
            for job in node.jobs:
                seen.add(job.job_id)
        return len(seen)


@dataclass(frozen=True)
class CommandResult:
    args: Tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass
class SnapshotBuilderConfig:
    user: Optional[str] = None
    all_users: bool = False
    command_timeout_s: float = 8.0
    max_workers: int = 16
    now: float = field(default=0.0)
