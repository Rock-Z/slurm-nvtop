from __future__ import annotations

import os
import time
from typing import Iterable, Optional

from .models import ClusterSnapshot, GPUDevice, GPUProcess, NodeSnapshot, SlurmJob


def render_snapshot(snapshot: ClusterSnapshot, *, width: Optional[int] = None) -> str:
    width = width or _terminal_width()
    lines = [
        _clip(
            "slurm-gpu-top  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snapshot.generated_at))}  "
            f"jobs={snapshot.job_count} nodes={len(snapshot.nodes)} gpus={snapshot.gpu_count}",
            width,
        ),
        "=" * min(width, 120),
    ]

    if snapshot.errors:
        lines.extend(_clip(f"Slurm error: {error}", width) for error in snapshot.errors)
        return "\n".join(lines)

    if not snapshot.nodes:
        lines.append("No running GPU-backed Slurm jobs found.")
        return "\n".join(lines)

    for node in snapshot.nodes:
        lines.extend(_render_node(node, width))
    return "\n".join(lines)


def _render_node(node: NodeSnapshot, width: int) -> Iterable[str]:
    yield ""
    yield _clip(f"{node.node}  jobs: {_format_jobs(node.jobs)}", width)
    yield "-" * min(width, 120)
    if node.error:
        yield _clip(f"  ! {node.error}", width)
        return
    if not node.gpus:
        yield "  No GPUs reported by nvidia-smi."
        return

    header = (
        "  GPU  Name                         GPU%       MEM             TEMP      POWER"
    )
    yield _clip(header, width)
    for gpu in sorted(node.gpus, key=lambda item: item.index):
        yield _clip(_format_gpu(gpu), width)
        for proc in sorted(gpu.processes, key=lambda item: item.pid):
            yield _clip(_format_process(proc), width)


def _format_jobs(jobs: Iterable[SlurmJob]) -> str:
    parts = []
    for job in jobs:
        label = f"{job.job_id} {job.user}/{job.name} {job.elapsed}"
        parts.append(label)
    return ", ".join(parts) if parts else "unknown"


def _format_gpu(gpu: GPUDevice) -> str:
    util = _percent(gpu.gpu_util_percent)
    mem_util = _percent(gpu.mem_util_percent)
    mem = _memory(gpu.mem_used_mib, gpu.mem_total_mib)
    temp = _value(gpu.temperature_c, "C")
    power = _power(gpu.power_draw_w, gpu.power_limit_w)
    bar = _bar(gpu.gpu_util_percent)
    return (
        f"  {gpu.index:<3}  {gpu.name[:28]:<28} "
        f"{util:>5} {bar}  {mem_util:>5} {mem:>15}  {temp:>7}  {power:>14}"
    )


def _format_process(proc: GPUProcess) -> str:
    mem = _value(proc.used_memory_mib, "MiB")
    return f"       pid={proc.pid:<7} mem={mem:>9}  {proc.name}"


def _percent(value: Optional[int]) -> str:
    return "  N/A" if value is None else f"{value:>3}%"


def _memory(used: Optional[int], total: Optional[int]) -> str:
    if used is None or total is None:
        return "N/A"
    return f"{used}/{total} MiB"


def _power(draw: Optional[float], limit: Optional[float]) -> str:
    if draw is None and limit is None:
        return "N/A"
    if limit is None:
        return f"{draw:.0f}W"
    if draw is None:
        return f"N/A/{limit:.0f}W"
    return f"{draw:.0f}/{limit:.0f}W"


def _value(value: object, unit: str) -> str:
    return "N/A" if value is None else f"{value}{unit}"


def _bar(value: Optional[int], *, width: int = 10) -> str:
    if value is None:
        return "[" + "?" * width + "]"
    filled = max(0, min(width, round(width * value / 100)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _clip(text: str, width: int) -> str:
    if width <= 1 or len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 120
