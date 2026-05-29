from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

from .commands import CommandRunner, run_command
from .models import GPUDevice, GPUProcess

PROCESS_MARKER = "__SLURM_GPU_TOP_PROCESSES__"

REMOTE_NVIDIA_SMI_QUERY = (
    "nvidia-smi "
    "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,"
    "memory.total,temperature.gpu,power.draw,power.limit "
    "--format=csv,noheader,nounits; "
    f"printf '\\n{PROCESS_MARKER}\\n'; "
    "nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid "
    "--format=csv,noheader,nounits 2>/dev/null || true"
)


def poll_node_gpus(
    node: str,
    *,
    ssh_options: Sequence[str] = ("BatchMode=yes", "ConnectTimeout=5"),
    runner: CommandRunner = run_command,
    timeout: float = 8.0,
) -> Tuple[Tuple[GPUDevice, ...], Optional[str]]:
    ssh_args = ["ssh"]
    for option in ssh_options:
        ssh_args.extend(["-o", option])
    ssh_args.extend([node, REMOTE_NVIDIA_SMI_QUERY])

    result = runner(ssh_args, timeout)
    if not result.ok:
        error = result.stderr.strip() or result.stdout.strip() or "GPU query failed"
        if result.timed_out:
            error = f"timed out after {timeout:g}s: {error}"
        return (), error

    try:
        return parse_nvidia_smi_output(node, result.stdout), None
    except ValueError as exc:
        return (), str(exc)


def parse_nvidia_smi_output(node: str, output: str) -> Tuple[GPUDevice, ...]:
    if PROCESS_MARKER in output:
        gpu_text, process_text = output.split(PROCESS_MARKER, 1)
    else:
        gpu_text, process_text = output, ""

    processes_by_uuid: Dict[str, List[GPUProcess]] = {}
    for row in _csv_rows(process_text):
        if len(row) < 4:
            continue
        pid = _parse_int(row[0])
        if pid is None:
            continue
        proc = GPUProcess(
            pid=pid,
            name=row[1].strip(),
            used_memory_mib=_parse_int(row[2]),
            gpu_uuid=row[3].strip(),
        )
        processes_by_uuid.setdefault(proc.gpu_uuid, []).append(proc)

    gpus: List[GPUDevice] = []
    for row in _csv_rows(gpu_text):
        if len(row) < 10:
            raise ValueError(f"unexpected nvidia-smi GPU row with {len(row)} fields: {row!r}")
        index = _parse_int(row[0])
        if index is None:
            raise ValueError(f"GPU index is not an integer: {row[0]!r}")
        uuid = row[1].strip()
        gpus.append(
            GPUDevice(
                node=node,
                index=index,
                uuid=uuid,
                name=row[2].strip(),
                gpu_util_percent=_parse_int(row[3]),
                mem_util_percent=_parse_int(row[4]),
                mem_used_mib=_parse_int(row[5]),
                mem_total_mib=_parse_int(row[6]),
                temperature_c=_parse_int(row[7]),
                power_draw_w=_parse_float(row[8]),
                power_limit_w=_parse_float(row[9]),
                processes=tuple(processes_by_uuid.get(uuid, ())),
            )
        )
    return tuple(gpus)


def _csv_rows(text: str) -> List[List[str]]:
    rows = []
    reader = csv.reader(StringIO(text.strip()))
    for row in reader:
        if not row:
            continue
        if len(row) == 1 and not row[0].strip():
            continue
        rows.append([item.strip() for item in row])
    return rows


def _parse_int(value: object) -> Optional[int]:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
