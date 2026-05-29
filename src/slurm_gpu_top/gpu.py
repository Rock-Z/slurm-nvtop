from __future__ import annotations

import csv
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

from .commands import CommandRunner, run_command
from .models import GPUDevice, GPUProcess, HostStats

META_MARKER = "__SLURM_GPU_TOP_META__"
GPU_MARKER = "__SLURM_GPU_TOP_GPUS__"
PROCESS_MARKER = "__SLURM_GPU_TOP_PROCESSES__"
PMON_MARKER = "__SLURM_GPU_TOP_PMON__"
PS_MARKER = "__SLURM_GPU_TOP_PS__"

REMOTE_NVIDIA_SMI_QUERY = (
    f"printf '{META_MARKER}\\n'; "
    "printf 'hostname='; (hostname -f 2>/dev/null || hostname) | head -n1; "
    "printf 'driver_version='; "
    "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -n1; "
    "printf 'cuda_version='; "
    "nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\\([^ |]*\\).*/\\1/p' | head -n1; "
    "awk 'BEGIN{getline < \"/proc/uptime\"; printf \"uptime_seconds=%s\\n\", $1}' 2>/dev/null; "
    "awk 'BEGIN{getline < \"/proc/loadavg\"; printf \"load_average=%s %s %s\\n\", $1, $2, $3}' 2>/dev/null; "
    "awk '/^MemTotal:/{mt=$2}/^MemAvailable:/{ma=$2}/^SwapTotal:/{st=$2}/^SwapFree:/{sf=$2}"
    "END{if(mt>0) printf \"memory=%d %d %.1f\\n\", (mt-ma)/1024, mt/1024, 100*(mt-ma)/mt; "
    "if(st>0) printf \"swap=%d %d %.1f\\n\", (st-sf)/1024, st/1024, 100*(st-sf)/st; "
    "else print \"swap=0 0 0.0\"}' /proc/meminfo 2>/dev/null; "
    "awk 'BEGIN{getline; i1=$5; t1=0; for(i=2;i<=NF;i++) t1+=$i; close(\"/proc/stat\"); "
    "system(\"sleep 0.1\"); getline < \"/proc/stat\"; i2=$5; t2=0; for(i=2;i<=NF;i++) t2+=$i; "
    "if(t2>t1) printf \"cpu_percent=%.1f\\n\", 100*(1-(i2-i1)/(t2-t1));}' /proc/stat 2>/dev/null; "
    f"printf '\\n{GPU_MARKER}\\n'; "
    "nvidia-smi --query-gpu=index,uuid,name,persistence_mode,pci.bus_id,display_active,"
    "mig.mode.current,ecc.errors.uncorrected.volatile.total,fan.speed,temperature.gpu,pstate,"
    "power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory,"
    "compute_mode,clocks.current.sm --format=csv,noheader,nounits 2>/dev/null || "
    "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,"
    "memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits; "
    f"printf '\\n{PROCESS_MARKER}\\n'; "
    "nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid "
    "--format=csv,noheader,nounits 2>/dev/null || true; "
    f"printf '\\n{PMON_MARKER}\\n'; "
    "nvidia-smi pmon -c 1 -s um 2>/dev/null || true; "
    f"printf '\\n{PS_MARKER}\\n'; "
    "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | "
    "awk 'NF{printf \"%s,\", $1}' | sed 's/,$//'); "
    "if [ -n \"$pids\" ]; then ps -o pid=,user=,pcpu=,pmem=,etime=,args= -p \"$pids\" 2>/dev/null; fi"
)


def poll_node_gpus(
    node: str,
    *,
    ssh_options: Sequence[str] = ("BatchMode=yes", "ConnectTimeout=5"),
    runner: CommandRunner = run_command,
    timeout: float = 8.0,
) -> Tuple[Tuple[GPUDevice, ...], Optional[str]]:
    gpus, _host, error = poll_node(node, ssh_options=ssh_options, runner=runner, timeout=timeout)
    return gpus, error


def poll_node(
    node: str,
    *,
    ssh_options: Sequence[str] = ("BatchMode=yes", "ConnectTimeout=5"),
    runner: CommandRunner = run_command,
    timeout: float = 8.0,
) -> Tuple[Tuple[GPUDevice, ...], HostStats, Optional[str]]:
    ssh_args = ["ssh"]
    for option in ssh_options:
        ssh_args.extend(["-o", option])
    ssh_args.extend([node, REMOTE_NVIDIA_SMI_QUERY])

    result = runner(ssh_args, timeout)
    if not result.ok:
        error = result.stderr.strip() or result.stdout.strip() or "GPU query failed"
        if result.timed_out:
            error = f"timed out after {timeout:g}s: {error}"
        return (), HostStats(hostname=node), error

    try:
        return (*parse_node_probe_output(node, result.stdout), None)
    except ValueError as exc:
        return (), HostStats(hostname=node), str(exc)


def parse_nvidia_smi_output(node: str, output: str) -> Tuple[GPUDevice, ...]:
    return parse_node_probe_output(node, output)[0]


def parse_node_probe_output(node: str, output: str) -> Tuple[Tuple[GPUDevice, ...], HostStats]:
    sections = _split_sections(output)
    host = parse_host_stats(node, sections.get(META_MARKER, ""))
    gpu_text = sections.get(GPU_MARKER)
    process_text = sections.get(PROCESS_MARKER, "")
    pmon_text = sections.get(PMON_MARKER, "")
    ps_text = sections.get(PS_MARKER, "")

    if gpu_text is None:
        # Backward-compatible parser for old fixture strings.
        if PROCESS_MARKER in output:
            gpu_text, process_text = output.split(PROCESS_MARKER, 1)
        else:
            gpu_text = output

    pmon_by_pid_gpu = parse_pmon_output(pmon_text)
    ps_by_pid = parse_ps_output(ps_text)

    processes_by_uuid: Dict[str, List[GPUProcess]] = {}
    for row in _csv_rows(process_text):
        if len(row) < 4:
            continue
        pid = _parse_int(row[0])
        if pid is None:
            continue
        uuid = row[3].strip()
        ps_info = ps_by_pid.get(pid, {})
        pmon_info = pmon_by_pid_gpu.get((pid, uuid)) or pmon_by_pid_gpu.get((pid, "")) or {}
        proc = GPUProcess(
            pid=pid,
            name=row[1].strip(),
            used_memory_mib=_parse_int(row[2]),
            gpu_uuid=uuid,
            type=pmon_info.get("type") or "C",
            user=ps_info.get("user", ""),
            cpu_percent=_parse_float(ps_info.get("cpu")),
            mem_percent=_parse_float(ps_info.get("mem")),
            elapsed=ps_info.get("elapsed", ""),
            command=ps_info.get("command", ""),
            sm_util_percent=_parse_int(pmon_info.get("sm")),
            mem_bw_util_percent=_parse_int(pmon_info.get("mem")),
        )
        processes_by_uuid.setdefault(proc.gpu_uuid, []).append(proc)

    gpus: List[GPUDevice] = []
    for row in _csv_rows(gpu_text):
        if len(row) >= 19:
            gpus.append(_parse_rich_gpu_row(node, row, processes_by_uuid))
            continue
        if len(row) >= 10:
            gpus.append(_parse_legacy_gpu_row(node, row, processes_by_uuid))
            continue
        raise ValueError(f"unexpected nvidia-smi GPU row with {len(row)} fields: {row!r}")
    return tuple(gpus), host


def parse_host_stats(node: str, text: str) -> HostStats:
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()

    memory = [_parse_float(part) for part in fields.get("memory", "").split()]
    swap = [_parse_float(part) for part in fields.get("swap", "").split()]
    load = tuple(
        value
        for value in (_parse_float(part) for part in fields.get("load_average", "").split())
        if value is not None
    )
    return HostStats(
        cpu_percent=_parse_float(fields.get("cpu_percent")),
        memory_used_mib=_float_to_int(memory[0] if len(memory) > 0 else None),
        memory_total_mib=_float_to_int(memory[1] if len(memory) > 1 else None),
        memory_percent=memory[2] if len(memory) > 2 else None,
        swap_used_mib=_float_to_int(swap[0] if len(swap) > 0 else None),
        swap_total_mib=_float_to_int(swap[1] if len(swap) > 1 else None),
        swap_percent=swap[2] if len(swap) > 2 else None,
        load_average=load[:3],
        uptime_seconds=_parse_float(fields.get("uptime_seconds")),
        hostname=fields.get("hostname", node) or node,
        driver_version=fields.get("driver_version", ""),
        cuda_version=fields.get("cuda_version", ""),
    )


def parse_pmon_output(text: str) -> Dict[Tuple[int, str], Dict[str, str]]:
    rows: Dict[Tuple[int, str], Dict[str, str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 7)
        if len(parts) < 7:
            continue
        gpu, pid_s, proc_type, sm, mem, _enc, _dec = parts[:7]
        pid = _parse_int(pid_s)
        if pid is None or pid < 0:
            continue
        rows[(pid, "")] = {"gpu": gpu, "type": proc_type, "sm": sm, "mem": mem}
    return rows


def parse_ps_output(text: str) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid = _parse_int(parts[0])
        if pid is None:
            continue
        rows[pid] = {
            "user": parts[1],
            "cpu": parts[2],
            "mem": parts[3],
            "elapsed": parts[4],
            "command": parts[5],
        }
    return rows


def _parse_rich_gpu_row(
    node: str,
    row: List[str],
    processes_by_uuid: Dict[str, List[GPUProcess]],
) -> GPUDevice:
    index = _parse_int(row[0])
    if index is None:
        raise ValueError(f"GPU index is not an integer: {row[0]!r}")
    uuid = row[1].strip()
    return GPUDevice(
        node=node,
        index=index,
        uuid=uuid,
        name=row[2].strip(),
        persistence_mode=row[3].strip(),
        pci_bus_id=row[4].strip(),
        display_active=row[5].strip(),
        mig_mode=row[6].strip(),
        ecc_errors=_parse_int(row[7]),
        fan_speed_percent=_parse_int(row[8]),
        temperature_c=_parse_int(row[9]),
        performance_state=row[10].strip(),
        power_draw_w=_parse_float(row[11]),
        power_limit_w=_parse_float(row[12]),
        mem_used_mib=_parse_int(row[13]),
        mem_total_mib=_parse_int(row[14]),
        gpu_util_percent=_parse_int(row[15]),
        mem_util_percent=_parse_int(row[16]),
        compute_mode=row[17].strip(),
        sm_clock_mhz=_parse_int(row[18]),
        processes=tuple(processes_by_uuid.get(uuid, ())),
    )


def _parse_legacy_gpu_row(
    node: str,
    row: List[str],
    processes_by_uuid: Dict[str, List[GPUProcess]],
) -> GPUDevice:
    index = _parse_int(row[0])
    if index is None:
        raise ValueError(f"GPU index is not an integer: {row[0]!r}")
    uuid = row[1].strip()
    return GPUDevice(
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


def _split_sections(output: str) -> Dict[str, str]:
    markers = {META_MARKER, GPU_MARKER, PROCESS_MARKER, PMON_MARKER, PS_MARKER}
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped in markers:
            current = stripped
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(value) for key, value in sections.items()}


def _old_parse_nvidia_smi_output(node: str, output: str) -> Tuple[GPUDevice, ...]:
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


def _float_to_int(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(value)
