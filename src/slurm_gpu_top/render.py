from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Iterable, Optional, Sequence

from .models import ClusterSnapshot, GPUDevice, GPUProcess, HostStats, NodeSnapshot, SlurmJob

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\033[0m"
COLORS = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_cyan": "\033[96m",
}
PARTIALS = "▏▎▍▌▋▊▉"


def render_snapshot(
    snapshot: ClusterSnapshot,
    *,
    width: Optional[int] = None,
    color: Optional[bool] = None,
    unicode: bool = True,
    all_gpu_history: Sequence[Optional[int]] = (),
    gpu_histories: Optional[dict[tuple[str, str], Sequence[Optional[int]]]] = None,
    version: str = "0.1.0",
) -> str:
    del all_gpu_history, gpu_histories
    width = max(79, width or _terminal_width())
    color = _should_color() if color is None else color
    gpus = _all_gpus(snapshot)
    avg_util = _avg(gpu.gpu_util_percent for gpu in gpus)
    avg_mem = _avg(gpu.mem_util_percent for gpu in gpus)

    lines = [time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(snapshot.generated_at))]

    if snapshot.errors:
        lines.insert(0, _style(f"SGTOP {version}", "bold", color))
        lines.extend(_style(f"Slurm error: {error}", "bright_red", color) for error in snapshot.errors)
        return "\n".join(lines)

    if not snapshot.nodes:
        lines.insert(0, _style(f"SGTOP {version}", "bold", color))
        lines.append(_style("No running GPU-backed Slurm jobs found.", "yellow", color))
        return "\n".join(lines)

    lines.extend(
        _cluster_gpu_box(
            snapshot,
            width=width,
            color=color,
            unicode=unicode,
            version=version,
            avg_util=avg_util,
            avg_mem=avg_mem,
        )
    )
    lines.append("")
    lines.extend(_cluster_process_box(snapshot, width=width, color=color, unicode=unicode))
    return "\n".join(lines)


def _cluster_gpu_box(
    snapshot: ClusterSnapshot,
    *,
    width: int,
    color: bool,
    unicode: bool,
    version: str,
    avg_util: Optional[float],
    avg_mem: Optional[float],
) -> Iterable[str]:
    chars = _box_chars(unicode)
    w = min(width, 140)
    inner = w - 2
    c1, c2, c3 = _gpu_col_widths(inner)
    c4 = inner - c1 - c2 - c3 - 3
    sample_host = next(
        (node.host for node in snapshot.nodes if node.host.driver_version or node.host.cuda_version),
        HostStats(),
    )
    title = _fit_right(
        _style(f"SGTOP {version}", "bold", color),
        (
            f"Driver Version: {sample_host.driver_version or 'N/A'}      "
            f"CUDA Driver Version: {sample_host.cuda_version or 'N/A'}      "
            f"ALL GPUs util={_percent(avg_util)} mem={_percent(avg_mem)}"
        ),
        inner,
    )

    yield chars["tl"] + chars["h2"] * inner + chars["tr"]
    yield _box_line(title, inner, chars)
    for node_idx, node in enumerate(snapshot.nodes):
        yield chars["ml_bold"] + chars["h2"] * inner + chars["mr_bold"]
        yield _box_line(_node_status(node, color=color), inner, chars)
        if node.error:
            yield _box_line(_style(f"! {node.error}", "bright_red", color), inner, chars)
            continue
        yield _joint_line(chars, (c1, c2, c3, c4), "top")
        yield _row(
            ("GPU  Name        Persistence-M", "Bus-Id        Disp.A", "MIG M.   Uncorr. ECC", ""),
            (c1, c2, c3, c4),
            chars,
        )
        yield _row(
            ("Fan  Temp  Perf  Pwr:Usage/Cap", "        Memory-Usage", "GPU-Util  Compute M.", ""),
            (c1, c2, c3, c4),
            chars,
        )
        yield _joint_line(chars, (c1, c2, c3, c4), "mid_bold")
        if not node.gpus:
            yield _box_line(_style("No GPUs reported by nvidia-smi.", "yellow", color), inner, chars)
            continue
        for gpu_idx, gpu in enumerate(sorted(node.gpus, key=lambda item: item.index)):
            if gpu_idx:
                yield _joint_line(chars, (c1, c2, c3, c4), "mid")
            yield from _gpu_rows(gpu, (c1, c2, c3, c4), chars, color=color, unicode=unicode)
    yield chars["bl"] + chars["h2"] * inner + chars["br"]


def _gpu_col_widths(inner: int) -> tuple[int, int, int]:
    if inner >= 118:
        return 31, 22, 22
    if inner >= 98:
        return 28, 19, 19
    return 24, 16, 16


def _gpu_rows(
    gpu: GPUDevice,
    widths: Sequence[int],
    chars: dict[str, str],
    *,
    color: bool,
    unicode: bool,
) -> Iterable[str]:
    mem_percent = _memory_percent(gpu)
    util = gpu.gpu_util_percent
    yield _row(
        (
            f"{gpu.index:>3}  {_gpu_name(gpu.name):<17.17} {_short(_on_off(gpu.persistence_mode), 3):>8}",
            f"{_short(gpu.pci_bus_id, 16):<16} {_short(_on_off(gpu.display_active), 3):>3}",
            f"{_short(gpu.mig_mode, 8):<8} {_na_int(gpu.ecc_errors):>11}",
            _bar_stat("MEM", mem_percent, _mem_usage_short(gpu), color=color, unicode=unicode),
        ),
        widths,
        chars,
    )
    yield _row(
        (
            f"{_fan(gpu):>3}  {_temp(gpu):>4} {_short(gpu.performance_state, 4):>4} {_power(gpu):>13}",
            f"{_mem_usage(gpu):>21}",
            f"{_percent(util):>7} {_short(gpu.compute_mode, 12):>12}",
            _bar_stat("UTL", util, f"{_percent(util):>4} @ {_clock(gpu)}", color=color, unicode=unicode),
        ),
        widths,
        chars,
    )


def _node_status(node: NodeSnapshot, *, color: bool) -> str:
    host = node.host
    host_bits = []
    if host.cpu_percent is not None:
        host_bits.append(f"CPU {_percent(host.cpu_percent)}")
    if host.memory_percent is not None:
        host_bits.append(f"MEM {_percent(host.memory_percent)}")
    if host.load_average:
        host_bits.append("LOAD " + " ".join(f"{value:.2f}" for value in host.load_average))
    if host.uptime_seconds is not None:
        host_bits.append(f"UP {_uptime(host.uptime_seconds)}")
    suffix = f" -- {'  '.join(host_bits)}" if host_bits else ""
    return (
        _style(f"[{node.node}]", "bright_cyan", color)
        + " "
        + _style(_format_jobs(node.jobs), "yellow", color)
        + suffix
    )


def _cluster_process_box(
    snapshot: ClusterSnapshot,
    *,
    width: int,
    color: bool,
    unicode: bool,
) -> Iterable[str]:
    chars = _box_chars(unicode)
    w = min(width, 140)
    inner = w - 2
    rows = [
        (node, gpu, proc)
        for node in snapshot.nodes
        for gpu in sorted(node.gpus, key=lambda item: item.index)
        for proc in gpu.processes
    ]
    yield chars["tl"] + chars["h2"] * inner + chars["tr"]
    yield _box_line(
        _fit_right("Processes:", f"{snapshot.job_count} jobs across {len(snapshot.nodes)} nodes", inner),
        inner,
        chars,
    )
    yield _box_line("NODE          GPU     PID      USER  GPU-MEM %SM %GMBW  %CPU  %MEM     TIME  COMMAND", inner, chars)
    yield chars["ml_bold"] + chars["h2"] * inner + chars["mr_bold"]
    if not rows:
        yield _box_line(_style("No running GPU compute processes found.", "bright_black", color), inner, chars)
    for idx, (node, gpu, proc) in enumerate(rows):
        if idx:
            yield chars["ml"] + chars["h1"] * inner + chars["mr"]
        yield _box_line(_format_cluster_process_row(node, gpu, proc, inner), inner, chars)
    yield chars["bl"] + chars["h2"] * inner + chars["br"]


def _format_cluster_process_row(node: NodeSnapshot, gpu: GPUDevice, proc: GPUProcess, width: int) -> str:
    command = proc.command or proc.name
    fixed = (
        f"{node.node:<12} {gpu.index:>3}  {proc.pid:>6} {_short(proc.type, 3):<3} "
        f"{_short(proc.user, 5):<5} {_value(proc.used_memory_mib, 'MiB'):>8} "
        f"{_na_int(proc.sm_util_percent):>3} {_na_int(proc.mem_bw_util_percent):>5} "
        f"{_na_float(proc.cpu_percent):>5} {_na_float(proc.mem_percent):>5} "
        f"{(proc.elapsed or 'N/A'):>8}  "
    )
    return fixed + _clip_visible(command, max(0, width - _visible_len(fixed)))


def _render_node(node: NodeSnapshot, *, width: int, color: bool, unicode: bool) -> Iterable[str]:
    host = node.host
    title_host = host.hostname or node.node
    yield _clip(
        _style(f"[{node.node}]", "bright_cyan", color)
        + " "
        + _style(_format_jobs(node.jobs), "yellow", color),
        width,
    )

    if node.error:
        yield from _simple_box([_style(f"! {node.error}", "bright_red", color)], width, unicode=unicode)
        return

    yield from _gpu_box(node.gpus, host, width=width, color=color, unicode=unicode)
    yield from _host_lines(host, width=width, color=color, unicode=unicode)
    yield ""
    yield from _process_box(node, title_host, width=width, color=color, unicode=unicode)


def _gpu_box(
    gpus: Sequence[GPUDevice],
    host: HostStats,
    *,
    width: int,
    color: bool,
    unicode: bool,
) -> Iterable[str]:
    chars = _box_chars(unicode)
    w = min(width, 120)
    inner = w - 2
    if inner >= 98:
        c1, c2, c3 = 31, 22, 22
    else:
        c1, c2, c3 = 24, 16, 16
    c4 = max(12, inner - c1 - c2 - c3 - 3)
    sep = chars["v"]

    title = (
        _style("NVITOP", "bold", color)
        + " "
        + _style("1.6.2-like", "green", color)
        + f"      Driver Version: {host.driver_version or 'N/A'}"
        + f"      CUDA Driver Version: {host.cuda_version or 'N/A'}"
    )

    yield chars["tl"] + chars["h2"] * inner + chars["tr"]
    yield _box_line(_clip_visible(title, inner), inner, chars)
    yield _joint_line(chars, (c1, c2, c3, c4), "top")
    yield _row(("GPU  Name        Persistence-M", "Bus-Id        Disp.A", "MIG M.   Uncorr. ECC", ""), (c1, c2, c3, c4), chars)
    yield _row(("Fan  Temp  Perf  Pwr:Usage/Cap", "        Memory-Usage", "GPU-Util  Compute M.", ""), (c1, c2, c3, c4), chars)
    yield _joint_line(chars, (c1, c2, c3, c4), "mid_bold")

    if not gpus:
        yield _box_line(_style("No GPUs reported by nvidia-smi.", "yellow", color), inner, chars)
    for idx, gpu in enumerate(sorted(gpus, key=lambda item: item.index)):
        if idx:
            yield _joint_line(chars, (c1, c2, c3, c4), "mid")
        mem_percent = _memory_percent(gpu)
        util = gpu.gpu_util_percent
        yield _row(
            (
                f"{gpu.index:>3}  {_gpu_name(gpu.name):<17.17} {_short(_on_off(gpu.persistence_mode), 3):>8}",
                f"{_short(gpu.pci_bus_id, 16):<16} {_short(_on_off(gpu.display_active), 3):>3}",
                f"{_short(gpu.mig_mode, 8):<8} {_na_int(gpu.ecc_errors):>11}",
                _bar_stat("MEM", mem_percent, _mem_usage_short(gpu), color=color, unicode=unicode),
            ),
            (c1, c2, c3, c4),
            chars,
        )
        yield _row(
            (
                f"{_fan(gpu):>3}  {_temp(gpu):>4} {_short(gpu.performance_state, 4):>4} {_power(gpu):>13}",
                f"{_mem_usage(gpu):>21}",
                f"{_percent(util):>7} {_short(gpu.compute_mode, 12):>12}",
                _bar_stat("UTL", util, f"{_percent(util):>4} @ {_clock(gpu)}", color=color, unicode=unicode),
            ),
            (c1, c2, c3, c4),
            chars,
        )
    yield _joint_line(chars, (c1, c2, c3, c4), "bottom")


def _host_lines(host: HostStats, *, width: int, color: bool, unicode: bool) -> Iterable[str]:
    cpu = _host_bar("CPU", host.cpu_percent, 24, color=color, unicode=unicode)
    uptime = _uptime(host.uptime_seconds)
    loads = " ".join(f"{value:.2f}" for value in host.load_average) or "N/A"
    yield _clip(f"[ {cpu:<44} UPTIME: {uptime:>10} ]  ( Load Average: {loads} )", width)

    mem = _host_bar("MEM", host.memory_percent, 10, color=color, unicode=unicode)
    mem_used = _human_mib(host.memory_used_mib)
    swp = _host_bar("SWP", host.swap_percent, 10, color=color, unicode=unicode)
    yield _clip(f"[ {mem:<50} USED: {mem_used:>9} ]  [ {swp:<28} ]", width)


def _process_box(
    node: NodeSnapshot,
    title_host: str,
    *,
    width: int,
    color: bool,
    unicode: bool,
) -> Iterable[str]:
    chars = _box_chars(unicode)
    w = min(width, 120)
    inner = w - 2
    rows = [(gpu, proc) for gpu in sorted(node.gpus, key=lambda item: item.index) for proc in gpu.processes]

    yield chars["tl"] + chars["h2"] * inner + chars["tr"]
    yield _box_line(
        _fit_right("Processes:", f"{_user_hint(rows)}@{title_host}", inner),
        inner,
        chars,
    )
    yield _box_line("GPU     PID      USER  GPU-MEM %SM %GMBW  %CPU  %MEM     TIME  COMMAND", inner, chars)
    yield chars["ml_bold"] + chars["h2"] * inner + chars["mr_bold"]
    if not rows:
        yield _box_line(_style("No running GPU compute processes found.", "bright_black", color), inner, chars)
    for idx, (gpu, proc) in enumerate(rows):
        if idx:
            yield chars["ml"] + chars["h1"] * inner + chars["mr"]
        yield _box_line(_format_process_row(gpu, proc, inner), inner, chars)
    yield chars["bl"] + chars["h2"] * inner + chars["br"]


def _format_process_row(gpu: GPUDevice, proc: GPUProcess, width: int) -> str:
    command = proc.command or proc.name
    fixed = (
        f"{gpu.index:>3}  {proc.pid:>6} {_short(proc.type, 3):<3} "
        f"{_short(proc.user, 5):<5} {_value(proc.used_memory_mib, 'MiB'):>8} "
        f"{_na_int(proc.sm_util_percent):>3} {_na_int(proc.mem_bw_util_percent):>5} "
        f"{_na_float(proc.cpu_percent):>5} {_na_float(proc.mem_percent):>5} "
        f"{(proc.elapsed or 'N/A'):>8}  "
    )
    return fixed + _clip_visible(command, max(0, width - _visible_len(fixed)))


def _format_jobs(jobs: Iterable[SlurmJob]) -> str:
    parts = [f"{job.job_id} {job.user}/{job.name} {job.elapsed}" for job in jobs]
    return ", ".join(parts) if parts else "unknown"


def _bar_stat(label: str, percent: Optional[float], suffix: str, *, color: bool, unicode: bool) -> str:
    bar = _fraction_bar(percent, width=8, color=color, unicode=unicode)
    return f"{label}: {bar} {suffix}"


def _host_bar(label: str, percent: Optional[float], width: int, *, color: bool, unicode: bool) -> str:
    return f"{label}: {_fraction_bar(percent, width=width, color=color, unicode=unicode)} {_percent(percent):>5}"


def _fraction_bar(percent: Optional[float], *, width: int, color: bool, unicode: bool) -> str:
    if percent is None:
        raw = "?" * width
        return _style(raw, "bright_black", color)
    percent = max(0.0, min(100.0, float(percent)))
    if not unicode:
        full = round(width * percent / 100)
        return _style("#" * full, _util_color(percent), color) + _style("." * (width - full), "bright_black", color)
    units = percent / 100 * width
    full = int(units)
    remainder = units - full
    partial = ""
    if full < width and remainder > 0:
        partial = PARTIALS[min(len(PARTIALS) - 1, max(0, int(remainder * len(PARTIALS))))]
    empty = max(0, width - full - (1 if partial else 0))
    return (
        _style("█" * full + partial, _util_color(percent), color)
        + _style(" " * empty, "bright_black", color)
    )


def _row(values: Sequence[str], widths: Sequence[int], chars: dict[str, str]) -> str:
    return chars["v"] + chars["v"].join(_pad(value, width) for value, width in zip(values, widths)) + chars["v"]


def _box_line(value: str, width: int, chars: dict[str, str]) -> str:
    return chars["v"] + _pad(value, width) + chars["v"]


def _joint_line(chars: dict[str, str], widths: Sequence[int], kind: str) -> str:
    if kind == "top":
        left, joint, right, fill = chars["lt"], chars["tj"], chars["rt"], chars["h1"]
    elif kind == "mid":
        left, joint, right, fill = chars["ml"], chars["mj"], chars["mr"], chars["h1"]
    elif kind == "mid_bold":
        left, joint, right, fill = chars["ml_bold"], chars["mj_bold"], chars["mr_bold"], chars["h2"]
    elif kind == "bottom":
        left, joint, right, fill = chars["bl"], chars["bj"], chars["br"], chars["h2"]
    else:
        left, joint, right, fill = chars["ml"], chars["mj"], chars["mr"], chars["h1"]
    return left + joint.join(fill * width for width in widths) + right


def _simple_box(lines: Sequence[str], width: int, *, unicode: bool) -> Iterable[str]:
    chars = _box_chars(unicode)
    inner = min(width, 120) - 2
    yield chars["tl"] + chars["h2"] * inner + chars["tr"]
    for line in lines:
        yield _box_line(_clip_visible(line, inner), inner, chars)
    yield chars["bl"] + chars["h2"] * inner + chars["br"]


def _box_chars(unicode: bool) -> dict[str, str]:
    if not unicode:
        return {
            "tl": "+",
            "tr": "+",
            "bl": "+",
            "br": "+",
            "lt": "+",
            "rt": "+",
            "ml": "+",
            "mr": "+",
            "ml_bold": "+",
            "mr_bold": "+",
            "tj": "+",
            "mj": "+",
            "mj_bold": "+",
            "bj": "+",
            "v": "|",
            "h1": "-",
            "h2": "=",
        }
    return {
        "tl": "╒",
        "tr": "╕",
        "bl": "╘",
        "br": "╛",
        "lt": "├",
        "rt": "┤",
        "ml": "├",
        "mr": "┤",
        "ml_bold": "╞",
        "mr_bold": "╡",
        "tj": "┬",
        "mj": "┼",
        "mj_bold": "╪",
        "bj": "╧",
        "v": "│",
        "h1": "─",
        "h2": "═",
    }


def _gpu_name(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("SXM4-", "").strip()


def _on_off(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"enabled", "enable", "active", "yes", "on"}:
        return "On"
    if normalized in {"disabled", "disable", "inactive", "no", "off"}:
        return "Off"
    return value


def _memory_percent(gpu: GPUDevice) -> Optional[float]:
    if gpu.mem_used_mib is None or not gpu.mem_total_mib:
        return gpu.mem_util_percent
    return 100 * gpu.mem_used_mib / gpu.mem_total_mib


def _mem_usage(gpu: GPUDevice) -> str:
    if gpu.mem_used_mib is None or gpu.mem_total_mib is None:
        return "N/A"
    return f"{gpu.mem_used_mib}MiB / {_human_mib(gpu.mem_total_mib)}"


def _mem_usage_short(gpu: GPUDevice) -> str:
    percent = _memory_percent(gpu)
    if gpu.mem_used_mib is None:
        return f"N/A {_percent(percent):>7}"
    return f"{gpu.mem_used_mib}MiB {_percent(percent):>7}"


def _human_mib(mib: Optional[int]) -> str:
    if mib is None:
        return "N/A"
    if mib >= 1024:
        return f"{mib / 1024:.2f}GiB"
    return f"{mib}MiB"


def _fan(gpu: GPUDevice) -> str:
    return "N/A" if gpu.fan_speed_percent is None else f"{gpu.fan_speed_percent}%"


def _temp(gpu: GPUDevice) -> str:
    return "N/A" if gpu.temperature_c is None else f"{gpu.temperature_c}C"


def _power(gpu: GPUDevice) -> str:
    if gpu.power_draw_w is None and gpu.power_limit_w is None:
        return "N/A"
    if gpu.power_limit_w is None:
        return f"{gpu.power_draw_w:.0f}W"
    if gpu.power_draw_w is None:
        return f"N/A / {gpu.power_limit_w:.0f}W"
    return f"{gpu.power_draw_w:.0f}W / {gpu.power_limit_w:.0f}W"


def _clock(gpu: GPUDevice) -> str:
    return "N/A" if gpu.sm_clock_mhz is None else f"{gpu.sm_clock_mhz}MHz"


def _uptime(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    days = seconds / 86400
    if days >= 1:
        return f"{days:.1f} days"
    hours = seconds / 3600
    return f"{hours:.1f} hours"


def _percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.0f}%"


def _value(value: object, unit: str) -> str:
    return "N/A" if value is None else f"{value}{unit}"


def _na_int(value: Optional[int]) -> str:
    return "N/A" if value is None else str(value)


def _na_float(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def _short(value: object, width: int) -> str:
    text = "N/A" if value is None or str(value) == "" else str(value)
    return _clip_visible(text, width)


def _fit_right(left: str, right: str, width: int) -> str:
    gap = max(1, width - _visible_len(left) - _visible_len(right))
    return left + " " * gap + right


def _user_hint(rows: Sequence[tuple[GPUDevice, GPUProcess]]) -> str:
    for _gpu, proc in rows:
        if proc.user:
            return proc.user
    return os.environ.get("USER", "user")


def _all_gpus(snapshot: ClusterSnapshot) -> list[GPUDevice]:
    return [gpu for node in snapshot.nodes for gpu in node.gpus]


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _util_color(value: Optional[float]) -> str:
    if value is None:
        return "yellow"
    if value >= 75:
        return "bright_red"
    if value >= 10:
        return "bright_yellow"
    return "bright_green"


def _style(text: object, color_name: str, enabled: bool) -> str:
    text = str(text)
    if not enabled:
        return text
    code = COLORS.get(color_name, "")
    return f"{code}{text}{RESET}" if code else text


def _pad(text: str, width: int) -> str:
    text = _clip_visible(text, width)
    return text + " " * max(0, width - _visible_len(text))


def _clip(text: str, width: int) -> str:
    if width <= 1 or _visible_len(text) <= width:
        return text
    return _clip_visible(text, width - 1) + "…"


def _clip_visible(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _visible_len(text) <= width:
        return text
    out = []
    visible = 0
    idx = 0
    had_ansi = False
    while idx < len(text) and visible < width:
        match = ANSI_RE.match(text, idx)
        if match:
            had_ansi = True
            out.append(match.group(0))
            idx = match.end()
            continue
        out.append(text[idx])
        visible += 1
        idx += 1
    if had_ansi:
        out.append(RESET)
    return "".join(out)


def _visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def _should_color() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("ANSI_COLORS_DISABLED"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    term = os.environ.get("TERM", "")
    return bool(term and term != "dumb" and sys.stdout.isatty())


def _terminal_width() -> int:
    return shutil.get_terminal_size(fallback=(120, 24)).columns
