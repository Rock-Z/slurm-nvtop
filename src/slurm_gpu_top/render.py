from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Iterable, Optional, Sequence

from .models import ClusterSnapshot, GPUDevice, GPUProcess, NodeSnapshot, SlurmJob

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\033[0m"
COLORS = {
    "dim": "\033[2m",
    "bold": "\033[1m",
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
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
}
SPARKLINE = "▁▂▃▄▅▆▇█"
ASCII_SPARKLINE = "._:-=+*#"


def render_snapshot(
    snapshot: ClusterSnapshot,
    *,
    width: Optional[int] = None,
    color: Optional[bool] = None,
    unicode: bool = True,
    all_gpu_history: Sequence[Optional[int]] = (),
    gpu_histories: Optional[dict[tuple[str, str], Sequence[Optional[int]]]] = None,
) -> str:
    width = width or _terminal_width()
    color = _should_color() if color is None else color
    gpu_histories = gpu_histories or {}
    gpus = _all_gpus(snapshot)
    avg_util = _avg([gpu.gpu_util_percent for gpu in gpus])
    avg_mem = _avg([gpu.mem_util_percent for gpu in gpus])
    process_count = sum(len(gpu.processes) for gpu in gpus)
    lines = [
        _clip(
            _style("slurm-gpu-top", "bold", color)
            + "  "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snapshot.generated_at))}  "
            + _style(
                f"jobs={snapshot.job_count} nodes={len(snapshot.nodes)} "
                f"gpus={snapshot.gpu_count} procs={process_count}",
                "cyan",
                color,
            ),
            width,
        ),
        _rule(width, "═" if unicode else "=", color),
    ]

    if snapshot.errors:
        lines.extend(
            _clip(_style(f"Slurm error: {error}", "bright_red", color), width)
            for error in snapshot.errors
        )
        return "\n".join(lines)

    if not snapshot.nodes:
        lines.append(_style("No running GPU-backed Slurm jobs found.", "yellow", color))
        return "\n".join(lines)

    graph_width = max(12, min(48, width - 42))
    all_history = tuple(all_gpu_history) or (avg_util,)
    lines.extend(
        _render_cluster_summary(
            avg_util=avg_util,
            avg_mem=avg_mem,
            graph=_sparkline(all_history, graph_width, unicode=unicode),
            color=color,
            width=width,
        )
    )

    for node in snapshot.nodes:
        lines.extend(_render_node(node, width, color=color, unicode=unicode, gpu_histories=gpu_histories))
    lines.extend(_render_process_table(snapshot, width, color=color, unicode=unicode))
    return "\n".join(lines)


def _render_cluster_summary(
    *,
    avg_util: Optional[int],
    avg_mem: Optional[int],
    graph: str,
    color: bool,
    width: int,
) -> Iterable[str]:
    util_color = _util_color(avg_util)
    prefix = (
        "ALL GPUs  "
        + _style(f"avg util {_percent(avg_util).strip():>4}", util_color, color)
        + "  "
        + _style(f"avg mem {_percent(avg_mem).strip():>4}", _util_color(avg_mem, memory=True), color)
    )
    graph_text = "  util history " + _style(graph, util_color, color)
    if _visible_len(prefix + graph_text) <= width:
        yield _clip(prefix + graph_text, width)
        return
    yield _clip(prefix, width)
    yield _clip("  util history " + _style(graph.strip() or graph, util_color, color), width)


def _render_node(
    node: NodeSnapshot,
    width: int,
    *,
    color: bool,
    unicode: bool,
    gpu_histories: dict[tuple[str, str], Sequence[Optional[int]]],
) -> Iterable[str]:
    yield ""
    node_avg = _avg([gpu.gpu_util_percent for gpu in node.gpus])
    node_title = (
        _style(node.node, "bright_cyan", color)
        + "  "
        + _style(f"avg {_percent(node_avg).strip():>4}", _util_color(node_avg), color)
        + "  jobs: "
        + _format_jobs(node.jobs, color=color)
    )
    yield _clip(node_title, width)
    yield _rule(width, "─" if unicode else "-", color)
    if node.error:
        yield _clip("  " + _style(f"! {node.error}", "bright_red", color), width)
        return
    if not node.gpus:
        yield _style("  No GPUs reported by nvidia-smi.", "yellow", color)
        return

    compact = width < 110
    graph_width = max(8, min(16, width - 104)) if not compact else max(12, min(36, width - 20))
    header = (
        "  GPU  UTIL      MEM           TEMP       POWER"
        if compact
        else "  GPU  UTIL      MEM           TEMP       POWER       NAME                         HISTORY"
    )
    yield _style(_clip(header, width), "bright_black", color)
    for gpu in sorted(node.gpus, key=lambda item: item.index):
        history = gpu_histories.get(_gpu_key(gpu), (gpu.gpu_util_percent,))
        yield _clip(
            _format_gpu(
                gpu,
                history,
                graph_width=graph_width,
                color=color,
                unicode=unicode,
                show_history=not compact,
            ),
            width,
        )
        if compact:
            graph = _style(
                _sparkline(history, graph_width, unicode=unicode).strip() or "?",
                _util_color(gpu.gpu_util_percent),
                color,
            )
            yield _clip(f"       {gpu.name[:24]:<24} history {graph}", width)
        if gpu.processes:
            for proc in sorted(gpu.processes, key=lambda item: item.pid):
                yield _clip(_format_process(proc, color=color), width)
        else:
            yield _clip(_style("       no running GPU compute processes", "bright_black", color), width)


def _render_process_table(
    snapshot: ClusterSnapshot,
    width: int,
    *,
    color: bool,
    unicode: bool,
) -> Iterable[str]:
    rows = []
    for node in snapshot.nodes:
        for gpu in node.gpus:
            for proc in gpu.processes:
                rows.append((node, gpu, proc))

    yield ""
    yield _clip(_style("Processes", "bold", color), width)
    yield _rule(width, "─" if unicode else "-", color)
    if not rows:
        yield _style("  No running GPU compute processes found.", "bright_black", color)
        return

    yield _style("  NODE          GPU  PID       GPU MEM     PROCESS", "bright_black", color)
    for node, gpu, proc in sorted(rows, key=lambda item: (item[0].node, item[1].index, item[2].pid)):
        mem = _value(proc.used_memory_mib, "MiB")
        yield _clip(
            f"  {node.node:<12}  {gpu.index:<3}  {proc.pid:<8}  {mem:>9}   {proc.name}",
            width,
        )


def _format_jobs(jobs: Iterable[SlurmJob], *, color: bool) -> str:
    parts = []
    for job in jobs:
        label = (
            _style(job.job_id, "yellow", color)
            + f" {job.user}/{job.name} {job.elapsed}"
        )
        parts.append(label)
    return ", ".join(parts) if parts else "unknown"


def _format_gpu(
    gpu: GPUDevice,
    history: Sequence[Optional[int]],
    *,
    graph_width: int,
    color: bool,
    unicode: bool,
    show_history: bool,
) -> str:
    util_color = _util_color(gpu.gpu_util_percent)
    mem_color = _util_color(gpu.mem_util_percent, memory=True)
    util = _style(_percent(gpu.gpu_util_percent), util_color, color)
    mem_util = _style(_percent(gpu.mem_util_percent), mem_color, color)
    mem = _memory(gpu.mem_used_mib, gpu.mem_total_mib)
    temp = _value(gpu.temperature_c, "C")
    power = _power(gpu.power_draw_w, gpu.power_limit_w)
    bar = _bar(gpu.gpu_util_percent, color=color, unicode=unicode)
    line = (
        f"  {gpu.index:<3}  {util:>5} {bar}  {mem_util:>5} {mem:>15}  "
        f"{temp:>7}  {power:>11}"
    )
    if show_history:
        graph = _style(_sparkline(history, graph_width, unicode=unicode), util_color, color)
        line += f"  {gpu.name[:22]:<22} {graph}"
    return line


def _format_process(proc: GPUProcess, *, color: bool) -> str:
    mem = _value(proc.used_memory_mib, "MiB")
    return (
        "       "
        + _style(f"pid={proc.pid:<7}", "bright_black", color)
        + f" mem={mem:>9}  {proc.name}"
    )


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


def _bar(
    value: Optional[int],
    *,
    width: int = 12,
    color: bool = False,
    unicode: bool = True,
) -> str:
    if value is None:
        return "[" + _style("?" * width, "bright_black", color) + "]"
    filled = max(0, min(width, round(width * value / 100)))
    fill_char = "█" if unicode else "#"
    empty_char = "░" if unicode else "."
    return (
        "["
        + _style(fill_char * filled, _util_color(value), color)
        + _style(empty_char * (width - filled), "bright_black", color)
        + "]"
    )


def _sparkline(
    values: Sequence[Optional[int]],
    width: int,
    *,
    unicode: bool,
) -> str:
    symbols = SPARKLINE if unicode else ASCII_SPARKLINE
    usable = [value for value in values if value is not None]
    if not usable:
        return "?" * width
    padded: list[Optional[int]] = [None] * max(0, width - len(values)) + list(values[-width:])
    chars = []
    for value in padded:
        if value is None:
            chars.append(" ")
            continue
        idx = round(max(0, min(100, value)) / 100 * (len(symbols) - 1))
        chars.append(symbols[idx])
    return "".join(chars)


def _avg(values: Iterable[Optional[int]]) -> Optional[int]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid))


def _all_gpus(snapshot: ClusterSnapshot) -> list[GPUDevice]:
    return [gpu for node in snapshot.nodes for gpu in node.gpus]


def _gpu_key(gpu: GPUDevice) -> tuple[str, str]:
    return (gpu.node, gpu.uuid or str(gpu.index))


def _util_color(value: Optional[int], *, memory: bool = False) -> str:
    if value is None:
        return "yellow"
    moderate, heavy = (10, 80) if memory else (10, 75)
    if value >= heavy:
        return "bright_red"
    if value >= moderate:
        return "bright_yellow"
    return "bright_green"


def _style(text: object, color_name: str, enabled: bool) -> str:
    text = str(text)
    if not enabled:
        return text
    codes = []
    if color_name == "bold":
        codes.append(COLORS["bold"])
    else:
        codes.append(COLORS.get(color_name, ""))
    prefix = "".join(codes)
    return f"{prefix}{text}{RESET}" if prefix else text


def _rule(width: int, char: str, color: bool) -> str:
    return _style(char * min(width, 120), "bright_black", color)


def _clip(text: str, width: int) -> str:
    if width <= 1 or _visible_len(text) <= width:
        return text
    out = []
    visible = 0
    idx = 0
    while idx < len(text) and visible < width - 1:
        match = ANSI_RE.match(text, idx)
        if match:
            out.append(match.group(0))
            idx = match.end()
            continue
        out.append(text[idx])
        visible += 1
        idx += 1
    if ANSI_RE.search("".join(out)):
        out.append(RESET)
    return "".join(out) + "…"


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
