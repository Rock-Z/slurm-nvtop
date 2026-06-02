from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Iterable, Mapping, Optional, Sequence

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
MIN_DASHBOARD_WIDTH = 80


def render_snapshot(
    snapshot: ClusterSnapshot,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    color: Optional[bool] = None,
    unicode: bool = True,
    all_gpu_history: Sequence[Optional[int]] = (),
    gpu_histories: Optional[dict[tuple[str, str], Sequence[Optional[int]]]] = None,
    node_util_histories: Optional[dict[str, Sequence[Optional[int]]]] = None,
    node_mem_histories: Optional[dict[str, Sequence[Optional[int]]]] = None,
    version: str = "0.2.1",
) -> str:
    del all_gpu_history, gpu_histories
    term_width, _term_height = _terminal_size()
    requested_width = width or term_width
    color = _should_color() if color is None else color
    if requested_width < MIN_DASHBOARD_WIDTH:
        return _render_too_narrow(
            requested_width,
            min_width=MIN_DASHBOARD_WIDTH,
            color=color,
            unicode=unicode,
            version=version,
        )
    width = requested_width
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

    gpu_lines = list(
        _cluster_gpu_box(
            snapshot,
            width=width,
            color=color,
            unicode=unicode,
            version=version,
            avg_util=avg_util,
            avg_mem=avg_mem,
            node_util_histories=node_util_histories or {},
            node_mem_histories=node_mem_histories or {},
        ),
    )
    process_lines = ["", *_cluster_process_box(snapshot, width=width, color=color, unicode=unicode)]
    lines.extend(gpu_lines)
    if height is None or len(lines) + len(process_lines) <= height:
        lines.extend(process_lines)
    lines = _fit_height(lines, height, color=color)
    return "\n".join(lines)


def _render_too_narrow(width: int, *, min_width: int, color: bool, unicode: bool, version: str) -> str:
    width = max(1, width)
    if width < 24:
        return _clip_visible(f"SGTOP {version}: need {min_width}+ columns", width)

    chars = _box_chars(unicode)
    inner = width - 2
    lines = [
        chars["tl"] + chars["h2"] * inner + chars["tr"],
        _box_line(_style(f"SGTOP {version}", "bold", color), inner, chars),
        chars["ml_bold"] + chars["h2"] * inner + chars["mr_bold"],
        _box_line(_style("Terminal width is too narrow.", "bright_red", color), inner, chars),
        _box_line(f"Current width: {width}; minimum: {min_width}.", inner, chars),
        _box_line(f"Resize to at least {min_width} columns.", inner, chars),
        chars["bl"] + chars["h2"] * inner + chars["br"],
    ]
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
    node_util_histories: Mapping[str, Sequence[Optional[int]]],
    node_mem_histories: Mapping[str, Sequence[Optional[int]]],
) -> Iterable[str]:
    chars = _box_chars(unicode)
    w = min(width, 140)
    inner = w - 2
    c1, c2 = _gpu_col_widths(inner)
    c3 = inner - c1 - c2 - 2
    gpu_widths = (c1, c2, c3)
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
    yield _column_transition_line(chars, (), gpu_widths)
    yield _row(
        ("GPU  Name        Persistence-M", "MIG M.   Uncorr. ECC", ""),
        gpu_widths,
        chars,
    )
    yield _row(
        ("Fan  Temp  Perf  Pwr:Usage/Cap", "        Memory-Usage", ""),
        gpu_widths,
        chars,
    )
    yield _column_transition_line(chars, gpu_widths, ())
    current_columns: tuple[int, ...] | None = None
    node_started = False
    for node in snapshot.nodes:
        if current_columns is None and node_started:
            yield chars["ml_bold"] + chars["h2"] * inner + chars["mr_bold"]
        elif current_columns is not None:
            yield _column_transition_line(chars, current_columns, ())
            current_columns = None
        node_started = True
        yield _box_line(_node_status(node, color=color), inner, chars)
        if node.error:
            yield _box_line(_style(f"! {node.error}", "bright_red", color), inner, chars)
            continue
        yield _joint_line(chars, gpu_widths, "top")
        if not node.gpus:
            yield _box_line(_style("No GPUs reported by nvidia-smi.", "yellow", color), inner, chars)
            continue
        for gpu_idx, gpu in enumerate(sorted(node.gpus, key=lambda item: item.index)):
            if gpu_idx:
                yield _joint_line(chars, gpu_widths, "mid")
            yield from _gpu_rows(gpu, gpu_widths, chars, color=color, unicode=unicode)
        current_columns = gpu_widths
    history_widths, history_rows = _node_history_layout(
        snapshot,
        width=inner,
        color=color,
        unicode=unicode,
        node_util_histories=node_util_histories,
        node_mem_histories=node_mem_histories,
    )
    if history_rows:
        yield _column_transition_line(chars, current_columns or (), history_widths)
        yield from _node_history_rows(history_rows, history_widths, chars)
        current_columns = tuple(history_widths)
    if current_columns is None:
        yield chars["bl"] + chars["h2"] * inner + chars["br"]
    else:
        yield _joint_line(chars, current_columns, "bottom")


def _gpu_col_widths(inner: int) -> tuple[int, int]:
    if inner >= 118:
        return 31, 22
    if inner >= 98:
        return 28, 19
    return 24, 16


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
            f"{_short(gpu.mig_mode, 8):<8} {_na_int(gpu.ecc_errors):>11}",
            _bar_stat(
                "MEM",
                mem_percent,
                _mem_usage_short(gpu),
                color=color,
                unicode=unicode,
                width=_stat_bar_width(widths[-1]),
                suffix_width=_stat_suffix_width(widths[-1]),
            ),
        ),
        widths,
        chars,
    )
    yield _row(
        (
            f"{_fan(gpu):>3}  {_temp(gpu):>4} {_short(gpu.performance_state, 4):>4} {_power(gpu):>13}",
            f"{_mem_usage(gpu):>21}",
            _bar_stat(
                "UTL",
                util,
                f"{_percent(util):>4} @ {_clock(gpu)}",
                color=color,
                unicode=unicode,
                width=_stat_bar_width(widths[-1]),
                suffix_width=_stat_suffix_width(widths[-1]),
            ),
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


def _node_history_layout(
    snapshot: ClusterSnapshot,
    *,
    width: int,
    color: bool,
    unicode: bool,
    node_util_histories: Mapping[str, Sequence[Optional[int]]],
    node_mem_histories: Mapping[str, Sequence[Optional[int]]],
) -> tuple[list[int], list[list[str]]]:
    nodes = [
        node
        for node in snapshot.nodes
        if node.gpus and (node.node in node_util_histories or node.node in node_mem_histories)
    ]
    if not nodes:
        return [], []

    inner = width
    max_cells = max(1, (inner + 1) // 18)
    omitted = max(0, len(nodes) - max_cells)
    nodes = nodes[:max_cells]
    node_labels = [node.node for node in nodes]
    if omitted:
        node_labels[-1] = f"{node_labels[-1]} (+{omitted})"
    cell_widths = _split_widths(inner, len(nodes))
    side_height = max(_node_history_side_height(width) for width in cell_widths)

    return cell_widths, [
        _node_history_cell(
            node_labels[idx],
            width=cell_widths[idx],
            util_history=node_util_histories.get(node.node, ()),
            mem_history=node_mem_histories.get(node.node, ()),
            label=idx == 0,
            color=color,
            unicode=unicode,
            side_height=side_height,
        )
        for idx, node in enumerate(nodes)
    ]


def _node_history_rows(
    cells: Sequence[Sequence[str]],
    cell_widths: Sequence[int],
    chars: dict[str, str],
) -> Iterable[str]:
    row_count = max(len(cell) for cell in cells)
    for row_idx in range(row_count):
        yield _multi_cell_row(
            [cell[row_idx] if row_idx < len(cell) else "" for cell in cells],
            cell_widths,
            chars,
        )


def _node_history_cell(
    node: str,
    *,
    width: int,
    util_history: Sequence[Optional[int]],
    mem_history: Sequence[Optional[int]],
    label: bool,
    color: bool,
    unicode: bool,
    side_height: Optional[int] = None,
) -> list[str]:
    side_height = _node_history_side_height(width) if side_height is None else side_height
    label_text = "MEM ↑ / UTL ↓" if unicode else "MEM up / UTL down"
    mem_graph = _history_graph_lines(mem_history, width=width, height=side_height, direction="up", unicode=unicode)
    util_graph = _history_graph_lines(
        util_history,
        width=width,
        height=side_height + 1,
        direction="down",
        unicode=unicode,
    )
    if label and mem_graph:
        mem_graph[0] = _overlay_left(mem_graph[0], label_text, width)
    return [
        _style(_clip_visible(node, width), "bright_cyan", color),
        *_style_lines(mem_graph, "magenta", color),
        _style(_timeline_axis(width, unicode=unicode), "bright_black", color),
        *_style_lines(util_graph, "cyan", color),
    ]


def _node_history_side_height(width: int) -> int:
    return 3 if width < 34 else 4


def _history_graph_lines(
    values: Sequence[Optional[int]],
    *,
    width: int,
    height: int,
    direction: str,
    unicode: bool,
) -> list[str]:
    if width <= 0 or height <= 0:
        return []
    if not unicode:
        return _ascii_history_graph(values, width=width, height=height, direction=direction)
    samples = _history_samples(values, width * 2)
    levels = [_history_level(value, height) for value in samples]
    lines = []
    for row in range(height):
        chars = []
        for idx in range(0, len(levels), 2):
            chars.append(_braille_history_cell(levels[idx], levels[idx + 1], row, height, direction=direction))
        lines.append("".join(chars))
    return lines


def _ascii_history_graph(
    values: Sequence[Optional[int]],
    *,
    width: int,
    height: int,
    direction: str,
) -> list[str]:
    samples = _history_samples(values, width)
    levels = [_history_level(value, height) for value in samples]
    lines = []
    for row in range(height):
        chars = []
        for level in levels:
            if direction == "up":
                row_from_bottom = height - 1 - row
                chars.append("#" if level > row_from_bottom * 4 else " ")
            else:
                row_from_top = row
                chars.append("#" if level > row_from_top * 4 else " ")
        lines.append("".join(chars))
    return lines


def _history_samples(values: Sequence[Optional[int]], count: int) -> list[Optional[int]]:
    recent = list(values)[-count:]
    return [None] * max(0, count - len(recent)) + recent


def _history_level(value: Optional[int], height: int) -> int:
    if value is None:
        return 0
    value = max(0.0, min(100.0, float(value)))
    if value == 0.0:
        return 0
    return max(1, round(value / 100.0 * height * 4))


def _braille_history_cell(left_level: int, right_level: int, row: int, height: int, *, direction: str) -> str:
    left_count = _subcell_count(left_level, row, height, direction=direction)
    right_count = _subcell_count(right_level, row, height, direction=direction)
    mask = _braille_mask(left_count, side="left", direction=direction)
    mask |= _braille_mask(right_count, side="right", direction=direction)
    return " " if mask == 0 else chr(0x2800 + mask)


def _subcell_count(level: int, row: int, height: int, *, direction: str) -> int:
    if direction == "up":
        base = (height - 1 - row) * 4
    else:
        base = row * 4
    return min(max(level - base, 0), 4)


def _braille_mask(count: int, *, side: str, direction: str) -> int:
    if count <= 0:
        return 0
    if side == "left":
        dots = (7, 3, 2, 1) if direction == "up" else (1, 2, 3, 7)
    else:
        dots = (8, 6, 5, 4) if direction == "up" else (4, 5, 6, 8)
    mask = 0
    for dot in dots[:count]:
        mask |= 1 << (dot - 1)
    return mask


def _timeline_axis(width: int, *, unicode: bool) -> str:
    axis = "─" if unicode else "-"
    line = [axis] * width
    labels = [(3, "now")]
    for seconds in (30, 60, *range(120, 60 * width + 1, 60)):
        label = f"╴{seconds}s├" if unicode else f"-{seconds}s|"
        labels.append((seconds // 2 + len(label) - 1, label))
    occupied_until = width + 1
    for offset, label in labels:
        if offset > width:
            continue
        start = max(0, width - offset)
        if start + len(label) > width or start + len(label) > occupied_until:
            continue
        line[start : start + len(label)] = label
        occupied_until = start
    return "".join(line)


def _style_lines(lines: Sequence[str], color_name: str, enabled: bool) -> list[str]:
    return [_style(line, color_name, enabled) for line in lines]


def _overlay_left(line: str, label: str, width: int) -> str:
    label = _clip_visible(label, width)
    return (label + line[_visible_len(label) :])[:width]


def _split_widths(inner: int, count: int) -> list[int]:
    if count <= 0:
        return []
    available = max(count, inner - (count - 1))
    base, remainder = divmod(available, count)
    return [base + (1 if idx < remainder else 0) for idx in range(count)]


def _multi_cell_row(values: Sequence[str], widths: Sequence[int], chars: dict[str, str]) -> str:
    return chars["v"] + chars["v"].join(_pad(value, width) for value, width in zip(values, widths)) + chars["v"]


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
    process_rows = [_cluster_process_row_data(node, gpu, proc) for node, gpu, proc in rows]
    process_columns, command_width = _process_table_layout(process_rows, inner)
    yield _box_line(_format_process_header(process_columns, command_width), inner, chars)
    yield chars["ml_bold"] + chars["h2"] * inner + chars["mr_bold"]
    if not rows:
        yield _box_line(_style("No running GPU compute processes found.", "bright_black", color), inner, chars)
    for idx, row in enumerate(process_rows):
        if idx:
            yield chars["ml"] + chars["h1"] * inner + chars["mr"]
        yield _box_line(_format_process_table_row(row, process_columns, command_width), inner, chars)
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


def _cluster_process_row_data(node: NodeSnapshot, gpu: GPUDevice, proc: GPUProcess) -> dict[str, str]:
    return {
        "node": node.node,
        "gpu": str(gpu.index),
        "pid": str(proc.pid),
        "type": _short(proc.type, 3),
        "user": proc.user or "N/A",
        "gpu_mem": _value(proc.used_memory_mib, "MiB"),
        "sm": _na_int(proc.sm_util_percent),
        "gmbw": _na_int(proc.mem_bw_util_percent),
        "cpu": _na_float(proc.cpu_percent),
        "mem": _na_float(proc.mem_percent),
        "time": proc.elapsed or "N/A",
        "command": proc.command or proc.name or "N/A",
    }


def _process_table_layout(
    rows: Sequence[Mapping[str, str]],
    inner: int,
) -> tuple[list[tuple[str, str, str, int]], int]:
    specs = [
        ("node", "NODE", "left", 4, 24),
        ("gpu", "GPU", "right", 3, 4),
        ("pid", "PID", "right", 3, 10),
        ("type", "T", "left", 1, 3),
        ("user", "USER", "left", 4, 14),
        ("gpu_mem", "GPU-MEM", "right", 5, 10),
        ("sm", "%SM", "right", 3, 4),
        ("gmbw", "%GMBW", "right", 5, 5),
        ("cpu", "%CPU", "right", 4, 6),
        ("mem", "%MEM", "right", 4, 5),
        ("time", "TIME", "right", 4, 10),
    ]
    widths = {
        key: min(max_width, max(min_width, _visible_len(label), *(len(row.get(key, "")) for row in rows)))
        for key, label, _align, min_width, max_width in specs
    }

    def fixed_width() -> int:
        return sum(widths[key] for key, *_rest in specs) + len(specs) - 1 + 2

    shrink_order = ("node", "user", "time", "pid", "gpu_mem", "cpu", "mem", "sm", "gmbw", "type", "gpu")
    min_widths = {key: min_width for key, _label, _align, min_width, _max_width in specs}
    while fixed_width() > inner:
        for key in shrink_order:
            if widths[key] > min_widths[key]:
                widths[key] -= 1
                break
        else:
            break

    columns = [(key, label, align, widths[key]) for key, label, align, _min_width, _max_width in specs]
    return columns, max(0, inner - fixed_width())


def _format_process_header(columns: Sequence[tuple[str, str, str, int]], command_width: int) -> str:
    return _format_process_table_row(
        {key: label for key, label, _align, _width in columns} | {"command": "COMMAND"},
        columns,
        command_width,
    )


def _format_process_table_row(
    row: Mapping[str, str],
    columns: Sequence[tuple[str, str, str, int]],
    command_width: int,
) -> str:
    cells = []
    for key, _label, align, width in columns:
        value = _clip_visible(row.get(key, ""), width)
        padding = " " * max(0, width - _visible_len(value))
        cells.append(padding + value if align == "right" else value + padding)
    fixed = " ".join(cells)
    if command_width <= 0:
        return fixed
    return fixed + "  " + _clip_visible(row.get("command", ""), command_width)


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
        c1, c2 = 31, 22
    else:
        c1, c2 = 24, 16
    c3 = max(12, inner - c1 - c2 - 2)
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
    yield _joint_line(chars, (c1, c2, c3), "top")
    yield _row(("GPU  Name        Persistence-M", "MIG M.   Uncorr. ECC", ""), (c1, c2, c3), chars)
    yield _row(("Fan  Temp  Perf  Pwr:Usage/Cap", "        Memory-Usage", ""), (c1, c2, c3), chars)
    yield _joint_line(chars, (c1, c2, c3), "mid")

    if not gpus:
        yield _box_line(_style("No GPUs reported by nvidia-smi.", "yellow", color), inner, chars)
    for idx, gpu in enumerate(sorted(gpus, key=lambda item: item.index)):
        if idx:
            yield _joint_line(chars, (c1, c2, c3), "mid")
        mem_percent = _memory_percent(gpu)
        util = gpu.gpu_util_percent
        yield _row(
            (
                f"{gpu.index:>3}  {_gpu_name(gpu.name):<17.17} {_short(_on_off(gpu.persistence_mode), 3):>8}",
                f"{_short(gpu.mig_mode, 8):<8} {_na_int(gpu.ecc_errors):>11}",
                _bar_stat(
                    "MEM",
                    mem_percent,
                    _mem_usage_short(gpu),
                    color=color,
                    unicode=unicode,
                    width=_stat_bar_width(c3),
                    suffix_width=_stat_suffix_width(c3),
                ),
            ),
            (c1, c2, c3),
            chars,
        )
        yield _row(
            (
                f"{_fan(gpu):>3}  {_temp(gpu):>4} {_short(gpu.performance_state, 4):>4} {_power(gpu):>13}",
                f"{_mem_usage(gpu):>21}",
                _bar_stat(
                    "UTL",
                    util,
                    f"{_percent(util):>4} @ {_clock(gpu)}",
                    color=color,
                    unicode=unicode,
                    width=_stat_bar_width(c3),
                    suffix_width=_stat_suffix_width(c3),
                ),
            ),
            (c1, c2, c3),
            chars,
        )
    yield _joint_line(chars, (c1, c2, c3), "bottom")


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


def _bar_stat(
    label: str,
    percent: Optional[float],
    suffix: str,
    *,
    color: bool,
    unicode: bool,
    width: int = 8,
    suffix_width: int = 14,
) -> str:
    bar = _fraction_bar(percent, width=width, color=color, unicode=unicode)
    suffix = _clip_visible(str(suffix).strip(), suffix_width)
    return f"{label}: {bar} {suffix:<{suffix_width}}"


def _stat_bar_width(column_width: int) -> int:
    return max(8, min(28, column_width - 24))


def _stat_suffix_width(column_width: int) -> int:
    return max(12, min(18, column_width - _stat_bar_width(column_width) - 7))


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


def _column_transition_line(
    chars: dict[str, str],
    upper_widths: Sequence[int],
    lower_widths: Sequence[int],
) -> str:
    width = sum(upper_widths) + max(0, len(upper_widths) - 1)
    if not upper_widths:
        width = sum(lower_widths) + max(0, len(lower_widths) - 1)
    upper_joints = _column_joint_positions(upper_widths)
    lower_joints = _column_joint_positions(lower_widths)
    positions = upper_joints | lower_joints
    body = []
    for pos in range(1, width + 1):
        if pos in positions:
            if pos in upper_joints and pos in lower_joints:
                body.append(chars["mj_bold"])
            elif pos in upper_joints:
                body.append(chars["bj"])
            else:
                body.append(chars["tj_bold"])
        else:
            body.append(chars["h2"])
    return chars["ml_bold"] + "".join(body) + chars["mr_bold"]


def _column_joint_positions(widths: Sequence[int]) -> set[int]:
    positions = set()
    offset = 0
    for width in widths[:-1]:
        offset += width + 1
        positions.add(offset)
    return positions


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
            "tj_bold": "+",
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
        "tj_bold": "╤",
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


def _fit_height(lines: list[str], height: Optional[int], *, color: bool) -> list[str]:
    if height is None or height <= 0 or len(lines) <= height:
        return lines
    if height == 1:
        return [_style("Output truncated: terminal is too short.", "yellow", color)]
    omitted = len(lines) - height + 1
    footer = _style(f"... {omitted} lines hidden; enlarge terminal for full view ...", "yellow", color)
    return lines[: height - 1] + [footer]


def _should_color() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("ANSI_COLORS_DISABLED"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    term = os.environ.get("TERM", "")
    return bool(term and term != "dumb" and sys.stdout.isatty())


def _terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(120, 24))
    return size.columns, size.lines
