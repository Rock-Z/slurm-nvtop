from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from .commands import CommandRunner, run_command
from .models import SlurmJob

SQUEUE_GPU_FORMAT = "%A|%j|%u|%T|%M|%D|%R|%b|%G"
SQUEUE_FALLBACK_FORMAT = "%A|%j|%u|%T|%M|%D|%R|%G"


class SlurmError(RuntimeError):
    pass


def discover_gpu_jobs(
    *,
    user: Optional[str] = None,
    all_users: bool = False,
    runner: CommandRunner = run_command,
    timeout: float = 8.0,
) -> Tuple[SlurmJob, ...]:
    jobs = _query_squeue(user=user, all_users=all_users, runner=runner, timeout=timeout)
    gpu_jobs: List[SlurmJob] = []
    for job in jobs:
        detailed = _with_scontrol_details(job, runner=runner, timeout=timeout)
        if not detailed.gpu_hint:
            continue
        nodes = expand_nodelist(detailed.nodelist, runner=runner, timeout=timeout)
        gpu_jobs.append(
            SlurmJob(
                job_id=detailed.job_id,
                name=detailed.name,
                user=detailed.user,
                state=detailed.state,
                elapsed=detailed.elapsed,
                node_count=detailed.node_count,
                nodelist=detailed.nodelist,
                gres=detailed.gres,
                tres=detailed.tres,
                nodes=tuple(nodes),
            )
        )
    return tuple(gpu_jobs)


def _query_squeue(
    *,
    user: Optional[str],
    all_users: bool,
    runner: CommandRunner,
    timeout: float,
) -> Tuple[SlurmJob, ...]:
    effective_user = user or os.environ.get("USER")
    base = ["squeue", "--states=RUNNING", "--noheader"]
    if not all_users and effective_user:
        base.extend(["--user", effective_user])

    result = runner([*base, "--format", SQUEUE_GPU_FORMAT], timeout)
    parser = parse_squeue_line
    if not result.ok:
        fallback = runner([*base, "--format", SQUEUE_FALLBACK_FORMAT], timeout)
        if not fallback.ok:
            message = fallback.stderr.strip() or result.stderr.strip() or "squeue failed"
            raise SlurmError(message)
        result = fallback
        parser = parse_squeue_fallback_line

    parsed = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            parsed.append(parser(line))
    return tuple(parsed)


def parse_squeue_line(line: str) -> SlurmJob:
    parts = line.split("|", 8)
    if len(parts) != 9:
        raise ValueError(f"unexpected squeue line with {len(parts)} fields: {line!r}")
    job_id, name, user, state, elapsed, node_count, nodelist, tres_per_node, gres = parts
    return SlurmJob(
        job_id=job_id.strip(),
        name=name.strip(),
        user=user.strip(),
        state=state.strip(),
        elapsed=elapsed.strip(),
        node_count=_parse_optional_int(node_count),
        nodelist=nodelist.strip(),
        gres=gres.strip(),
        tres=tres_per_node.strip(),
    )


def parse_squeue_fallback_line(line: str) -> SlurmJob:
    parts = line.split("|", 7)
    if len(parts) != 8:
        raise ValueError(f"unexpected fallback squeue line with {len(parts)} fields: {line!r}")
    job_id, name, user, state, elapsed, node_count, nodelist, gres = parts
    return SlurmJob(
        job_id=job_id.strip(),
        name=name.strip(),
        user=user.strip(),
        state=state.strip(),
        elapsed=elapsed.strip(),
        node_count=_parse_optional_int(node_count),
        nodelist=nodelist.strip(),
        gres=gres.strip(),
    )


def _with_scontrol_details(
    job: SlurmJob,
    *,
    runner: CommandRunner,
    timeout: float,
) -> SlurmJob:
    result = runner(["scontrol", "show", "job", "-o", job.job_id], timeout)
    if not result.ok:
        return job

    fields = parse_scontrol_key_values(result.stdout)
    gres = " ".join(
        value
        for key, value in fields.items()
        if key.lower() in {"gres", "gresdetail"}
    )
    tres = " ".join(
        value
        for key, value in fields.items()
        if key.lower() in {"trespernode", "alloctres", "tresalloc"}
    )
    nodelist = fields.get("NodeList", job.nodelist)
    return SlurmJob(
        job_id=job.job_id,
        name=fields.get("JobName", job.name),
        user=job.user,
        state=fields.get("JobState", job.state),
        elapsed=fields.get("RunTime", job.elapsed),
        node_count=_parse_optional_int(fields.get("NumNodes", "")) or job.node_count,
        nodelist=nodelist,
        gres=" ".join(part for part in (job.gres, gres) if part).strip(),
        tres=" ".join(part for part in (job.tres, tres) if part).strip(),
        nodes=job.nodes,
    )


def parse_scontrol_key_values(text: str) -> dict:
    fields = {}
    for match in re.finditer(r"(\S+?)=([^=]*?)(?=\s+\S+?=|$)", text.strip()):
        fields[match.group(1)] = match.group(2).strip()
    return fields


def expand_nodelist(
    nodelist: str,
    *,
    runner: CommandRunner = run_command,
    timeout: float = 8.0,
) -> Tuple[str, ...]:
    nodelist = nodelist.strip()
    if not nodelist or nodelist in {"(null)", "None", "N/A"}:
        return ()

    result = runner(["scontrol", "show", "hostnames", nodelist], timeout)
    if result.ok:
        nodes = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if nodes:
            return nodes

    return tuple(_expand_nodelist_locally(nodelist))


def _expand_nodelist_locally(nodelist: str) -> List[str]:
    expanded: List[str] = []
    for chunk in _split_top_level_commas(nodelist):
        expanded.extend(_expand_bracket_expr(chunk))
    return expanded


def _split_top_level_commas(value: str) -> List[str]:
    chunks: List[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(value):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            chunks.append(value[start:idx])
            start = idx + 1
    chunks.append(value[start:])
    return [chunk for chunk in (c.strip() for c in chunks) if chunk]


def _expand_bracket_expr(expr: str) -> List[str]:
    open_idx = expr.find("[")
    if open_idx == -1:
        return [expr]
    close_idx = expr.find("]", open_idx)
    if close_idx == -1:
        return [expr]

    prefix = expr[:open_idx]
    body = expr[open_idx + 1 : close_idx]
    suffix = expr[close_idx + 1 :]
    values: List[str] = []
    for item in body.split(","):
        values.extend(_expand_range(item.strip()))

    expanded: List[str] = []
    for value in values:
        for tail in _expand_bracket_expr(suffix):
            expanded.append(f"{prefix}{value}{tail}")
    return expanded


def _expand_range(item: str) -> Iterable[str]:
    if "-" not in item:
        return [item]
    start_s, end_s = item.split("-", 1)
    if not (start_s.isdigit() and end_s.isdigit()):
        return [item]
    width = max(len(start_s), len(end_s))
    start = int(start_s)
    end = int(end_s)
    step = 1 if end >= start else -1
    return [f"{num:0{width}d}" for num in range(start, end + step, step)]


def _parse_optional_int(value: object) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except ValueError:
        return None
