from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

from .dashboard import build_snapshot
from .history import UtilizationHistory
from . import __version__
from .models import (
    ClusterSnapshot,
    GPUDevice,
    GPUProcess,
    HostStats,
    NodeSnapshot,
    SlurmJob,
    SnapshotBuilderConfig,
)
from .render import render_snapshot


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    color = _color_enabled(args.color)
    if args.mock_json:
        snapshot = _load_mock_snapshot(args.mock_json)
        history = UtilizationHistory(maxlen=args.history)
        history.record(snapshot)
        print(
            render_snapshot(
                snapshot,
                color=color,
                unicode=not args.no_unicode,
                all_gpu_history=history.all_history(),
                gpu_histories={key: tuple(values) for key, values in history.by_gpu.items()},
                version=__version__,
            )
        )
        return 0

    config = SnapshotBuilderConfig(
        user=args.user,
        all_users=args.all_users,
        ssh_options=tuple(args.ssh_option),
        command_timeout_s=args.timeout,
        max_workers=args.max_workers,
    )
    history = UtilizationHistory(maxlen=args.history)

    try:
        if not args.once and not args.no_clear and sys.stdout.isatty():
            print("\033[?25l", end="", flush=True)
        while True:
            snapshot = build_snapshot(config=config)
            history.record(snapshot)
            if not args.no_clear and not args.once:
                print("\033[H\033[J", end="")
            print(
                render_snapshot(
                    snapshot,
                    color=color,
                    unicode=not args.no_unicode,
                    all_gpu_history=history.all_history(),
                    gpu_histories={key: tuple(values) for key, values in history.by_gpu.items()},
                    version=__version__,
                ),
                flush=True,
            )
            if args.once:
                return 1 if snapshot.errors else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if sys.stdout.isatty():
            print()
        return 130
    finally:
        if not args.once and not args.no_clear and sys.stdout.isatty():
            print("\033[?25h", end="", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slurm-gpu-top",
        description="nvitop-like dashboard for GPU-backed Slurm allocations across nodes.",
    )
    parser.add_argument("--once", action="store_true", help="render one snapshot and exit")
    parser.add_argument("--interval", type=float, default=2.0, help="refresh interval in seconds")
    parser.add_argument("--user", default=os.environ.get("USER"), help="Slurm user to monitor")
    parser.add_argument("--all-users", action="store_true", help="show all visible running GPU jobs")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=["BatchMode=yes", "ConnectTimeout=5"],
        help="SSH -o option for node polling; may be passed multiple times",
    )
    parser.add_argument("--timeout", type=float, default=8.0, help="per-command timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=16, help="maximum concurrent node polls")
    parser.add_argument("--no-clear", action="store_true", help="do not clear the terminal between refreshes")
    parser.add_argument("--no-unicode", action="store_true", help="use ASCII fallbacks instead of Unicode bars")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="color output policy",
    )
    parser.add_argument("--history", type=int, default=120, help="number of utilization samples to keep")
    parser.add_argument("--mock-json", help="render a saved JSON snapshot instead of polling Slurm")
    return parser.parse_args(argv)


def _color_enabled(policy: str) -> Optional[bool]:
    if policy == "always":
        return True
    if policy == "never":
        return False
    return None


def _load_mock_snapshot(path: str) -> ClusterSnapshot:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    nodes = []
    for node_data in data.get("nodes", []):
        jobs = tuple(SlurmJob(**job) for job in node_data.get("jobs", []))
        gpus = []
        for gpu_data in node_data.get("gpus", []):
            processes = tuple(GPUProcess(**proc) for proc in gpu_data.pop("processes", []))
            gpus.append(GPUDevice(**gpu_data, processes=processes))
        nodes.append(
            NodeSnapshot(
                node=node_data["node"],
                jobs=jobs,
                gpus=tuple(gpus),
                host=HostStats(**node_data.get("host", {})),
                error=node_data.get("error"),
            )
        )
    return ClusterSnapshot(
        nodes=tuple(nodes),
        errors=tuple(data.get("errors", [])),
        generated_at=float(data.get("generated_at", time.time())),
        user_filter=data.get("user_filter"),
    )
