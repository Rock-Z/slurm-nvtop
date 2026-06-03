from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import sys
import termios
import time
import tty
from typing import Optional, Sequence

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
    if argv is None and not args.once and not args.mock_json:
        _set_short_process_title("sgtop")
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
                gpu_util_histories=history.gpu_util_histories(),
                gpu_mem_histories=history.gpu_mem_histories(),
                version=__version__,
            )
        )
        return 0

    config = SnapshotBuilderConfig(
        user=args.user,
        all_users=args.all_users,
        command_timeout_s=args.timeout,
        max_workers=args.max_workers,
    )
    history = UtilizationHistory(maxlen=args.history)
    live_fullscreen = not args.once and not args.no_clear and sys.stdout.isatty()
    keyboard_enabled = live_fullscreen
    terminal_attrs = None

    try:
        if live_fullscreen:
            print("\033[?1049h\033[?25l\033[H\033[J", end="", flush=True)
        if keyboard_enabled and sys.stdin.isatty():
            terminal_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            attrs = termios.tcgetattr(sys.stdin.fileno())
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
        while True:
            snapshot = build_snapshot(config=config)
            history.record(snapshot)
            terminal_size = shutil.get_terminal_size(fallback=(120, 24))
            if not args.no_clear and not args.once:
                print("\033[H\033[J", end="")
            rendered = render_snapshot(
                snapshot,
                width=terminal_size.columns,
                height=terminal_size.lines if live_fullscreen else None,
                color=color,
                unicode=not args.no_unicode,
                gpu_util_histories=history.gpu_util_histories(),
                gpu_mem_histories=history.gpu_mem_histories(),
                version=__version__,
            )
            if live_fullscreen:
                sys.stdout.write(rendered)
                sys.stdout.flush()
            else:
                print(rendered, flush=True)
            if args.once:
                return 1 if snapshot.errors else 0
            if _sleep_or_quit(args.interval, keyboard_enabled=keyboard_enabled):
                return 0
    except KeyboardInterrupt:
        if sys.stdout.isatty():
            print()
        return 130
    finally:
        if terminal_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_attrs)
        if live_fullscreen:
            print("\033[?25h\033[?1049l", end="", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slurm-gpu-top",
        description="nvitop-like dashboard for GPU-backed Slurm allocations across nodes.",
    )
    parser.add_argument("--once", action="store_true", help="render one snapshot and exit")
    parser.add_argument("--interval", type=float, default=2.0, help="refresh interval in seconds")
    parser.add_argument("--user", default=os.environ.get("USER"), help="Slurm user to monitor")
    parser.add_argument("--all-users", action="store_true", help="show all visible running GPU jobs")
    parser.add_argument("--timeout", type=float, default=8.0, help="per-command timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=16, help="maximum concurrent job probes")
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


def _set_short_process_title(title: str) -> None:
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        pr_set_name = 15
        libc.prctl(pr_set_name, title.encode("utf-8")[:15], 0, 0, 0)
    except Exception:
        return


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


def _sleep_or_quit(interval: float, *, keyboard_enabled: bool) -> bool:
    if not keyboard_enabled:
        time.sleep(interval)
        return False

    deadline = time.monotonic() + max(0.0, interval)
    while True:
        timeout = max(0.0, min(0.1, deadline - time.monotonic()))
        readable, _writable, _errored = select.select([sys.stdin], [], [], timeout)
        if readable:
            char = sys.stdin.read(1)
            if char == "":
                if time.monotonic() >= deadline:
                    return False
                time.sleep(timeout)
                continue
            if char in {"q", "Q"}:
                return True
            if char == "\x03":
                raise KeyboardInterrupt
        if time.monotonic() >= deadline:
            return False
