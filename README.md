# slurm-gpu-top

`slurm-gpu-top` is a lightweight, cluster-agnostic dashboard for GPU-backed
Slurm jobs. It can be launched from any node that can run Slurm client commands
and SSH to allocated compute nodes.

It discovers current running Slurm jobs, filters to GPU-enabled allocations,
polls each allocated node with `nvidia-smi`, and redraws a grouped dashboard in
the terminal. The TUI mirrors the default `nvitop` report: boxed GPU tables,
driver/CUDA versions, persistence mode, bus id, display/MIG/ECC state,
fan/temp/perf/power, memory and utilization bars, SM clock, host
CPU/MEM/SWAP/load/uptime, and a running GPU-process table.

## Quick start

```bash
python -m pip install -e .
sgtop
```

Useful modes:

```bash
sgtop --once
sgtop --interval 5
sgtop --color always
sgtop --no-unicode
sgtop --all-users
sgtop --ssh-option BatchMode=yes --ssh-option ConnectTimeout=4
```

By default, only the current user's running jobs are shown. Use `--all-users`
to monitor every visible running GPU job.

## Requirements

- Slurm client commands in `PATH`: `squeue`, and preferably `scontrol`
- SSH access from the launch node to allocated compute nodes
- NVIDIA GPUs on the compute nodes with `nvidia-smi` in `PATH`

The tool does not depend on a specific cluster naming scheme, GPU partition
name, or local `q`/`myqueue` wrapper. GPU jobs are identified from Slurm GRES
and TRES fields when available, with a conservative fallback that polls nodes
only for jobs Slurm reports as GPU-backed.

## Behavior

- Re-discovers Slurm jobs every refresh, so new jobs appear and ended jobs
  disappear without restarting the dashboard.
- Uses Unicode bars and ANSI colors by default on interactive terminals.
  Use `--color never` or `--no-unicode` for plain terminals and logs.
- Polls nodes concurrently with per-node timeouts, so one slow or dead node does
  not freeze the whole display.
- Keeps node-level errors visible in the dashboard instead of crashing.
- Uses stable CSV output from `nvidia-smi` instead of parsing the interactive
  `nvitop` UI.

## Development

```bash
env PYTHONPATH=src UV_CACHE_DIR=/tmp/uv-cache uv run pytest
python -m slurm_gpu_top --once --mock-json tests/fixtures/sample_snapshot.json
```

`--mock-json` accepts a saved snapshot shape and is intended for renderer
development and smoke tests on machines without Slurm or GPUs.

For a live smoke test without installing the package:

```bash
env PYTHONPATH=src python -m slurm_gpu_top --once
```
