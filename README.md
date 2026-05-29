# slurm-nvtop

`sgtop` is a nvitop-like TUI for GPU Slurm jobs across all currently allocated
nodes. Run it from any node with Slurm client commands and SSH access to the
allocated compute nodes.

## Install

```bash
uv tool install "git+ssh://git@github.com/Rock-Z/slurm-nvtop.git" && sgtop
```

One-off without installing:

```bash
uvx --from "git+ssh://git@github.com/Rock-Z/slurm-nvtop.git" sgtop
```

Local development:

```bash
uv run sgtop --mock-json tests/fixtures/sample_snapshot.json --color never
```

## Usage

```bash
sgtop                  # live dashboard; press q to quit
sgtop --once           # print one snapshot
sgtop --interval 5     # refresh every 5 seconds
sgtop --all-users      # show all visible GPU jobs
sgtop --no-unicode     # ASCII fallback
```

## Demo

```text
╒══════════════════════════════════════════════════════════════════════════════╕
│SGTOP 0.1.0 Driver Version: 570.195.03 CUDA Driver Version: 12.8 ALL GPUs... │
╞══════════════════════════════════════════════════════════════════════════════╡
│[r817u15n06] 1967875 ez275/train 03:56:10 -- CPU 23% MEM 11% LOAD 17.59 ... │
├────────────────────────┬────────────────┬──────────────────────────────────┤
│GPU  Name        Persist│MIG M.   Uncorr.│                                  │
│Fan  Temp  Perf  Pwr:Usa│        Memory-U│                                  │
╞════════════════════════╪════════════════╪══════════════════════════════════╡
│  0  H100 80GB HBM3   On│Disabled       0│MEM: ███████▌ 22003MiB 27%        │
│N/A   58C   P0 532W/700W│22003MiB/79.65G│UTL: ██████████████ 78% @ 1965MHz  │
╞════════════════════════╧════════════════╧══════════════════════════════════╡
│r817u15n06                                                                │
│MEM ↑ / UTL ↓                                                             │
│                    ⣀⣀⣰⣶                                                │
│╴60s├────────────╴30s├────────────now                                      │
│                    ⠹⠁ ⢸⣿                                                │
╘══════════════════════════════════════════════════════════════════════════════╛
```

## Requirements

- `uv` for install/run
- Slurm commands: `squeue`, ideally `scontrol`
- SSH from the launch node to allocated compute nodes
- `nvidia-smi` on compute nodes

`sgtop` re-discovers jobs every refresh, keeps node polling errors visible, and
does not depend on cluster-specific node names or queue wrapper scripts.
