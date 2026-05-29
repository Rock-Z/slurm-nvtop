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

![Synthetic sgtop demo](assets/sgtop-demo.gif)

## Requirements

- `uv` for install/run
- Slurm commands: `squeue`, ideally `scontrol`
- SSH from the launch node to allocated compute nodes
- `nvidia-smi` on compute nodes

`sgtop` re-discovers jobs every refresh, keeps node polling errors visible, and
does not depend on cluster-specific node names or queue wrapper scripts.
