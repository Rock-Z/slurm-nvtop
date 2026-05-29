import json

from slurm_gpu_top.cli import main
from slurm_gpu_top.dashboard import build_snapshot
from slurm_gpu_top.history import UtilizationHistory
from slurm_gpu_top.models import ClusterSnapshot, CommandResult, NodeSnapshot, SnapshotBuilderConfig
from slurm_gpu_top.render import render_snapshot


def test_build_snapshot_rediscovers_jobs_each_time_for_additions_and_ends():
    squeue_outputs = [
        "101|first|ez275|RUNNING|0:01|1|gpu001|gres/gpu:1|gpu:1",
        "202|second|ez275|RUNNING|0:02|1|gpu002|gres/gpu:1|gpu:1",
    ]

    def runner(args, timeout):
        command = tuple(args)
        if command[:1] == ("squeue",):
            return CommandResult(command, 0, squeue_outputs.pop(0))
        if command[:3] == ("scontrol", "show", "job"):
            job_id = command[-1]
            node = "gpu001" if job_id == "101" else "gpu002"
            return CommandResult(command, 0, f"JobId={job_id} AllocTRES=gres/gpu=1 NodeList={node}")
        if command[:3] == ("scontrol", "show", "hostnames"):
            return CommandResult(command, 0, command[-1] + "\n")
        if command[:1] == ("ssh",):
            return CommandResult(
                command,
                0,
                "0, GPU-uuid, NVIDIA A40, 10, 20, 1000, 46068, 40, 80, 300\n",
            )
        raise AssertionError(command)

    config = SnapshotBuilderConfig(user="ez275", max_workers=1, now=10)
    first = build_snapshot(config=config, runner=runner)
    second = build_snapshot(config=config, runner=runner)

    assert [node.node for node in first.nodes] == ["gpu001"]
    assert [node.node for node in second.nodes] == ["gpu002"]
    assert first.job_count == second.job_count == 1


def test_build_snapshot_keeps_node_poll_errors_visible():
    def runner(args, timeout):
        command = tuple(args)
        if command[:1] == ("squeue",):
            return CommandResult(command, 0, "101|job|ez275|RUNNING|0:01|1|gpu001|gres/gpu:1|gpu:1")
        if command[:3] == ("scontrol", "show", "job"):
            return CommandResult(command, 0, "JobId=101 AllocTRES=gres/gpu=1 NodeList=gpu001")
        if command[:3] == ("scontrol", "show", "hostnames"):
            return CommandResult(command, 0, "gpu001\n")
        if command[:1] == ("ssh",):
            return CommandResult(command, 255, "", "connection refused")
        raise AssertionError(command)

    snapshot = build_snapshot(config=SnapshotBuilderConfig(user="ez275", now=10), runner=runner)
    rendered = render_snapshot(snapshot, width=120)

    assert snapshot.nodes[0].error == "connection refused"
    assert "connection refused" in rendered


def test_render_snapshot_groups_by_node_and_lists_processes():
    def runner(args, timeout):
        command = tuple(args)
        if command[:1] == ("squeue",):
            return CommandResult(
                command,
                0,
                "101|train|ez275|RUNNING|1:00|1|gpu001|gres/gpu:2|gpu:a100:2",
            )
        if command[:3] == ("scontrol", "show", "job"):
            return CommandResult(command, 0, "JobId=101 JobName=train AllocTRES=gres/gpu=2 NodeList=gpu001")
        if command[:3] == ("scontrol", "show", "hostnames"):
            return CommandResult(command, 0, "gpu001\n")
        if command[:1] == ("ssh",):
            return CommandResult(
                command,
                0,
                "0, GPU-a, NVIDIA A100, 75, 50, 40000, 81920, 60, 250, 400\n"
                "__SLURM_GPU_TOP_PROCESSES__\n"
                "1234, python train.py, 39000, GPU-a\n",
            )
        raise AssertionError(command)

    snapshot = build_snapshot(config=SnapshotBuilderConfig(user="ez275", now=10), runner=runner)
    rendered = render_snapshot(snapshot, width=120)

    assert "gpu001" in rendered
    assert "101 ez275/train" in rendered
    assert "A100" in rendered
    assert "1234" in rendered


def test_render_snapshot_rich_mode_has_color_graphs_averages_and_process_table():
    snapshot = _single_gpu_snapshot(util=83, mem=40)
    history = UtilizationHistory(maxlen=8)
    for util in (0, 20, 40, 60, 83):
        history.record(_single_gpu_snapshot(util=util, mem=40))

    rendered = render_snapshot(
        snapshot,
        width=120,
        color=True,
        unicode=True,
        all_gpu_history=history.all_history(),
        gpu_histories={key: tuple(values) for key, values in history.by_gpu.items()},
    )

    assert "\x1b[" in rendered
    assert "ALL GPUs" in rendered
    assert "NVITOP" in rendered
    assert "Driver Version" in rendered
    assert "Memory-Usage" in rendered
    assert "GPU-Util" in rendered
    assert "MEM:" in rendered
    assert "UTL:" in rendered
    assert "CPU:" in rendered
    assert "SWP:" in rendered
    assert "Processes" in rendered
    assert "python train.py" in rendered


def test_render_snapshot_ascii_fallback_keeps_graph_visible():
    snapshot = _single_gpu_snapshot(util=75, mem=50)
    rendered = render_snapshot(
        snapshot,
        width=100,
        color=False,
        unicode=False,
        all_gpu_history=(0, 25, 50, 75),
        gpu_histories={("gpu001", "GPU-a"): (0, 25, 50, 75)},
    )

    assert "\x1b[" not in rendered
    assert "NVITOP" in rendered
    assert "MEM:" in rendered
    assert "UTL:" in rendered
    assert "#" in rendered
    assert "█" not in rendered
    assert "░" not in rendered
    assert "Processes" in rendered


def test_render_snapshot_compact_width_keeps_history_graph_visible():
    snapshot = _single_gpu_snapshot(util=75, mem=50)
    rendered = render_snapshot(
        snapshot,
        width=80,
        color=False,
        unicode=True,
        all_gpu_history=(0, 25, 50, 75),
        gpu_histories={("gpu001", "GPU-a"): (0, 25, 50, 75)},
    )

    assert "NVITOP" in rendered
    assert "MEM:" in rendered
    assert "UTL:" in rendered
    assert "A100" in rendered


def test_render_snapshot_handles_empty_state():
    rendered = render_snapshot(ClusterSnapshot(generated_at=10), width=80)

    assert "No running GPU-backed Slurm jobs found." in rendered


def _single_gpu_snapshot(*, util: int, mem: int) -> ClusterSnapshot:
    return build_snapshot(
        config=SnapshotBuilderConfig(user="ez275", now=10, max_workers=1),
        runner=_single_gpu_runner(util=util, mem=mem),
    )


def _single_gpu_runner(*, util: int, mem: int):
    def runner(args, timeout):
        command = tuple(args)
        if command[:1] == ("squeue",):
            return CommandResult(
                command,
                0,
                "101|train|ez275|RUNNING|1:00|1|gpu001|gres/gpu:2|gpu:a100:2",
            )
        if command[:3] == ("scontrol", "show", "job"):
            return CommandResult(command, 0, "JobId=101 JobName=train AllocTRES=gres/gpu=2 NodeList=gpu001")
        if command[:3] == ("scontrol", "show", "hostnames"):
            return CommandResult(command, 0, "gpu001\n")
        if command[:1] == ("ssh",):
            return CommandResult(
                command,
                0,
                "__SLURM_GPU_TOP_META__\n"
                "hostname=gpu001.example\n"
                "driver_version=570.195.03\n"
                "cuda_version=12.8\n"
                "uptime_seconds=1572480\n"
                "load_average=1.00 2.00 3.00\n"
                "memory=1024 8192 12.5\n"
                "swap=0 4096 0.0\n"
                "cpu_percent=41.1\n"
                "__SLURM_GPU_TOP_GPUS__\n"
                f"0, GPU-a, NVIDIA A100, On, 00000000:55:00.0, Off, Disabled, 0, N/A, 60, P0, 250, 400, 40000, 81920, {util}, {mem}, Default, 1980\n"
                "__SLURM_GPU_TOP_PROCESSES__\n"
                "1234, python train.py, 39000, GPU-a\n"
                "__SLURM_GPU_TOP_PMON__\n"
                "# gpu pid type sm mem enc dec command\n"
                "0 1234 C 6 2 - - python\n"
                "__SLURM_GPU_TOP_PS__\n"
                "1234 ez275 595.5 1.1 2:13:03 python train.py\n",
            )
        raise AssertionError(command)

    return runner


def test_cli_mock_json_smoke(tmp_path, capsys):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "generated_at": 10,
                "nodes": [
                    {
                        "node": "gpu001",
                        "jobs": [
                            {
                                "job_id": "1",
                                "name": "train",
                                "user": "ez275",
                                "state": "RUNNING",
                                "elapsed": "0:01",
                                "node_count": 1,
                                "nodelist": "gpu001",
                                "gres": "gpu:1",
                                "tres": "gres/gpu=1",
                                "nodes": ["gpu001"],
                            }
                        ],
                        "gpus": [
                            {
                                "node": "gpu001",
                                "index": 0,
                                "uuid": "GPU-a",
                                "name": "NVIDIA A40",
                                "gpu_util_percent": 1,
                                "mem_util_percent": 2,
                                "mem_used_mib": 3,
                                "mem_total_mib": 4,
                                "temperature_c": 5,
                                "power_draw_w": 6.0,
                                "power_limit_w": 7.0,
                                "processes": [],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--mock-json", str(snapshot_path)]) == 0
    assert "gpu001" in capsys.readouterr().out
