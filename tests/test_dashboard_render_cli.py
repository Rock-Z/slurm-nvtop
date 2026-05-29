import json

from slurm_gpu_top.cli import main
from slurm_gpu_top.dashboard import build_snapshot
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
    assert "NVIDIA A100" in rendered
    assert "pid=1234" in rendered


def test_render_snapshot_handles_empty_state():
    rendered = render_snapshot(ClusterSnapshot(generated_at=10), width=80)

    assert "No running GPU-backed Slurm jobs found." in rendered


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
