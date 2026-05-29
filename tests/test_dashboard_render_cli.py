import json

import slurm_gpu_top.cli as cli_module
from slurm_gpu_top.cli import main
from slurm_gpu_top.dashboard import build_snapshot
from slurm_gpu_top.history import UtilizationHistory
from slurm_gpu_top.models import (
    ClusterSnapshot,
    CommandResult,
    GPUDevice,
    GPUProcess,
    NodeSnapshot,
    SlurmJob,
    SnapshotBuilderConfig,
)
from slurm_gpu_top.render import _format_process_table_row, _process_table_layout, render_snapshot


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
    assert "SGTOP 0.1.0" in rendered
    assert "Driver Version" in rendered
    assert "Memory-Usage" in rendered
    assert "GPU-Util" in rendered
    assert "Bus-Id" not in rendered
    assert "MEM:" in rendered
    assert "UTL:" in rendered
    assert "CPU 41%" in rendered
    assert "MEM 12%" in rendered
    assert "Processes" in rendered
    assert "python train.py" in rendered
    assert rendered.count("GPU  Name        Persistence-M") == 1


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
    assert "SGTOP 0.1.0" in rendered
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

    assert "SGTOP 0.1.0" in rendered
    assert "MEM:" in rendered
    assert "UTL:" in rendered
    assert "A100" in rendered


def test_render_snapshot_uses_one_gpu_table_grouped_by_node():
    first = _single_gpu_snapshot(util=75, mem=50)
    second_node = NodeSnapshot(
        node="gpu002",
        jobs=first.nodes[0].jobs,
        gpus=first.nodes[0].gpus,
        host=first.nodes[0].host,
    )
    snapshot = ClusterSnapshot(nodes=(first.nodes[0], second_node), generated_at=10)

    rendered = render_snapshot(snapshot, width=120, color=False, unicode=True)

    assert rendered.count("SGTOP 0.1.0") == 1
    assert rendered.count("SGTOP 0.1.0") == 1
    assert "[gpu001]" in rendered
    assert "[gpu002]" in rendered
    assert rendered.index("[gpu001]") < rendered.index("GPU  Name")
    assert rendered.index("[gpu002]") > rendered.index("UTL:")


def test_render_snapshot_omits_repeated_gpu_label_block_and_draws_node_history():
    first = _single_gpu_snapshot(util=75, mem=50)
    gpu = first.nodes[0].gpus[0]
    second_gpu = GPUDevice(
        node="gpu002",
        index=0,
        uuid="GPU-b",
        name=gpu.name,
        gpu_util_percent=25,
        mem_util_percent=20,
        mem_used_mib=16000,
        mem_total_mib=gpu.mem_total_mib,
        temperature_c=gpu.temperature_c,
        power_draw_w=gpu.power_draw_w,
        power_limit_w=gpu.power_limit_w,
        persistence_mode=gpu.persistence_mode,
        pci_bus_id="00000000:E6:00.0",
        display_active=gpu.display_active,
        mig_mode=gpu.mig_mode,
        ecc_errors=gpu.ecc_errors,
        fan_speed_percent=gpu.fan_speed_percent,
        performance_state=gpu.performance_state,
        compute_mode=gpu.compute_mode,
        sm_clock_mhz=gpu.sm_clock_mhz,
        processes=gpu.processes,
    )
    snapshot = ClusterSnapshot(
        nodes=(
            first.nodes[0],
            NodeSnapshot(node="gpu002", jobs=first.nodes[0].jobs, gpus=(second_gpu,), host=first.nodes[0].host),
        ),
        generated_at=10,
    )

    rendered = render_snapshot(
        snapshot,
        width=120,
        color=False,
        unicode=True,
        node_util_histories={
            "gpu001": (10, 30, 60, 90, 80, 70, 75, 75),
            "gpu002": (90, 70, 50, 30, 20, 25, 25, 25),
        },
        node_mem_histories={
            "gpu001": (5, 10, 15, 20, 25, 30, 35, 40),
            "gpu002": (40, 35, 30, 25, 20, 20, 20, 20),
        },
    )

    assert rendered.count("GPU  Name        Persistence-M") == 1
    assert rendered.count("GPU MEM") == 1
    assert "╴30s├" in rendered
    assert "now" in rendered
    assert "gpu001" in rendered
    assert "gpu002" in rendered
    assert "Bus-Id" not in rendered
    assert rendered.index("GPU MEM") < rendered.index("Processes:")
    lines = rendered.splitlines()
    second_node_line = next(idx for idx, line in enumerate(lines) if "[gpu002]" in line)
    assert lines[second_node_line + 1].startswith("├")
    assert lines[second_node_line - 1].startswith("╞")
    assert "╧" in lines[second_node_line - 1]
    history_node_line = next(idx for idx, line in enumerate(lines) if "gpu001" in line and "gpu002" in line)
    history_line = lines[history_node_line - 1]
    assert "╧" in history_line
    assert "╤" in history_line


def test_render_snapshot_respects_terminal_height():
    snapshot = _single_gpu_snapshot(util=75, mem=50)

    rendered = render_snapshot(snapshot, width=120, height=6, color=False, unicode=True)

    assert len(rendered.splitlines()) == 6
    assert "lines hidden" in rendered


def test_render_snapshot_drops_process_table_before_gpu_truncation():
    snapshot = _single_gpu_snapshot(util=75, mem=50)

    rendered = render_snapshot(snapshot, width=120, height=13, color=False, unicode=True)

    assert "UTL:" in rendered
    assert "Processes:" not in rendered
    assert "lines hidden" not in rendered


def test_render_snapshot_keeps_history_before_processes_when_height_is_tight():
    snapshot = _single_gpu_snapshot(util=75, mem=50)

    rendered = render_snapshot(
        snapshot,
        width=120,
        height=26,
        color=False,
        unicode=True,
        node_util_histories={"gpu001": (10, 30, 60, 90, 80, 70, 75, 75)},
        node_mem_histories={"gpu001": (5, 10, 15, 20, 25, 30, 35, 40)},
    )

    assert "GPU MEM" in rendered
    assert "Processes:" not in rendered
    assert "lines hidden" not in rendered


def test_cluster_process_columns_align_dynamically_and_clip():
    rows = [
        {
            "node": "n1",
            "gpu": "0",
            "pid": "7",
            "type": "C",
            "user": "ez",
            "gpu_mem": "1MiB",
            "sm": "6",
            "gmbw": "2",
            "cpu": "5.5",
            "mem": "1.1",
            "time": "0:01",
            "command": "python short.py",
        },
        {
            "node": "very-long-node-name-that-must-clip",
            "gpu": "10",
            "pid": "1234567",
            "type": "C",
            "user": "longusername",
            "gpu_mem": "12345MiB",
            "sm": "100",
            "gmbw": "99",
            "cpu": "1234.5",
            "mem": "12.3",
            "time": "12:34:56",
            "command": "/very/long/command/that/should/be/clipped/at/the/right/edge",
        },
    ]

    columns, command_width = _process_table_layout(rows, inner=86)
    rendered_rows = [_format_process_table_row(row, columns, command_width) for row in rows]
    pid_start = next_start = 0
    for key, _label, _align, width in columns:
        if key == "pid":
            break
        next_start += width + 1
    pid_start = next_start
    pid_width = next(width for key, _label, _align, width in columns if key == "pid")

    assert all(len(row) <= 86 for row in rendered_rows)
    assert rendered_rows[0][pid_start : pid_start + pid_width].endswith("7")
    assert rendered_rows[1][pid_start : pid_start + pid_width].endswith("1234567")


def test_render_snapshot_handles_empty_state():
    rendered = render_snapshot(ClusterSnapshot(generated_at=10), width=80)

    assert "No running GPU-backed Slurm jobs found." in rendered


def test_rendered_process_lines_clip_to_terminal_width():
    job = SlurmJob(
        job_id="101",
        name="train",
        user="ez275",
        state="RUNNING",
        elapsed="1:00",
        node_count=1,
        nodelist="very-long-node-name-that-must-clip",
        tres="gres/gpu=1",
        nodes=("very-long-node-name-that-must-clip",),
    )
    proc = GPUProcess(
        pid=123456789,
        name="python",
        used_memory_mib=98765,
        gpu_uuid="GPU-a",
        user="verylonguser",
        cpu_percent=1234.5,
        mem_percent=12.3,
        elapsed="12:34:56",
        command="/very/long/path/to/python train.py --with-many-arguments-that-must-not-overflow",
        sm_util_percent=100,
        mem_bw_util_percent=99,
    )
    gpu = GPUDevice(
        node="very-long-node-name-that-must-clip",
        index=12,
        uuid="GPU-a",
        name="NVIDIA A100",
        gpu_util_percent=99,
        mem_util_percent=50,
        mem_used_mib=98765,
        mem_total_mib=131072,
        temperature_c=60,
        power_draw_w=250.0,
        power_limit_w=400.0,
        processes=(proc,),
    )
    snapshot = ClusterSnapshot(
        nodes=(NodeSnapshot(node="very-long-node-name-that-must-clip", jobs=(job,), gpus=(gpu,)),),
        generated_at=10,
    )

    rendered = render_snapshot(snapshot, width=86, color=False, unicode=True)

    assert "Processes:" in rendered
    assert all(len(line) <= 86 for line in rendered.splitlines())


def test_live_sleep_returns_true_on_q(monkeypatch):
    class FakeInput:
        def read(self, _size):
            return "q"

    monkeypatch.setattr(cli_module.sys, "stdin", FakeInput())
    monkeypatch.setattr(cli_module.select, "select", lambda read, _write, _error, _timeout: (read, [], []))

    assert cli_module._sleep_or_quit(10, keyboard_enabled=True) is True


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


def test_cli_live_mode_refreshes_one_screen(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_module,
        "build_snapshot",
        lambda config: ClusterSnapshot(generated_at=10),
    )

    def stop_after_first_sleep(_interval):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module.time, "sleep", stop_after_first_sleep)

    assert main(["--interval", "0.1", "--color", "never"]) == 130
    assert capsys.readouterr().out.startswith("\033[H\033[J")
