from slurm_gpu_top.gpu import (
    PROCESS_MARKER,
    parse_node_probe_output,
    poll_job,
)
from slurm_gpu_top.models import CommandResult


def test_parse_node_probe_output_with_processes():
    output = (
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 83, 40, 32500, 81920, 62, 241.5, 400.0\n"
        "1, GPU-bbb, NVIDIA A100-SXM4-80GB, 0, 0, 0, 81920, 35, 72.0, 400.0\n"
        f"{PROCESS_MARKER}\n"
        "9100, python, 32000, GPU-aaa\n"
    )

    gpus, _host = parse_node_probe_output("gpu001", output)

    assert len(gpus) == 2
    assert gpus[0].node == "gpu001"
    assert gpus[0].gpu_util_percent == 83
    assert gpus[0].processes[0].pid == 9100
    assert gpus[1].processes == ()


def test_parse_node_probe_output_accepts_na_values():
    output = "0, GPU-aaa, NVIDIA L40S, N/A, [N/A], 0, 46068, N/A, N/A, 350.0\n"

    gpu = parse_node_probe_output("gpu010", output)[0][0]

    assert gpu.gpu_util_percent is None
    assert gpu.mem_util_percent is None
    assert gpu.temperature_c is None
    assert gpu.power_draw_w is None
    assert gpu.power_limit_w == 350.0


def test_parse_node_probe_output_stamps_job_and_maps_physical_index():
    # nvidia-smi renumbers GPUs from 0 inside the job cgroup; SLURM_STEP_GPUS=2
    # tells us the real device index, and every row/process belongs to the job.
    output = (
        "__SLURM_GPU_TOP_META__\n"
        "hostname=gpu001.example\n"
        "step_gpus=2\n"
        "driver_version=570.195.03\n"
        "cuda_version=12.8\n"
        "uptime_seconds=1572480\n"
        "load_average=23.42 21.24 20.95\n"
        "memory=113971 1032192 11.0\n"
        "swap=0 8192 0.0\n"
        "cpu_percent=41.1\n"
        "__SLURM_GPU_TOP_GPUS__\n"
        "0, GPU-aaa, NVIDIA H100 80GB HBM3, On, 00000000:55:00.0, Off, Disabled, 0, N/A, 36, P0, 138, 700, 9723, 81559, 10, 2, Default, 1980\n"
        "__SLURM_GPU_TOP_PROCESSES__\n"
        "514876, python train.py, 9710, GPU-aaa\n"
        "__SLURM_GPU_TOP_PMON__\n"
        "# gpu pid type sm mem enc dec command\n"
        "0 514876 C 6 2 - - python\n"
        "__SLURM_GPU_TOP_PS__\n"
        "514876 ez275 595.5 1.1 2:13:03 /path/train.py\n"
    )

    gpus, host = parse_node_probe_output("gpu001", output, slurm_job_id="101")

    assert host.hostname == "gpu001.example"
    assert host.driver_version == "570.195.03"
    assert host.cuda_version == "12.8"
    assert host.cpu_percent == 41.1
    assert host.load_average == (23.42, 21.24, 20.95)
    gpu = gpus[0]
    assert gpu.index == 2  # mapped from cgroup-relative 0 via SLURM_STEP_GPUS
    assert gpu.slurm_job_id == "101"
    assert gpu.persistence_mode == "On"
    assert gpu.pci_bus_id == "00000000:55:00.0"
    assert gpu.mig_mode == "Disabled"
    assert gpu.performance_state == "P0"
    assert gpu.compute_mode == "Default"
    assert gpu.sm_clock_mhz == 1980
    proc = gpu.processes[0]
    assert proc.user == "ez275"
    assert proc.cpu_percent == 595.5
    assert proc.sm_util_percent == 6
    assert proc.mem_bw_util_percent == 2
    assert proc.command == "/path/train.py"
    assert proc.slurm_job_id == "101"


def test_parse_node_probe_output_rejects_malformed_gpu_rows():
    try:
        parse_node_probe_output("gpu001", "0, GPU-aaa\n")
    except ValueError as exc:
        assert "unexpected nvidia-smi GPU row" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_poll_job_builds_srun_overlap_command_and_parses_stdout():
    calls = []

    def runner(args, timeout):
        calls.append(tuple(args))
        return CommandResult(
            tuple(args),
            0,
            "0, GPU-aaa, NVIDIA H100, 50, 25, 10000, 81559, 45, 200, 700\n",
        )

    gpus, _host, error = poll_job("1984799", "gpu007", runner=runner)

    assert error is None
    assert gpus[0].name == "NVIDIA H100"
    assert gpus[0].slurm_job_id == "1984799"
    command = calls[0]
    assert command[0] == "srun"
    assert "--overlap" in command
    assert command[command.index("--jobid") + 1] == "1984799"
    assert command[command.index("--nodelist") + 1] == "gpu007"


def test_poll_job_returns_error_for_timeout():
    def runner(args, timeout):
        return CommandResult(tuple(args), 124, "", "hung", timed_out=True)

    gpus, _host, error = poll_job("1984799", "gpu007", runner=runner, timeout=3)

    assert gpus == ()
    assert "timed out after 3s" in error
