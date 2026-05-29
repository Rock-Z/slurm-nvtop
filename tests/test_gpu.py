from slurm_gpu_top.gpu import PROCESS_MARKER, parse_nvidia_smi_output, poll_node_gpus
from slurm_gpu_top.models import CommandResult


def test_parse_nvidia_smi_output_with_processes():
    output = (
        "0, GPU-aaa, NVIDIA A100-SXM4-80GB, 83, 40, 32500, 81920, 62, 241.5, 400.0\n"
        "1, GPU-bbb, NVIDIA A100-SXM4-80GB, 0, 0, 0, 81920, 35, 72.0, 400.0\n"
        f"{PROCESS_MARKER}\n"
        "9100, python, 32000, GPU-aaa\n"
    )

    gpus = parse_nvidia_smi_output("gpu001", output)

    assert len(gpus) == 2
    assert gpus[0].node == "gpu001"
    assert gpus[0].gpu_util_percent == 83
    assert gpus[0].processes[0].pid == 9100
    assert gpus[1].processes == ()


def test_parse_nvidia_smi_output_accepts_na_values():
    output = "0, GPU-aaa, NVIDIA L40S, N/A, [N/A], 0, 46068, N/A, N/A, 350.0\n"

    gpu = parse_nvidia_smi_output("gpu010", output)[0]

    assert gpu.gpu_util_percent is None
    assert gpu.mem_util_percent is None
    assert gpu.temperature_c is None
    assert gpu.power_draw_w is None
    assert gpu.power_limit_w == 350.0


def test_parse_nvidia_smi_output_rejects_malformed_gpu_rows():
    try:
        parse_nvidia_smi_output("gpu001", "0, GPU-aaa\n")
    except ValueError as exc:
        assert "unexpected nvidia-smi GPU row" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_poll_node_gpus_builds_ssh_command_and_parses_stdout():
    calls = []

    def runner(args, timeout):
        calls.append(tuple(args))
        return CommandResult(
            tuple(args),
            0,
            "0, GPU-aaa, NVIDIA H100, 50, 25, 10000, 81559, 45, 200, 700\n",
        )

    gpus, error = poll_node_gpus("gpu007", ssh_options=("BatchMode=yes",), runner=runner)

    assert error is None
    assert gpus[0].name == "NVIDIA H100"
    assert calls[0][0:3] == ("ssh", "-o", "BatchMode=yes")
    assert calls[0][3] == "gpu007"


def test_poll_node_gpus_returns_error_for_timeout():
    def runner(args, timeout):
        return CommandResult(tuple(args), 124, "", "hung", timed_out=True)

    gpus, error = poll_node_gpus("gpu007", runner=runner, timeout=3)

    assert gpus == ()
    assert "timed out after 3s" in error
