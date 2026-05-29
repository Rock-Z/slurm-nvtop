from slurm_gpu_top.models import CommandResult
from slurm_gpu_top.slurm import (
    discover_gpu_jobs,
    expand_nodelist,
    parse_scontrol_key_values,
    parse_squeue_fallback_line,
    parse_squeue_line,
)


def test_parse_squeue_line_with_gpu_fields():
    job = parse_squeue_line("123|train|ez275|RUNNING|1:02:03|2|gpu[001-002]|gres/gpu:2")

    assert job.job_id == "123"
    assert job.name == "train"
    assert job.node_count == 2
    assert job.nodelist == "gpu[001-002]"
    assert job.gpu_hint


def test_parse_squeue_fallback_line():
    job = parse_squeue_fallback_line("124|debug|ez275|RUNNING|0:10|1|node01")

    assert job.job_id == "124"
    assert job.nodelist == "node01"
    assert not job.gpu_hint


def test_parse_scontrol_key_values_preserves_spaces_inside_values():
    fields = parse_scontrol_key_values(
        "JobId=123 JobName=my job UserId=ez275(1000) "
        "AllocTRES=cpu=8,mem=32G,gres/gpu=1 NodeList=gpu001"
    )

    assert fields["JobName"] == "my job"
    assert fields["AllocTRES"] == "cpu=8,mem=32G,gres/gpu=1"
    assert fields["NodeList"] == "gpu001"


def test_expand_nodelist_uses_scontrol_when_available():
    calls = []

    def runner(args, timeout):
        calls.append(tuple(args))
        return CommandResult(tuple(args), 0, "gpu001\ngpu002\n")

    assert expand_nodelist("gpu[001-002]", runner=runner) == ("gpu001", "gpu002")
    assert calls == [("scontrol", "show", "hostnames", "gpu[001-002]")]


def test_expand_nodelist_local_fallback_handles_ranges_and_commas():
    def runner(args, timeout):
        return CommandResult(tuple(args), 1, "", "no scontrol")

    assert expand_nodelist("gpu[001-003,007],other9", runner=runner) == (
        "gpu001",
        "gpu002",
        "gpu003",
        "gpu007",
        "other9",
    )


def test_discover_gpu_jobs_filters_cpu_jobs_and_expands_nodes():
    def runner(args, timeout):
        command = tuple(args)
        if command[:1] == ("squeue",):
            return CommandResult(
                command,
                0,
                "\n".join(
                    [
                        "111|cpu|ez275|RUNNING|0:01|1|cpu001|(null)",
                        "222|gpu|ez275|RUNNING|0:02|2|gpu[001-002]|gres/gpu:1",
                    ]
                ),
            )
        if command[:3] == ("scontrol", "show", "job"):
            jobid = command[-1]
            if jobid == "111":
                return CommandResult(command, 0, "JobId=111 JobName=cpu AllocTRES=cpu=1 NodeList=cpu001")
            return CommandResult(
                command,
                0,
                "JobId=222 JobName=gpu JobState=RUNNING RunTime=0:02 NumNodes=2 "
                "AllocTRES=cpu=8,gres/gpu=2 NodeList=gpu[001-002]",
            )
        if command[:3] == ("scontrol", "show", "hostnames"):
            return CommandResult(command, 0, "gpu001\ngpu002\n")
        raise AssertionError(command)

    jobs = discover_gpu_jobs(user="ez275", runner=runner)

    assert [job.job_id for job in jobs] == ["222"]
    assert jobs[0].nodes == ("gpu001", "gpu002")
    assert "gres/gpu=2" in jobs[0].tres


def test_discover_gpu_jobs_falls_back_when_squeue_format_is_not_supported():
    seen_formats = []

    def runner(args, timeout):
        command = tuple(args)
        if command[:1] == ("squeue",):
            seen_formats.append(command[-1])
            if "%b" in command[-1]:
                return CommandResult(command, 1, "", "invalid format")
            return CommandResult(command, 0, "333|gpu|ez275|RUNNING|0:03|1|gpu009")
        if command[:3] == ("scontrol", "show", "job"):
            return CommandResult(command, 0, "JobId=333 AllocTRES=gres/gpu=1 NodeList=gpu009")
        if command[:3] == ("scontrol", "show", "hostnames"):
            return CommandResult(command, 1, "", "permission denied")
        raise AssertionError(command)

    jobs = discover_gpu_jobs(user="ez275", runner=runner)

    assert seen_formats == ["%A|%j|%u|%T|%M|%D|%R|%b", "%A|%j|%u|%T|%M|%D|%R"]
    assert jobs[0].job_id == "333"
    assert jobs[0].nodes == ("gpu009",)
