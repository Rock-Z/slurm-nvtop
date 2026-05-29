from __future__ import annotations

import subprocess
from typing import Protocol, Sequence

from .models import CommandResult


class CommandRunner(Protocol):
    def __call__(self, args: Sequence[str], timeout: float) -> CommandResult:
        ...


def run_command(args: Sequence[str], timeout: float) -> CommandResult:
    argv = tuple(str(arg) for arg in args)
    try:
        completed = subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return CommandResult(argv, 124, stdout, stderr, timed_out=True)
    except OSError as exc:
        return CommandResult(argv, 127, "", str(exc))

    return CommandResult(argv, completed.returncode, completed.stdout, completed.stderr)
