"""Run generated code in a separate Python process with a timeout.

This is process isolation only (no filesystem/network sandbox). Run the harness
inside a container if you need to treat model output as untrusted.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel


class RunResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    returncode: int | None = None

    def feedback(self, limit: int = 1500) -> str:
        if self.timed_out:
            return "The program timed out (infinite loop or too slow)."
        return "\n".join(x for x in (self.stdout, self.stderr) if x.strip())[-limit:]


def run_python(code: str, tests: str | None = None, timeout: float = 10.0) -> RunResult:
    program = code if not tests else f"{code}\n\n# --- tests ---\n{tests}\n"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "main.py"
        path.write_text(program)
        try:
            p = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(ok=False, timed_out=True, stdout=_s(e.stdout), stderr=_s(e.stderr))
    return RunResult(ok=p.returncode == 0, stdout=p.stdout, stderr=p.stderr)


def run_command(
    cmd: str | list[str],
    cwd: str | Path,
    timeout: float = 60.0,
    max_output: int = 20_000,
    env: dict[str, str] | None = None,
) -> RunResult:
    """Run a command (no shell: a string is split with shlex) in `cwd`.

    The command is always supplied by the caller, never chosen by the model.
    It runs in its own process group so a timeout kills any children too.
    Output is truncated to the last `max_output` characters. `env` is added to the
    inherited environment.
    """
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    if not argv:
        return RunResult(ok=False, stderr="empty command")
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
            env={**os.environ, **env} if env else None,
        )
    except OSError as e:
        return RunResult(ok=False, stderr=f"could not start {argv[0]!r}: {e}")
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        timed_out = True
    return RunResult(
        ok=proc.returncode == 0 and not timed_out,
        stdout=out[-max_output:], stderr=err[-max_output:],
        timed_out=timed_out, returncode=proc.returncode,
    )


def _s(v: bytes | str | None) -> str:
    return v.decode(errors="replace") if isinstance(v, bytes) else (v or "")
