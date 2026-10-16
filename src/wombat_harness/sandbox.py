"""Run generated code in a separate Python process with a timeout.

This is process isolation only (no filesystem/network sandbox). Run the harness
inside a container if you need to treat model output as untrusted.
"""
from __future__ import annotations

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

    def feedback(self, limit: int = 1500) -> str:
        if self.timed_out:
            return "The program timed out (infinite loop or too slow)."
        return (self.stderr or self.stdout)[-limit:]


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


def _s(v: bytes | str | None) -> str:
    return v.decode(errors="replace") if isinstance(v, bytes) else (v or "")
