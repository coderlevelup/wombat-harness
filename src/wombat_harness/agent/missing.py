"""Spot a command the model asked for that isn't installed, so a person can fix it instead of the model flailing.

Narrow on purpose. It fires for (1) a shell command that isn't found (exit 127 / "command not found") and
(2) `python -m X` failing with "No module named X". Other import errors are left alone: they may be the
model's own mistake rather than a missing tool.
"""
from __future__ import annotations

import re
import shutil
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class Fix(str, Enum):
    """What the person chooses when a tool is missing (like the old DOS "Retry, Fail?"). Esc still interrupts the turn."""

    RETRY = "retry"  # they fixed it; run the command again
    FAIL = "fail"  # hand the failure back to the model so it can recover another way


class Missing(BaseModel):
    kind: Literal["command", "module"]
    name: str
    command: str  # the command line that failed
    install_hint: str | None = None  # only when we are sure how (python -m X -> pip install X)
    interpreter: str | None = None  # path of the python that will be used, when relevant


_EXIT = re.compile(r"^exit code (\d+)")
_NOT_FOUND = [
    re.compile(r"command not found: (\S+)"),  # zsh
    re.compile(r"(\S+): (?:command )?not found"),  # bash / sh
]
_PY_M = re.compile(r"(?:^|[\s;&|(])(python[\d.]*)\s+-m\s+([\w.]+)")
_NO_MODULE = re.compile(r"No module named '?([\w.]+)'?")


def detect_missing(command: str, output: str) -> Missing | None:
    """`output` is a bash tool result ("exit code N\\n..."). Returns what is missing, or None."""
    code = _EXIT.match(output)
    if not code or code.group(1) == "0":
        return None

    py = _PY_M.search(command)
    if py:
        interp, module = py.group(1), py.group(2).split(".")[0]
        said = _NO_MODULE.search(output)
        if said and said.group(1).split(".")[0] == module:
            return Missing(kind="module", name=module, command=command,
                           install_hint=f"{interp} -m pip install {module}", interpreter=shutil.which(interp))

    if code.group(1) == "127" or any(p.search(output) for p in _NOT_FOUND[:1]) or "command not found" in output:
        for pattern in _NOT_FOUND:
            found = pattern.findall(output)
            if found:
                return Missing(kind="command", name=found[-1].rsplit("/", 1)[-1], command=command)
        if code.group(1) == "127" and command.split():
            return Missing(kind="command", name=command.split()[0].rsplit("/", 1)[-1], command=command)
    elif re.search(r"(?m)^\S*sh: \d+: \S+: not found$", output):  # dash: "/bin/sh: 1: pytest: not found"
        name = _NOT_FOUND[1].findall(output)[-1]
        return Missing(kind="command", name=name.rsplit("/", 1)[-1], command=command)
    return None


def describe(m: Missing, failures: int) -> str:
    """Plain text for the person, including how many times this has failed so far this turn."""
    count = "This has failed once." if failures == 1 else f"This has failed {failures} times."
    if m.kind == "module":
        where = f" for the Python wombat's shell uses ({m.interpreter})" if m.interpreter else ""
        body = f"`{m.name}` is not installed{where}."
        fix = f"\nInstall it, for example: {m.install_hint}" if m.install_hint else ""
    else:
        body = f"`{m.name}` was not found: it isn't installed or isn't on PATH."
        fix = ""
    return f"{body} {count}{fix}"
