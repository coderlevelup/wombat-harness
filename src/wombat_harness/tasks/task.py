from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from ..agent import Policy
from ..agent.loop import AgentPort, Event


class Task(BaseModel):
    name: str
    prompt: str
    files: dict[str, str]                                   # starting project: path -> content
    protected: list[str] = Field(default_factory=list)      # files that must stay byte-for-byte as given
    hidden: str                                             # python run in the project; exit 0 means done
    solution: dict[str, str] = Field(default_factory=dict)  # reference edit, used to test the task itself

    def write(self, root: Path, files: dict[str, str] | None = None) -> None:
        for rel, text in (self.files if files is None else files).items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


class Verdict(BaseModel):
    tests_pass: bool        # the visible `pytest` passes when we run it
    hidden_pass: bool       # checks the model never saw
    protected_intact: bool  # protected files are unmodified

    @property
    def done(self) -> bool:
        return self.tests_pass and self.hidden_pass and self.protected_intact


def _py(root: Path, *args: str) -> bool:
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, *args], cwd=root, capture_output=True, env=env).returncode == 0


def check_task(task: Task, root: Path) -> Verdict:
    return Verdict(
        tests_pass=_py(root, "-m", "pytest", "-q", "-p", "no:cacheprovider"),
        hidden_pass=_py(root, "-c", task.hidden),
        protected_intact=all((root / p).is_file() and (root / p).read_text() == task.files[p] for p in task.protected),
    )


class Result(BaseModel):
    task: str
    verdict: Verdict
    done: bool
    replied: bool
    secs: float
    tools: list[str]
    errors: list[str]
    crash: str | None = None


def run_task(task: Task, make_agent: Callable[[Path], AgentPort], *, root: Path | None = None) -> Result:
    """Run one task with the agent that `make_agent(project_dir)` builds, in yolo mode, and judge the result independently."""
    with tempfile.TemporaryDirectory() as tmp:
        project = root or Path(tmp)
        task.write(project)
        events: list[Event] = []
        agent = make_agent(project)
        t0 = time.time()
        try:
            reply, crash = agent.run_turn(task.prompt, events.append, lambda s, p: True), None
        except Exception as e:  # an engine crash is a result too
            reply, crash = "", f"{type(e).__name__}: {e}"
        verdict = check_task(task, project)
        return Result(
            task=task.name, verdict=verdict, done=verdict.done and bool(reply) and not crash, replied=bool(reply),
            secs=round(time.time() - t0, 1), tools=[e.text for e in events if e.kind == "tool"],
            errors=[e.text for e in events if e.kind == "error"], crash=crash)
