"""Generate -> validate -> execute -> feed errors back, until it passes or attempts run out."""
from __future__ import annotations

import tempfile

from pathlib import Path

from pydantic import BaseModel, Field

from .client import ChatClient, LLMError
from .models import CodeResponse, Task
from .sandbox import RunResult, run_command, run_python

SYSTEM = (
    "You write Python 3 code. Respond only with JSON matching the schema. "
    "The `code` field must be complete, runnable Python with no markdown fences. "
    "Use only the standard library unless told otherwise."
)


class Attempt(BaseModel):
    response: CodeResponse | None = None
    run: RunResult | None = None
    check: RunResult | None = None  # result of Task.check, if it was reached
    error: str | None = None  # LLM/validation error, when no code was produced


class HarnessResult(BaseModel):
    ok: bool
    code: str | None = None
    tests: str | None = None
    attempts: list[Attempt] = Field(default_factory=list)

    def save(self, path: str | Path, *, force: bool = False, write_tests: bool = False) -> list[Path]:
        """Write the code (and optionally `test_<name>.py` beside it); returns the paths written.

        Only successful runs can be saved. Every target is checked before anything is written.
        """
        if not self.ok or self.code is None:
            raise ValueError("run did not succeed; refusing to save its code")
        if write_tests and not self.tests:
            raise ValueError("no tests to write")
        out = Path(path).expanduser()
        targets = [(out, self.code)]
        if write_tests:
            targets.append((out.with_name(f"test_{out.name}"), self.tests))
        for t, _ in targets:
            _check_target(t, force)
        for t, text in targets:
            t.write_text(text.rstrip("\n") + "\n")
        return [t for t, _ in targets]


class Harness:
    def __init__(self, client: ChatClient, max_attempts: int = 3):
        self.client, self.max_attempts = client, max_attempts

    def run(self, task: Task) -> HarnessResult:
        user = task.prompt
        if task.tests:
            user += f"\n\nYour code must pass these tests:\n{task.tests}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        result = HarnessResult(ok=False, tests=task.tests)

        for _ in range(self.max_attempts):
            attempt = Attempt()
            result.attempts.append(attempt)
            try:
                attempt.response = self.client.complete(messages, CodeResponse)
            except LLMError as e:
                attempt.error = str(e)
                messages.append({"role": "user", "content": f"Your reply was invalid: {e}\nTry again."})
                continue

            code = attempt.response.code
            attempt.run = run_python(code, task.tests, task.timeout)
            result.code = code
            failure = None
            if attempt.run.ok and task.check:
                attempt.check = _run_check(task, code)
                if not attempt.check.ok:
                    failure = f"The check command `{task.check}` failed:\n{attempt.check.feedback()}"
            elif not attempt.run.ok:
                failure = f"That failed:\n{attempt.run.feedback()}"
            if failure is None:
                result.ok = True
                return result
            messages += [
                {"role": "assistant", "content": attempt.response.model_dump_json()},
                {"role": "user", "content": f"{failure}\nFix the code."},
            ]
        return result


def _run_check(task: Task, code: str) -> RunResult:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "solution.py").write_text(code)
        for name, content in task.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return run_command(task.check, root, task.check_timeout)

def _check_target(path: Path, force: bool) -> None:
    if path.is_symlink():
        raise ValueError(f"{path} is a symlink; refusing to write through it")
    if path.is_dir():
        raise ValueError(f"{path} is a directory; give a file path")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"directory {path.parent} does not exist")
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists")
