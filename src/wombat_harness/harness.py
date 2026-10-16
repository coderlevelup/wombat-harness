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
    attempts: list[Attempt] = Field(default_factory=list)


class Harness:
    def __init__(self, client: ChatClient, max_attempts: int = 3):
        self.client, self.max_attempts = client, max_attempts

    def run(self, task: Task) -> HarnessResult:
        user = task.prompt
        if task.tests:
            user += f"\n\nYour code must pass these tests:\n{task.tests}"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        result = HarnessResult(ok=False)

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
