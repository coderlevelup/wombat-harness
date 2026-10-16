"""Generate -> validate -> execute -> feed errors back, until it passes or attempts run out."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .client import ChatClient, LLMError
from .models import CodeResponse, Task
from .sandbox import RunResult, run_python

SYSTEM = (
    "You write Python 3 code. Respond only with JSON matching the schema. "
    "The `code` field must be complete, runnable Python with no markdown fences. "
    "Use only the standard library unless told otherwise."
)


class Attempt(BaseModel):
    response: CodeResponse | None = None
    run: RunResult | None = None
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
            if attempt.run.ok:
                result.ok = True
                return result
            messages += [
                {"role": "assistant", "content": attempt.response.model_dump_json()},
                {"role": "user", "content": f"That failed:\n{attempt.run.feedback()}\nFix the code."},
            ]
        return result
