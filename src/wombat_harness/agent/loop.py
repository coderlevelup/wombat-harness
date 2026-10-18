"""The agent loop: model picks a tool, we (maybe ask, then) run it, feed the result back."""
from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel

from ..client import ChatClient, LLMError, LLMRequestError
from .missing import Fix, Missing, describe, detect_missing
from .tools import Bash, Finish, Step, ToolError, clip, relativize
from .workspace import Workspace

SYSTEM = """\
You are wombat, a coding agent working in the project directory described below. \
You complete the user's task by calling ONE tool per step.

Reply ONLY with JSON: {"thought": "...", "action": {"tool": "<name>", ...arguments}}. Tools:
- list_dir(path): list a directory
- read_file(path): read a file
- grep(pattern, path): regex search in files under path
- write_file(path, content): create or overwrite a file with its COMPLETE content
- edit_file(path, old, new): replace exactly one occurrence of `old` with `new`
- bash(command): run a shell command in the project directory
- finish(message): you are done; `message` is your answer to the user

Rules: read files before editing them. Prefer edit_file for small changes. Fix the code, not the tests: don't change a test to make it pass unless the user asks. Do not try to install packages: if a command isn't installed, the user is asked to fix it. Verify your work \
(e.g. `python -m pytest -q` via bash; never run a .py test file with bash directly). Paths are relative to the project directory. When the task is \
done, or the user only asked a question, call finish.

Project directory: <ROOT>
Top level:
<LISTING>"""

LOOP_AFTER = 3  # the same call returning the same result this many times in one turn means the model is stuck
REPEAT_NOTE = "\n(You have made this exact call 3 times with the same result. Try something different, or call finish.)"
def fail_note(m: Missing) -> str:
    """What the model is told when the person presses Fail (= "I'm not retrying").

    It asks for the install steps rather than another workaround: the model can propose running the install with bash
    (the person is asked to approve) or finish with the steps so the person can run them and ask it to retry.
    """
    how = f" (`{m.install_hint}`)" if m.install_hint else ""
    return (f"\n(The user did not retry. `{m.name}` is not available. Tell the user exactly what to install{how}, then either "
            "propose running it with bash (they will be asked to approve) or call finish with those steps so they can run them "
            "and ask you to retry. Don't try other approaches unless they ask.)")


DENIED_NOTE = "The user denied this action. The turn has ended; wait for their next message."
OBSERVATION = "Tool result ({tool}):\n{output}"
ELIDED = "[earlier tool output elided to save context]"


class Policy(str, Enum):
    ASK = "ask"  # confirm every write, edit and command
    EDITS = "edits"  # file edits are automatic; commands still ask
    YOLO = "yolo"  # nothing asks


class Event(BaseModel):
    kind: Literal["thought", "tool", "result", "denied", "blocked", "reply", "error", "stopped"]
    text: str
    detail: str = ""
    ok: bool = True


def needs_approval(policy: Policy, call) -> bool:
    """The one approval rule every engine shares."""
    if policy is Policy.YOLO:
        return False
    if policy is Policy.EDITS:
        return isinstance(call, Bash)
    return call.needs_approval


Emit = Callable[[Event], None]
Approve = Callable[[str, str], bool]  # (summary, preview) -> allowed?
Resolve = Callable[[Missing, int], Fix]  # (what is missing, failures so far this turn) -> the person's choice


class AgentPort(Protocol):
    """What a UI needs from an agent. Nothing here knows about Textual."""

    policy: Policy
    workspace: Workspace

    def run_turn(self, text: str, on_event: Emit, approve: Approve,
                 should_stop: Callable[[], bool] = ..., resolve: Resolve | None = None) -> str: ...

    def reset(self) -> None: ...


class Agent:
    def __init__(
        self,
        client: ChatClient,
        workspace: Workspace,
        policy: Policy = Policy.ASK,
        max_steps: int = 25,
        context_chars: int = 16_000,
    ):
        self.client, self.workspace, self.policy = client, workspace, policy
        self.max_steps, self.context_chars = max_steps, context_chars
        self.reset()

    def reset(self) -> None:
        ws = self.workspace
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in ws.root.iterdir() if c.name != ".git")
        listing = "\n".join(names[:40]) + (f"\n... [{len(names) - 40} more]" if len(names) > 40 else "")
        self._turns: list[int] = []  # index of each user-typed message in self.messages
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM.replace("<ROOT>", str(ws.root)).replace("<LISTING>", listing or "(empty)")}
        ]

    def _needs_approval(self, call) -> bool:
        return needs_approval(self.policy, call)

    def run_turn(self, text: str, on_event: Emit, approve: Approve,
                 should_stop: Callable[[], bool] = ..., resolve: Resolve | None = None) -> str: ...

    def reset(self) -> None: ...


class Agent:
    def __init__(
        self,
        client: ChatClient,
        workspace: Workspace,
        policy: Policy = Policy.ASK,
        max_steps: int = 25,
        context_chars: int = 16_000,
    ):
        self.client, self.workspace, self.policy = client, workspace, policy
        self.max_steps, self.context_chars = max_steps, context_chars
        self.reset()

    def reset(self) -> None:
        ws = self.workspace
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in ws.root.iterdir() if c.name != ".git")
        listing = "\n".join(names[:40]) + (f"\n... [{len(names) - 40} more]" if len(names) > 40 else "")
        self._turns: list[int] = []  # index of each user-typed message in self.messages
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM.replace("<ROOT>", str(ws.root)).replace("<LISTING>", listing or "(empty)")}
        ]

    def _needs_approval(self, call) -> bool:
        if self.policy is Policy.YOLO:
            return False
        if self.policy is Policy.EDITS:
            return isinstance(call, Bash)
        return call.needs_approval

    def run_turn(self, text: str, on_event: Emit, approve: Approve,
                 should_stop: Callable[[], bool] | None = None, resolve: Resolve | None = None) -> str:
        """Run until the model calls finish, a limit trips, or `should_stop()`. Returns the reply.

        `resolve`, if given, is asked what to do when a shell command fails because a tool isn't installed
        (Retry / Fail); without it the failure just goes to the model.
        """
        stop = should_stop or (lambda: False)
        self._turns.append(len(self.messages))
        self.messages.append({"role": "user", "content": text})
        invalid = repeats = 0
        seen: Counter[tuple[str, str]] = Counter()  # (call, result) -> times seen this turn
        failures: Counter[str] = Counter()  # missing tool -> times it has failed this turn

        for _ in range(self.max_steps):
            if stop():
                on_event(Event(kind="stopped", text="stopped"))
                return ""
            self._trim()
            try:
                step = self.client.complete(self.messages, Step)
            except LLMRequestError as e:  # no point retrying a server that is down or lacks the model
                on_event(Event(kind="error", text=str(e), ok=False))
                return ""
            except LLMError as e:
                invalid += 1
                on_event(Event(kind="error", text="model reply was invalid", detail=str(e), ok=False))
                if invalid >= 3:
                    return ""
                self.messages.append({"role": "user", "content": f"Your reply was invalid: {clip(str(e), 400)}\nReply with the JSON again."})
                continue
            invalid = 0
            self.messages.append({"role": "assistant", "content": step.model_dump_json(exclude_defaults=True)})
            if step.thought.strip():
                on_event(Event(kind="thought", text=step.thought.strip()))

            call = relativize(step.action, self.workspace)
            if isinstance(call, Finish):
                on_event(Event(kind="reply", text=call.message))
                return call.message

            summary = call.summary()
            # An action that is certain to fail (edit text not found / ambiguous, path outside the workspace) is not
            # put to the person: it just runs, fails, and the model gets the error like any other tool failure.
            if self._needs_approval(call) and call.preflight(self.workspace) is None:
                if not approve(summary, call.preview(self.workspace) or ""):
                    # A denial hands control back to the person. The model can't ask a question mid-turn (only
                    # `finish` ends one), so letting it carry on just makes it guess its next action; the
                    # person's next message steers it instead, with the denial still in the history.
                    self._observe(call.tool, DENIED_NOTE)
                    on_event(Event(kind="denied", text=summary, ok=False))
                    return ""
            on_event(Event(kind="tool", text=summary))
            try:
                output, ok = call.run(self.workspace), True
            except ToolError as e:
                output, ok = f"error: {e}", False
            while ok and resolve and isinstance(call, Bash) and (missing := detect_missing(call.command, output)):
                failures[missing.name] += 1
                on_event(Event(kind="blocked", text=f"{missing.name} is missing",
                               detail=describe(missing, failures[missing.name]), ok=False))
                choice = resolve(missing, failures[missing.name])
                if choice is Fix.RETRY:  # the person fixed it: run it again, and the model never sees the failure
                    output = call.run(self.workspace)
                    continue
                output += fail_note(missing)  # Fail: let the model see the failure and recover another way
                break
            on_event(Event(kind="result", text=summary, detail=output, ok=ok))
            # Stuck detection: the same call getting the same result again. Looking at results, not just calls, catches
            # cycles (read, failing edit, read, failing edit ...) that a "same call back to back" check misses, and
            # doesn't fire on a legitimate re-read after the file changed, because the result differs.
            seen[(call.model_dump_json(), output)] += 1
            looping = seen[(call.model_dump_json(), output)] >= LOOP_AFTER
            self._observe(call.tool, output + (REPEAT_NOTE if looping else ""))
            if looping:
                repeats += 1
                on_event(Event(kind="error", text="model is repeating itself", ok=False))
                if repeats >= 2:
                    return ""

        on_event(Event(kind="error", text=f"stopped after {self.max_steps} steps without finishing", ok=False))
        return ""

    def _observe(self, tool: str, output: str) -> None:
        self.messages.append({"role": "user", "content": OBSERVATION.format(tool=tool, output=clip(output))})

    def _trim(self) -> None:
        """Keep the prompt under budget: elide old tool output first, then drop old turns."""
        def size() -> int:
            return sum(len(m["content"]) for m in self.messages)

        for m in self.messages[1:-1]:
            if size() <= self.context_chars:
                return
            if m["role"] == "user" and m["content"].startswith("Tool result") and m["content"] != ELIDED:
                m["content"] = ELIDED
        while size() > self.context_chars and len(self._turns) > 1:  # drop whole old turns
            cut = self._turns[1]
            del self.messages[1:cut]
            self._turns = [t - (cut - 1) for t in self._turns[1:]]
        start = self._turns[-1]  # last resort: oldest step pairs of the current turn
        while size() > self.context_chars and len(self.messages) > start + 3:
            del self.messages[start + 1 : start + 3]  # the task itself (start) always stays
