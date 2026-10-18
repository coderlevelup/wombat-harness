"""The agent as an idiomatic pydantic-ai `Agent`: native function tools + deferred approvals.

Reused unchanged from the builtin engine: `Workspace`, the tool implementations (each tool
function builds the same model and calls its `run`/`preview`), the approval rule, `Event`s.
What pydantic-ai replaces: the loop, the model client, retries/limits, message history, and
the approve-then-resume protocol (`DeferredToolRequests` / `DeferredToolResults`).

Requires a model with native tool calling (Ollama refuses tools for e.g. Gemma 3; use the
`pai-native` engine for those).
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable

from pydantic_ai import (
    Agent, ApprovalRequired, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied, UsageLimits,
)
from pydantic_ai.exceptions import (
    ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded,
)
from pydantic_ai.messages import ModelRequest, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import Model

from ..agent.loop import (
    DENIED_NOTE, FAIL_NOTE, LOOP_AFTER, REPEAT_NOTE, Approve, Emit, Event, Policy, Resolve, needs_approval,
)
from ..agent.missing import Fix, describe, detect_missing
from ..agent.tools import Bash, EditFile, Grep, ListDir, ReadFile, ToolError, WriteFile, clip
from ..agent.workspace import Workspace
from .client import describe_model_error

INSTRUCTIONS = """\
You are wombat, a coding agent working in the project directory described below. Complete the \
user's task by calling tools. Read files before editing them. Prefer edit_file for small changes. \
Do not try to install packages: if a command isn't installed, the user is asked to fix it. Verify your work (e.g. `python -m pytest -q` via bash; never run a .py test file with bash \
directly). Paths are relative to the project directory. When the task is done, or the user only \
asked a question, reply with a short plain-text answer and no tool call.

Project directory: <ROOT>
Top level:
<LISTING>"""

NOT_RUN_NOTE = "Not run: the user denied an earlier action in this batch, so the turn ended."
TOOLS = {c.model_fields["tool"].annotation.__args__[0]: c
         for c in (ListDir, ReadFile, Grep, WriteFile, EditFile, Bash)}


class _Abort(Exception):
    """Raised inside a tool to end the whole turn (e.g. the model keeps repeating itself)."""


class PydanticAIAgent:
    def __init__(
        self,
        model: Model | str,
        workspace: Workspace,
        policy: Policy = Policy.ASK,
        max_steps: int = 25,
    ):
        self.workspace, self.policy, self.max_steps = workspace, policy, max_steps
        self._emit: Emit = lambda e: None
        self._seen: Counter[tuple[str, str]] = Counter()
        self._repeats = 0
        self._resolve: Resolve | None = None
        self._failures: Counter[str] = Counter()
        self._loop = asyncio.new_event_loop()  # one loop for the engine's lifetime: the HTTP client is reused
        self._model_name = getattr(model, "model_name", str(model))
        self._provider = "anthropic" if getattr(model, "system", "") == "anthropic" else "ollama"
        self._base_url = getattr(getattr(model, "client", None), "base_url", "the model server")
        self._agent = self._build(model)
        self.reset()

    # -- AgentPort ------------------------------------------------------------------------
    def reset(self) -> None:
        self._history: list = []

    def run_turn(self, text: str, on_event: Emit, approve: Approve,
                 should_stop: Callable[[], bool] | None = None, resolve: Resolve | None = None) -> str:
        self._emit, self._seen, self._repeats = on_event, Counter(), 0
        self._resolve, self._failures = resolve, Counter()
        stop = should_stop or (lambda: False)
        try:
            return self._loop.run_until_complete(self._turn(text, approve, stop))
        except _Abort as e:
            on_event(Event(kind="error", text=str(e), ok=False))
        except UsageLimitExceeded:
            on_event(Event(kind="error", ok=False,
                           text=f"stopped after {self.max_steps} steps without finishing"))
        except (ModelHTTPError, ModelAPIError) as e:
            on_event(Event(kind="error", ok=False,
                           text=describe_model_error(e, self._model_name, str(self._base_url), self._provider)))
        except UnexpectedModelBehavior as e:
            on_event(Event(kind="error", text="model reply was invalid", detail=str(e), ok=False))
        return ""

    # -- the turn -------------------------------------------------------------------------
    async def _turn(self, text: str, approve: Approve, stop: Callable[[], bool]) -> str:
        prompt: str | None = text
        deferred: DeferredToolResults | None = None
        history, usage = list(self._history), None
        limits = UsageLimits(request_limit=self.max_steps)
        while True:
            async with self._agent.iter(
                prompt, message_history=history, deferred_tool_results=deferred,
                usage_limits=limits, usage=usage,
            ) as run:
                async for node in run:
                    if stop():
                        self._emit(Event(kind="stopped", text="stopped"))
                        return ""
                    if Agent.is_call_tools_node(node):
                        parts = node.model_response.parts
                        if any(isinstance(p, ToolCallPart) for p in parts):
                            for p in parts:
                                if isinstance(p, TextPart) and p.content.strip():
                                    self._emit(Event(kind="thought", text=p.content.strip()))
                result = run.result
            history, usage = result.all_messages(), result.usage
            if isinstance(result.output, DeferredToolRequests):
                approvals, denied_id = self._decide(result.output, approve)
                if denied_id:  # hand control back; see Agent.run_turn for why
                    self._history = self._close_pending(history, result.output, denied_id)
                    return ""
                deferred = DeferredToolResults(approvals=approvals)
                prompt = None
                if stop():
                    self._emit(Event(kind="stopped", text="stopped"))
                    return ""
                continue
            self._history = history
            self._emit(Event(kind="reply", text=result.output))
            return result.output

    def _decide(self, requests: DeferredToolRequests, approve: Approve) -> tuple[dict[str, bool], str | None]:
        """Ask for each pending approval in order, stopping at the first denial.

        Returns (approvals so far, tool_call_id of the denied call or None).
        """
        approvals: dict[str, bool] = {}
        for part in requests.approvals:
            call = TOOLS[part.tool_name](tool=part.tool_name, **part.args_as_dict())
            if not approve(call.summary(), call.preview(self.workspace) or ""):
                self._emit(Event(kind="denied", text=call.summary(), ok=False))
                return approvals, part.tool_call_id
            approvals[part.tool_call_id] = True
        return approvals, None

    def _close_pending(self, history: list, requests: DeferredToolRequests, denied_id: str) -> list:
        """End the turn at a denial without leaving a tool call unanswered (providers reject that).

        Every pending call in the batch gets a result: the denied one says so, the rest (including any approved
        before the denial) say "not run", because the batch is cut off. The results go into the trailing request
        if there is one, so the person's next message merges into the same request.
        """
        closing = [
            ToolReturnPart(tool_name=p.tool_name, tool_call_id=p.tool_call_id,
                           content=DENIED_NOTE if p.tool_call_id == denied_id else NOT_RUN_NOTE)
            for p in requests.approvals
        ]
        if history and isinstance(history[-1], ModelRequest):
            return [*history[:-1], ModelRequest(parts=[*history[-1].parts, *closing])]
        return [*history, ModelRequest(parts=closing)]

    # -- tools ----------------------------------------------------------------------------
    def _run(self, ctx: RunContext, call) -> str:
        if needs_approval(self.policy, call) and not ctx.tool_call_approved and call.preflight(self.workspace) is None:
            raise ApprovalRequired  # (a call certain to fail just runs and returns the error: no point asking)
        summary = call.summary()
        self._emit(Event(kind="tool", text=summary))
        try:
            output, ok = call.run(self.workspace), True
        except ToolError as e:
            output, ok = f"error: {e}", False
        # A shell command failed because a tool isn't installed: ask the person (Retry / Fail). This runs in the
        # tool's worker thread, so waiting on them here doesn't block pydantic-ai's event loop.
        while ok and self._resolve and isinstance(call, Bash) and (missing := detect_missing(call.command, output)):
            self._failures[missing.name] += 1
            self._emit(Event(kind="blocked", text=f"{missing.name} is missing",
                             detail=describe(missing, self._failures[missing.name]), ok=False))
            if self._resolve(missing, self._failures[missing.name]) is Fix.RETRY:
                output = call.run(self.workspace)
                continue
            output += FAIL_NOTE
            break
        self._emit(Event(kind="result", text=summary, detail=output, ok=ok))
        # Same stuck detection as the builtin loop: the same call getting the same result again.
        sig = (call.model_dump_json(), output)
        self._seen[sig] += 1
        if self._seen[sig] >= LOOP_AFTER:
            self._repeats += 1
            if self._repeats >= 2:
                raise _Abort("model is repeating itself")  # run_turn reports it
            self._emit(Event(kind="error", text="model is repeating itself", ok=False))
            return clip(output) + REPEAT_NOTE
        return clip(output)

    def _build(self, model: Model | str) -> Agent:
        # sequential=True on every tool: a model may batch dependent calls (edit_file + bash pytest) in one
        # response and pydantic-ai runs a batch concurrently by default, so the tests could run before the edit
        # lands. The builtin engine takes one action at a time; this keeps the same ordering.
        # retries=3 matches the builtin engine's tolerance for bad replies (pydantic-ai's default of 1
        # gives up after a single malformed tool call).
        agent = Agent(model, instructions=self._instructions, output_type=[str, DeferredToolRequests],
                      model_settings={"temperature": 0.2}, retries=3)

        @agent.tool(sequential=True)
        def list_dir(ctx: RunContext, path: str = ".") -> str:
            """List a directory."""
            return self._run(ctx, ListDir(tool="list_dir", path=path))

        @agent.tool(sequential=True)
        def read_file(ctx: RunContext, path: str) -> str:
            """Read a file, with line numbers."""
            return self._run(ctx, ReadFile(tool="read_file", path=path))

        @agent.tool(sequential=True)
        def grep(ctx: RunContext, pattern: str, path: str = ".") -> str:
            """Regex search in the files under path."""
            return self._run(ctx, Grep(tool="grep", pattern=pattern, path=path))

        @agent.tool(sequential=True)
        def write_file(ctx: RunContext, path: str, content: str) -> str:
            """Create or overwrite a file with its COMPLETE content."""
            return self._run(ctx, WriteFile(tool="write_file", path=path, content=content))

        @agent.tool(sequential=True)
        def edit_file(ctx: RunContext, path: str, old: str, new: str) -> str:
            """Replace exactly one occurrence of `old` with `new` in a file."""
            return self._run(ctx, EditFile(tool="edit_file", path=path, old=old, new=new))

        @agent.tool(sequential=True)
        def bash(ctx: RunContext, command: str) -> str:
            """Run a shell command in the project directory."""
            return self._run(ctx, Bash(tool="bash", command=command))

        return agent

    def _instructions(self) -> str:
        ws = self.workspace
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in ws.root.iterdir() if c.name != ".git")
        listing = "\n".join(names[:40]) + (f"\n... [{len(names) - 40} more]" if len(names) > 40 else "")
        return INSTRUCTIONS.replace("<ROOT>", str(ws.root)).replace("<LISTING>", listing or "(empty)")
