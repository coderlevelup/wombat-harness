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
from collections import deque
from collections.abc import Callable

from pydantic_ai import (
    Agent, ApprovalRequired, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied, UsageLimits,
)
from pydantic_ai.exceptions import (
    ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded,
)
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai.models import Model

from ..agent.loop import Approve, Emit, Event, Policy, needs_approval
from ..agent.tools import Bash, EditFile, Grep, ListDir, ReadFile, ToolError, WriteFile, clip
from ..agent.workspace import Workspace
from .client import describe_model_error

INSTRUCTIONS = """\
You are wombat, a coding agent working in the project directory described below. Complete the \
user's task by calling tools. Read files before editing them. Prefer edit_file for small changes. \
Verify your work (e.g. `python -m pytest -q` via bash; never run a .py test file with bash \
directly). Paths are relative to the project directory. When the task is done, or the user only \
asked a question, reply with a short plain-text answer and no tool call.

Project directory: <ROOT>
Top level:
<LISTING>"""

DENIED = "The user denied this action. Do not retry it; ask what they want instead."
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
        self._recent: deque[str] = deque(maxlen=2)
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
                 should_stop: Callable[[], bool] | None = None) -> str:
        self._emit, self._recent = on_event, deque(maxlen=2)
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
                deferred = self._decide(result.output, approve)
                prompt = None
                if stop():
                    self._emit(Event(kind="stopped", text="stopped"))
                    return ""
                continue
            self._history = history
            self._emit(Event(kind="reply", text=result.output))
            return result.output

    def _decide(self, requests: DeferredToolRequests, approve: Approve) -> DeferredToolResults:
        approvals: dict[str, bool | ToolDenied] = {}
        for part in requests.approvals:
            call = TOOLS[part.tool_name](tool=part.tool_name, **part.args_as_dict())
            if approve(call.summary(), call.preview(self.workspace) or ""):
                approvals[part.tool_call_id] = True
            else:
                self._emit(Event(kind="denied", text=call.summary(), ok=False))
                approvals[part.tool_call_id] = ToolDenied(DENIED)
        return DeferredToolResults(approvals=approvals)

    # -- tools ----------------------------------------------------------------------------
    def _run(self, ctx: RunContext, call) -> str:
        key = call.model_dump_json()
        if list(self._recent) == [key, key]:  # third identical call in a row
            self._recent.clear()
            raise _Abort("model is repeating itself")
        self._recent.append(key)
        if needs_approval(self.policy, call) and not ctx.tool_call_approved:
            raise ApprovalRequired
        summary = call.summary()
        self._emit(Event(kind="tool", text=summary))
        try:
            output, ok = call.run(self.workspace), True
        except ToolError as e:
            output, ok = f"error: {e}", False
        self._emit(Event(kind="result", text=summary, detail=output, ok=ok))
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
