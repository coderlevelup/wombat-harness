"""Chat front end for the agent. Depends on `AgentPort` only, never on the model or tools.

Threading: a turn is a blocking call, so it runs in a thread worker. The worker never
touches widgets; it posts messages (thread-safe), and the event loop renders them. Approval
is a rendezvous: the worker posts `ApprovalRequest` and waits on an Event; y/n/a on the UI
side fills the answer and sets it. State is one reactive (`state`); the input box, the
status line and which footer keys exist all derive from it.
"""
from __future__ import annotations

import threading
from enum import Enum

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

from ..agent import AgentPort, Event, Policy

HELP = (
    "/help           this help\n"
    "/clear          start a new conversation\n"
    "/mode ask|edits|yolo   what needs approval (now: {mode})\n"
    "/quit           exit\n"
    "Esc interrupts a running turn."
)


class ChatState(Enum):
    IDLE = "ready"
    RUNNING = "working"
    AWAITING = "waiting for your approval"
    STOPPING = "stopping"


class AgentEvent(Message):
    def __init__(self, event: Event) -> None:
        super().__init__()
        self.event = event


class ApprovalRequest(Message):
    def __init__(self, summary: str, preview: str) -> None:
        super().__init__()
        self.summary, self.preview = summary, preview


class TurnFinished(Message):
    def __init__(self, error: str | None = None) -> None:
        super().__init__()
        self.error = error


class ChatApp(App):
    CSS_PATH = "wombat.tcss"
    TITLE = "wombat"
    BINDINGS = [
        Binding("y", "approve('y')", "Allow"),
        Binding("n", "approve('n')", "Deny"),
        Binding("a", "approve('a')", "Allow all"),
        Binding("escape", "interrupt", "Interrupt"),
        Binding("ctrl+c", "interrupt_or_quit", "Quit", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    state: reactive[ChatState] = reactive(ChatState.IDLE)

    def __init__(self, agent: AgentPort):
        super().__init__()
        self.agent = agent
        self._cancel = threading.Event()
        self._answered = threading.Event()
        self._answer = False

    # -- layout ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="transcript", wrap=True, markup=False, highlight=False)
        yield Static(id="approval")
        yield Static(id="status")
        yield Input(placeholder="Ask wombat to do something…  (/help)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self._write(Text(f"workspace: {self.agent.workspace.root}", style="dim"))
        self._write(Text("Describe a task. /help for commands.", style="dim"))
        self.query_one("#prompt", Input).focus()
        self._render_state()

    # -- state ----------------------------------------------------------------
    def watch_state(self) -> None:
        self._render_state()
        self.refresh_bindings()

    def _render_state(self) -> None:
        if not self.is_mounted:
            return
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = self.state is not ChatState.IDLE
        status = self.query_one("#status", Static)
        status.update(f"{self.state.value} · mode: {self.agent.policy.value}")
        status.set_classes(self.state.name.lower())
        self.query_one("#approval").set_class(self.state is ChatState.AWAITING, "visible")
        self.sub_title = f"{self.agent.workspace.root} · {self.agent.policy.value}"
        if self.state is ChatState.IDLE:
            prompt.focus()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "approve":
            return self.state is ChatState.AWAITING
        if action == "interrupt":
            return self.state in (ChatState.RUNNING, ChatState.AWAITING)
        return True

    # -- input ----------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.state is not ChatState.IDLE:
            return
        if text.startswith("/"):
            self._command(text)
            return
        self._write(Text(f"› {text}", style="bold cyan"))
        self._cancel = threading.Event()
        self.state = ChatState.RUNNING
        self._run_turn(text, self._cancel)

    def _command(self, text: str) -> None:
        cmd, _, arg = text.partition(" ")
        if cmd == "/help":
            self._write(Text(HELP.format(mode=self.agent.policy.value), style="dim"))
        elif cmd == "/clear":
            self.agent.reset()
            self.query_one("#transcript", RichLog).clear()
            self._write(Text("new conversation", style="dim"))
        elif cmd == "/mode":
            try:
                self.agent.policy = Policy(arg.strip())
            except ValueError:
                self._write(Text("usage: /mode ask|edits|yolo", style="red"))
            else:
                self._write(Text(f"mode: {self.agent.policy.value}", style="dim"))
                self._render_state()
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            self._write(Text(f"unknown command {cmd} (try /help)", style="red"))

    # -- approval -------------------------------------------------------------
    def action_approve(self, choice: str) -> None:
        if self.state is not ChatState.AWAITING:
            return
        if choice == "a":
            self.agent.policy = Policy.YOLO
        self._answer = choice in ("y", "a")
        self._write(Text("  ✓ allowed" if self._answer else "  ✗ denied", style="green" if self._answer else "yellow"))
        self.state = ChatState.RUNNING
        self._answered.set()

    def action_interrupt(self) -> None:
        if self.state in (ChatState.RUNNING, ChatState.AWAITING):
            self._cancel.set()
            self._answer = False
            self.state = ChatState.STOPPING
            self._answered.set()  # a pending approval counts as "no"

    def action_interrupt_or_quit(self) -> None:
        if self.state is ChatState.IDLE:
            self.exit()
        else:
            self.action_interrupt()

    def on_approval_request(self, message: ApprovalRequest) -> None:
        if self._cancel.is_set():  # interrupted while the request was in flight
            self._answered.set()
            return
        self._write(Text(f"? {message.summary}", style="bold yellow"))
        self._write(_colour_diff(message.preview))
        self.query_one("#approval", Static).update(
            Text.assemble(("Allow this?  ", "bold"), ("y", "bold green"), " yes   ",
                          ("n", "bold red"), " no   ", ("a", "bold yellow"), " yes to everything this session"))
        self.state = ChatState.AWAITING

    # -- worker ---------------------------------------------------------------
    @work(thread=True, exclusive=True, group="turn")
    def _run_turn(self, text: str, cancel: threading.Event) -> None:
        """Runs in a thread: only posts messages, never touches widgets."""

        def approve(summary: str, preview: str) -> bool:
            self._answered.clear()
            self._answer = False
            self.post_message(ApprovalRequest(summary, preview))
            self._answered.wait()
            return self._answer and not cancel.is_set()

        try:
            self.agent.run_turn(text, lambda e: self.post_message(AgentEvent(e)), approve, cancel.is_set)
            self.post_message(TurnFinished())
        except Exception as e:  # a crash must not leave the UI stuck on "working"
            self.post_message(TurnFinished(f"{type(e).__name__}: {e}"))

    def on_agent_event(self, message: AgentEvent) -> None:
        e = message.event
        if e.kind == "thought":
            self._write(Text(f"  {e.text}", style="dim italic"))
        elif e.kind == "tool":
            self._write(Text(f"● {e.text}", style="bold yellow"))
        elif e.kind == "result":
            self._write(_result(e))
        elif e.kind == "denied":
            self._write(Text(f"✗ denied: {e.text}", style="yellow"))
        elif e.kind == "reply":
            self._write(Markdown(e.text))
        elif e.kind == "error":
            self._write(Text(f"! {e.text}" + (f"\n  {e.detail}" if e.detail else ""), style="red"))
        elif e.kind == "stopped":
            self._write(Text("■ stopped", style="yellow"))

    def on_turn_finished(self, message: TurnFinished) -> None:
        if message.error:
            self._write(Text(f"error: {message.error}", style="red"))
        elif self._cancel.is_set():
            self._write(Text("■ interrupted", style="yellow"))
        self.state = ChatState.IDLE

    def _write(self, renderable) -> None:
        self.query_one("#transcript", RichLog).write(renderable)


def _result(e: Event, max_lines: int = 12) -> Text:
    lines = (e.detail or e.text).strip().splitlines() or ["(no output)"]
    more = len(lines) - max_lines
    body = "\n".join("    " + ln for ln in lines[:max_lines])
    if more > 0:
        body += f"\n    … {more} more lines"
    return Text(body, style="dim" if e.ok else "red")


def _colour_diff(preview: str, max_lines: int = 30) -> Text:
    out = Text()
    lines = preview.splitlines()
    for ln in lines[:max_lines]:
        style = "green" if ln.startswith("+") and not ln.startswith("+++") else \
            "red" if ln.startswith("-") and not ln.startswith("---") else "dim"
        out.append("    " + ln + "\n", style=style)
    if len(lines) > max_lines:
        out.append(f"    … {len(lines) - max_lines} more lines\n", style="dim")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from ..agent import Agent, Workspace
    from ..client import OpenAICompatClient

    ap = argparse.ArgumentParser(prog="wombat-tui", description="Chat with a local-model coding agent.")
    ap.add_argument("--workspace", type=Path, default=Path("."), help="Directory the agent may read and change")
    ap.add_argument("--mode", choices=[p.value for p in Policy], default="ask")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--max-steps", type=int, default=25)
    a = ap.parse_args(argv)

    agent = Agent(OpenAICompatClient(a.model, a.base_url), Workspace(a.workspace), Policy(a.mode), a.max_steps)
    ChatApp(agent).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
