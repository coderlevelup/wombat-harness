"""Textual front end. Depends on the `Runner` port only, never on the engine directly.

Threading model: the blocking run lives in a thread worker. The worker never touches
widgets; it posts `LogLine` / `RunFinished` messages (post_message is thread-safe) and
the event loop does all rendering. State is one reactive, `state`, and everything that
should be disabled while busy derives from it.
"""
from __future__ import annotations

import threading
from enum import Enum

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, OptionList, RichLog
from textual.widgets.option_list import Option

from .runner import DemoRunner, Runner, Target, demo_targets


class RunState(Enum):
    IDLE = "Idle"
    RUNNING = "Running"
    TEARDOWN = "Teardown"  # stop requested; waiting for the worker to wind down


class LogLine(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class RunFinished(Message):
    def __init__(self, ok: bool, error: str | None = None) -> None:
        super().__init__()
        self.ok, self.error = ok, error


class WombatApp(App):
    CSS_PATH = "wombat.tcss"
    TITLE = "wombat"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "run", "Run"),
        Binding("s", "stop", "Stop"),
        Binding("c", "clear", "Clear log"),
    ]

    state: reactive[RunState] = reactive(RunState.IDLE)

    def __init__(self, runner: Runner | None = None, targets: list[Target] | None = None):
        super().__init__()
        self.runner: Runner = runner or DemoRunner()
        self.targets = targets if targets is not None else demo_targets()
        self.selected: Target | None = self.targets[0] if self.targets else None
        self._cancel = threading.Event()

    # -- layout ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("Targets", id="sidebar-title")
                yield OptionList(*(Option(t.name, id=t.name) for t in self.targets), id="targets")
            with Vertical(id="main"):
                with Horizontal(id="controls"):
                    yield Button("Run", id="run", variant="success")
                    yield Button("Stop", id="stop", variant="error")
                    yield Label("", id="status")
                yield Label("Output", id="log-title")
                yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        targets = self.query_one("#targets", OptionList)
        if self.selected:
            targets.highlighted = 0
        targets.focus()
        self._render_state()

    # -- state ----------------------------------------------------------------
    def watch_state(self) -> None:
        self._render_state()
        self.refresh_bindings()

    def _render_state(self) -> None:
        if not self.is_mounted:
            return
        idle = self.state is RunState.IDLE
        self.query_one("#targets", OptionList).disabled = not idle
        self.query_one("#run", Button).disabled = not (idle and self.selected)
        self.query_one("#stop", Button).disabled = self.state is not RunState.RUNNING
        status = self.query_one("#status", Label)
        status.update(self.state.value)
        status.set_classes(self.state.name.lower())

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Footer shows only the bindings that make sense right now."""
        if action == "run":
            return self.state is RunState.IDLE and self.selected is not None
        if action == "stop":
            return self.state is RunState.RUNNING
        return True

    # -- input ----------------------------------------------------------------
    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self.selected = self.targets[event.option_index]
        self._render_state()
        self.refresh_bindings()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.selected = self.targets[event.option_index]
        self.action_run()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            self.action_run()
        elif event.button.id == "stop":
            self.action_stop()

    def action_run(self) -> None:
        if self.state is not RunState.IDLE or self.selected is None:
            return
        self._cancel = threading.Event()
        self.state = RunState.RUNNING
        self._log(Text.from_markup(f"[bold]▶ {self.selected.name}[/bold]"))
        self._run_target(self.selected, self._cancel)

    def action_stop(self) -> None:
        if self.state is RunState.RUNNING:
            self.state = RunState.TEARDOWN
            self._cancel.set()

    def action_clear(self) -> None:
        self.query_one("#log", RichLog).clear()

    # -- worker ---------------------------------------------------------------
    @work(thread=True, exclusive=True, group="run")
    def _run_target(self, target: Target, cancel: threading.Event) -> None:
        """Runs in a thread. Only posts messages; never touches widgets."""
        try:
            ok = self.runner.run(target, lambda line: self.post_message(LogLine(line)), cancel)
            self.post_message(RunFinished(ok))
        except Exception as e:  # a crashing adapter must not wedge the UI in Running
            self.post_message(RunFinished(False, f"{type(e).__name__}: {e}"))

    def on_log_line(self, message: LogLine) -> None:
        self._log(Text.from_ansi(message.text))

    def on_run_finished(self, message: RunFinished) -> None:
        if message.error:
            self._log(Text.from_markup(f"[red]error: {Text(message.error).markup}[/red]"))
        elif self._cancel.is_set():
            self._log(Text.from_markup("[yellow]■ stopped[/yellow]"))
        else:
            self._log(Text.from_markup("[green]✔ finished[/green]" if message.ok else "[red]✘ failed[/red]"))
        self.state = RunState.IDLE
        self.query_one("#targets", OptionList).focus()

    def _log(self, renderable: Text) -> None:
        self.query_one("#log", RichLog).write(renderable)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="wombat-tui", description="Terminal UI for wombat-harness.")
    ap.add_argument("--live", action="store_true", help="Use a real local model instead of the demo runner")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args(argv)

    if a.live:
        from ..client import OpenAICompatClient
        from .runner import HarnessRunner, sample_targets

        app = WombatApp(HarnessRunner(OpenAICompatClient(a.model, a.base_url), a.attempts), sample_targets())
    else:
        app = WombatApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
