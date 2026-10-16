import threading

import pytest
from textual.widgets import Button, OptionList, RichLog

from wombat_harness import CodeResponse, Task
from wombat_harness.tui.app import RunState, WombatApp
from wombat_harness.tui.runner import DemoRunner, HarnessRunner, Target, demo_targets


def log_text(app) -> str:
    log = app.query_one("#log", RichLog)
    return "\n".join("".join(seg.text for seg in strip) for strip in log.lines)


async def settle(app, pilot):
    """Wait for the worker thread, then let its queued messages be handled."""
    await app.workers.wait_for_complete()
    await pilot.pause()


class BlockUntilCancelled:
    """Runs until cancelled, then lingers in teardown until the test calls release()."""

    def __init__(self):
        self.released = threading.Event()

    def release(self):
        self.released.set()

    def run(self, target, emit, cancel):
        emit("started")
        cancel.wait(5)
        self.released.wait(5)
        return False


class Crashes:
    def run(self, target, emit, cancel):
        raise RuntimeError("boom [bold]")


async def test_layout_and_initial_state():
    app = WombatApp(DemoRunner(delay=0))
    async with app.run_test() as pilot:
        for sel in ("Header", "Footer", "#targets", "#log", "#run", "#stop"):
            app.query_one(sel)
        assert app.state is RunState.IDLE
        assert app.query_one("#targets", OptionList).option_count == len(demo_targets())
        assert not app.query_one("#run", Button).disabled
        assert app.query_one("#stop", Button).disabled


async def test_run_streams_output_and_returns_to_idle():
    app = WombatApp(DemoRunner(delay=0, steps=3))
    async with app.run_test() as pilot:
        await pilot.press("r")
        await settle(app, pilot)
        assert app.state is RunState.IDLE
        text = log_text(app)
        assert "step 1/3" in text and "step 3/3" in text and "finished" in text
        assert "\x1b" not in text  # ANSI was rendered, not shown raw


async def test_controls_disable_while_running_and_reenable_after():
    runner = BlockUntilCancelled()
    app = WombatApp(runner)
    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.pause()
        assert app.query_one("#targets", OptionList).disabled
        assert app.query_one("#run", Button).disabled
        assert not app.query_one("#stop", Button).disabled
        await pilot.press("r")  # re-entrancy: a second run must not start
        assert app.state is RunState.RUNNING
        await pilot.press("s")
        assert app.state is RunState.TEARDOWN
        assert app.query_one("#stop", Button).disabled
        assert app.query_one("#targets", OptionList).disabled  # still busy while winding down
        runner.release()
        await settle(app, pilot)
        assert app.state is RunState.IDLE
        assert not app.query_one("#targets", OptionList).disabled
        assert not app.query_one("#run", Button).disabled
        assert "stopped" in log_text(app)


async def test_footer_hides_actions_that_do_not_apply():
    runner = BlockUntilCancelled()
    app = WombatApp(runner)
    async with app.run_test() as pilot:
        assert app.check_action("run", ()) and not app.check_action("stop", ())
        await pilot.press("r")
        await pilot.pause()
        assert not app.check_action("run", ()) and app.check_action("stop", ())
        await pilot.press("s")
        assert not app.check_action("run", ()) and not app.check_action("stop", ())  # teardown
        runner.release()
        await settle(app, pilot)


async def test_buttons_work_too():
    app = WombatApp(DemoRunner(delay=0, steps=1))
    async with app.run_test() as pilot:
        await pilot.click("#run")
        await settle(app, pilot)
        assert "finished" in log_text(app)


async def test_selecting_target_changes_what_runs():
    seen = []

    class Recorder:
        def run(self, target, emit, cancel):
            seen.append(target.name)
            return True

    app = WombatApp(Recorder())
    async with app.run_test() as pilot:
        await pilot.press("down", "r")
        await settle(app, pilot)
        assert seen == [demo_targets()[1].name]


async def test_crashing_runner_does_not_wedge_the_ui():
    app = WombatApp(Crashes())
    async with app.run_test() as pilot:
        await pilot.press("r")
        await settle(app, pilot)
        assert app.state is RunState.IDLE
        assert "RuntimeError: boom [bold]" in log_text(app)  # shown literally, not as markup


async def test_no_targets_disables_run():
    app = WombatApp(DemoRunner(delay=0), targets=[])
    async with app.run_test() as pilot:
        assert app.query_one("#run", Button).disabled
        await pilot.press("r")
        assert app.state is RunState.IDLE


async def test_clear():
    app = WombatApp(DemoRunner(delay=0, steps=1))
    async with app.run_test() as pilot:
        await pilot.press("r")
        await settle(app, pilot)
        await pilot.press("c")
        assert log_text(app).strip() == ""


# -- engine-side pieces, no Textual involved -----------------------------------------------

def test_demo_runner_stops_on_cancel():
    out, cancel = [], threading.Event()
    cancel.set()
    assert DemoRunner(delay=0).run(Target(name="t"), out.append, cancel) is False
    assert any("cancelled" in line for line in out)


class Scripted:
    def __init__(self, *codes):
        self.codes = list(codes)

    def complete(self, messages, schema):
        return CodeResponse(plan="p", code=self.codes.pop(0))


def test_harness_runner_streams_attempts():
    t = Target(name="add", task=Task(prompt="add", tests="assert add(1, 2) == 3"))
    out = []
    runner = HarnessRunner(Scripted("def add(a, b): return a - b", "def add(a, b): return a + b"))
    assert runner.run(t, out.append, threading.Event()) is True
    text = "\n".join(out)
    assert "attempt 1: failed" in text and "attempt 2: ok" in text and "passed" in text


def test_harness_runner_honours_cancel_between_attempts():
    t = Target(name="add", task=Task(prompt="add", tests="assert False"))
    cancel, out = threading.Event(), []
    runner = HarnessRunner(Scripted("x = 1", "x = 2", "x = 3"), max_attempts=3)
    runner.harness.client = type("C", (), {
        "complete": lambda self, m, s: (cancel.set(), CodeResponse(plan="p", code="x = 1"))[1]
    })()
    assert runner.run(t, out.append, cancel) is False
    assert sum("attempt" in line for line in out) == 1 and any("cancelled" in line for line in out)


def test_harness_runner_target_without_task():
    out = []
    assert HarnessRunner(Scripted()).run(Target(name="x"), out.append, threading.Event()) is False
