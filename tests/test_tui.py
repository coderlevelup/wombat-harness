import threading

from textual.widgets import Input, RichLog

from wombat_harness.agent import Event, Policy, Workspace
from wombat_harness.tui.app import ChatApp, ChatState


def transcript(app) -> str:
    log = app.query_one("#transcript", RichLog)
    return "\n".join("".join(seg.text for seg in strip) for strip in log.lines)


async def settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


async def submit(app, pilot, text):
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")


class FakeAgent:
    """Stands in for the real agent: the UI only knows the AgentPort."""

    def __init__(self, tmp_path, script=None):
        self.workspace = Workspace(tmp_path)
        self.policy = Policy.ASK
        self.script = script or self.default
        self.turns, self.resets = [], 0
        self.gate = threading.Event()
        self.gate.set()

    def reset(self):
        self.resets += 1

    def run_turn(self, text, on_event, approve, should_stop=lambda: False):
        self.turns.append(text)
        return self.script(self, text, on_event, approve, should_stop)

    @staticmethod
    def default(agent, text, on_event, approve, should_stop):
        on_event(Event(kind="thought", text="looking around"))
        on_event(Event(kind="tool", text="read_file a.py"))
        on_event(Event(kind="result", text="read_file a.py", detail="1  x = 1"))
        on_event(Event(kind="reply", text="All **done**."))
        return "All done."


def edit_script(agent, text, on_event, approve, should_stop):
    ok = approve("edit_file a.py", "--- a/a.py\n+++ b/a.py\n-x = 1\n+x = 2")
    if ok:
        on_event(Event(kind="tool", text="edit_file a.py"))
        on_event(Event(kind="result", text="edit_file a.py", detail="edited a.py"))
        on_event(Event(kind="reply", text="edited"))
    else:
        on_event(Event(kind="denied", text="edit_file a.py", ok=False))
        on_event(Event(kind="reply", text="not edited"))
    return ""


async def test_layout_and_idle(tmp_path):
    app = ChatApp(FakeAgent(tmp_path))
    async with app.run_test() as pilot:
        for sel in ("Header", "Footer", "#transcript", "#prompt", "#status", "#approval"):
            app.query_one(sel)
        assert app.state is ChatState.IDLE
        assert not app.query_one("#prompt", Input).disabled
        assert app.query_one("#prompt", Input).has_focus
        assert not app.query_one("#approval").has_class("visible")
        assert str(tmp_path.resolve()) in transcript(app)


async def test_chat_turn_renders_events_in_order(tmp_path):
    agent = FakeAgent(tmp_path)
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello agent")
        await settle(app, pilot)
        text = transcript(app)
        assert agent.turns == ["hello agent"]
        order = [text.index(s) for s in ("› hello agent", "looking around", "● read_file a.py", "x = 1", "All done.")]
        assert order == sorted(order)
        assert "**" not in text  # reply is rendered as markdown
        assert app.state is ChatState.IDLE
        assert app.query_one("#prompt", Input).value == ""


async def test_input_disabled_while_working(tmp_path):
    agent = FakeAgent(tmp_path)
    agent.gate.clear()

    def slow(a, text, on_event, approve, should_stop):
        a.gate.wait(5)
        return ""

    agent.script = slow
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "go")
        await pilot.pause()
        assert app.state is ChatState.RUNNING
        assert app.query_one("#prompt", Input).disabled
        assert not app.check_action("approve", ()) and app.check_action("interrupt", ())
        agent.gate.set()
        await settle(app, pilot)
        assert not app.query_one("#prompt", Input).disabled


async def test_approval_allow(tmp_path):
    agent = FakeAgent(tmp_path, edit_script)
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "edit it")
        await pilot.pause()
        assert app.state is ChatState.AWAITING
        assert app.query_one("#approval").has_class("visible")
        assert app.check_action("approve", ()) and app.query_one("#prompt", Input).disabled
        text = transcript(app)
        assert "? edit_file a.py" in text and "+x = 2" in text
        await pilot.press("y")
        await settle(app, pilot)
        text = transcript(app)
        assert "allowed" in text and "edited" in text
        assert app.state is ChatState.IDLE and not app.query_one("#approval").has_class("visible")
        assert agent.policy is Policy.ASK


async def test_approval_deny(tmp_path):
    app = ChatApp(FakeAgent(tmp_path, edit_script))
    async with app.run_test() as pilot:
        await submit(app, pilot, "edit it")
        await pilot.pause()
        await pilot.press("n")
        await settle(app, pilot)
        text = transcript(app)
        assert "denied" in text and "not edited" in text


async def test_approval_allow_all_switches_mode(tmp_path):
    agent = FakeAgent(tmp_path, edit_script)
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "edit it")
        await pilot.pause()
        await pilot.press("a")
        await settle(app, pilot)
        assert agent.policy is Policy.YOLO and "edited" in transcript(app)


async def test_y_n_a_do_nothing_when_not_awaiting(tmp_path):
    agent = FakeAgent(tmp_path)
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).blur()
        await pilot.press("a", "y", "n")
        assert agent.policy is Policy.ASK and app.state is ChatState.IDLE


async def test_escape_while_awaiting_denies_and_stops(tmp_path):
    seen = {}

    def script(a, text, on_event, approve, should_stop):
        seen["allowed"] = approve("bash: rm -rf x", "$ rm -rf x")
        a.gate.wait(5)  # linger so the test can observe the STOPPING state
        seen["stopped"] = should_stop()
        on_event(Event(kind="stopped", text="stopped"))
        return ""

    agent = FakeAgent(tmp_path, script)
    agent.gate.clear()
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "do it")
        await pilot.pause()
        assert app.state is ChatState.AWAITING
        await pilot.press("escape")
        assert app.state is ChatState.STOPPING
        assert not app.check_action("approve", ()) and not app.check_action("interrupt", ())
        agent.gate.set()
        await settle(app, pilot)
        assert seen == {"allowed": False, "stopped": True}
        assert app.state is ChatState.IDLE and "interrupted" in transcript(app)


async def test_escape_interrupts_running_turn(tmp_path):
    def script(a, text, on_event, approve, should_stop):
        while not should_stop():
            threading.Event().wait(0.01)
        return ""

    app = ChatApp(FakeAgent(tmp_path, script))
    async with app.run_test() as pilot:
        await submit(app, pilot, "go")
        await pilot.pause()
        assert app.state is ChatState.RUNNING
        await pilot.press("escape")
        await settle(app, pilot)
        assert app.state is ChatState.IDLE


async def test_crashing_agent_does_not_wedge_ui(tmp_path):
    def boom(*a):
        raise RuntimeError("kaboom [bold]")

    app = ChatApp(FakeAgent(tmp_path, boom))
    async with app.run_test() as pilot:
        await submit(app, pilot, "go")
        await settle(app, pilot)
        assert app.state is ChatState.IDLE
        assert "RuntimeError: kaboom [bold]" in transcript(app)  # literal, not markup


async def test_slash_commands(tmp_path):
    agent = FakeAgent(tmp_path)
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/mode yolo")
        assert agent.policy is Policy.YOLO and "mode: yolo" in transcript(app)
        await submit(app, pilot, "/mode nonsense")
        assert agent.policy is Policy.YOLO and "usage: /mode" in transcript(app)
        await submit(app, pilot, "/help")
        assert "/clear" in transcript(app)
        await submit(app, pilot, "/wat")
        assert "unknown command /wat" in transcript(app)
        await submit(app, pilot, "/clear")
        assert agent.resets == 1 and "new conversation" in transcript(app) and "/wat" not in transcript(app)
        assert agent.turns == []  # slash commands never reach the model


async def test_blank_input_ignored(tmp_path):
    agent = FakeAgent(tmp_path)
    app = ChatApp(agent)
    async with app.run_test() as pilot:
        await submit(app, pilot, "   ")
        assert agent.turns == [] and app.state is ChatState.IDLE


async def test_quit_commands(tmp_path):
    app = ChatApp(FakeAgent(tmp_path))
    async with app.run_test() as pilot:
        await submit(app, pilot, "/quit")
        await pilot.pause()
    assert app.return_code is not None or not app.is_running
