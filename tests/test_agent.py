import pytest

from wombat_harness import LLMError, LLMRequestError
from pydantic import ValidationError

from wombat_harness.agent import Agent, Policy, Step, ToolError, Workspace
from wombat_harness.agent.tools import Bash, EditFile, Grep, ListDir, ReadFile, WriteFile


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("hello\nworld\n")
    return Workspace(tmp_path)


# -- workspace ----------------------------------------------------------------------------

def test_workspace_blocks_escapes(ws, tmp_path):
    for bad in ("../x", "/etc/passwd", "sub/../../x"):
        with pytest.raises(ToolError, match="outside"):
            ws.resolve(bad)
    assert ws.resolve("sub/b.txt").name == "b.txt"
    assert ws.resolve("").name == tmp_path.name


def test_workspace_blocks_symlink_escape(ws, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret").write_text("s")
    (ws.root / "link").symlink_to(outside)
    with pytest.raises(ToolError, match="outside"):
        ws.resolve("link/secret")


# -- tools --------------------------------------------------------------------------------

def test_list_read_grep(ws):
    assert ListDir(tool="list_dir").run(ws).splitlines() == ["a.py", "sub/"]
    assert "   2      return a - b" in ReadFile(tool="read_file", path="a.py").run(ws)
    assert Grep(tool="grep", pattern="hel+o").run(ws) == "sub/b.txt:1: hello"
    assert Grep(tool="grep", pattern="zzz").run(ws) == "no matches"
    with pytest.raises(ToolError, match="bad regex"):
        Grep(tool="grep", pattern="(").run(ws)
    with pytest.raises(ToolError, match="does not exist"):
        ReadFile(tool="read_file", path="nope.py").run(ws)


def test_grep_skips_junk_dirs_and_binary(ws):
    (ws.root / ".git").mkdir()
    (ws.root / ".git" / "x").write_text("hello")
    (ws.root / "bin").write_bytes(b"\xff\xfe hello")
    assert Grep(tool="grep", pattern="hello").run(ws) == "sub/b.txt:1: hello"


def test_write_and_edit(ws):
    w = WriteFile(tool="write_file", path="new/dir/f.py", content="x = 1\n")
    assert "new file" not in w.preview(ws) and "+x = 1" in w.preview(ws)
    w.run(ws)
    assert (ws.root / "new/dir/f.py").read_text() == "x = 1\n"

    e = EditFile(tool="edit_file", path="a.py", old="a - b", new="a + b")
    assert "-    return a - b" in e.preview(ws) and "+    return a + b" in e.preview(ws)
    e.run(ws)
    assert "a + b" in (ws.root / "a.py").read_text()


def test_edit_errors_are_actionable(ws):
    with pytest.raises(ToolError, match="not found"):
        EditFile(tool="edit_file", path="a.py", old="zzz", new="y").run(ws)
    (ws.root / "dup.py").write_text("x\nx\n")
    with pytest.raises(ToolError, match="matches 2"):
        EditFile(tool="edit_file", path="dup.py", old="x", new="y").run(ws)
    assert "will fail" in EditFile(tool="edit_file", path="a.py", old="zzz", new="y").preview(ws)


def test_bash_runs_in_workspace(ws):
    out = Bash(tool="bash", command="ls && echo oops >&2; exit 3").run(ws)
    assert out.startswith("exit code 3") and "a.py" in out and "oops" in out
    Bash.timeout = 0.3
    try:
        assert "timed out" in Bash(tool="bash", command="sleep 5").run(ws)
    finally:
        Bash.timeout = 60.0


def test_step_schema_enforces_per_tool_arguments():
    ok = Step.model_validate({"thought": "t", "action": {"tool": "read_file", "path": "a.py"}})
    assert ok.action.path == "a.py"
    ok = Step.model_validate({"thought": "t", "action": {"tool": "list_dir"}})
    assert ok.action.path == "."
    for bad in (
        {"tool": "read_file"},                       # missing path
        {"tool": "read_file", "path": ""},           # empty path
        {"tool": "edit_file", "path": "a", "new": "b"},  # missing old
        {"tool": "write_file", "path": "a"},         # missing content
        {"tool": "rm", "path": "a"},                 # unknown tool
    ):
        with pytest.raises(ValidationError):
            Step.model_validate({"thought": "t", "action": bad})
    # empty content is a legitimate (empty) file when the model says so explicitly
    Step.model_validate({"thought": "t", "action": {"tool": "write_file", "path": "a", "content": ""}})


# -- loop ---------------------------------------------------------------------------------

class Scripted:
    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def complete(self, messages, schema):
        self.seen.append([dict(m) for m in messages])
        s = self.steps.pop(0)
        if isinstance(s, Exception):
            raise s
        return s


def step(tool, **kw):
    return Step.model_validate({"thought": f"do {tool}", "action": {"tool": tool, **kw}})


def run(agent, text="go", approve=lambda s, p: True):
    events = []
    reply = agent.run_turn(text, events.append, approve)
    return reply, events


def test_read_then_finish(ws):
    c = Scripted(step("read_file", path="a.py"), step("finish", message="it subtracts"))
    reply, ev = run(Agent(c, ws))
    assert reply == "it subtracts"
    assert [e.kind for e in ev] == ["thought", "tool", "result", "thought", "reply"]
    assert "return a - b" in c.seen[1][-1]["content"]  # observation was fed back


def test_ask_policy_gates_writes_and_denial_is_reported(ws):
    asked = []
    c = Scripted(step("edit_file", path="a.py", old="a - b", new="a + b"),
                 step("finish", message="ok"))

    def deny(summary, preview):
        asked.append((summary, preview))
        return False

    reply, ev = run(Agent(c, ws), approve=deny)
    assert asked and "+    return a + b" in asked[0][1]
    assert "a - b" in (ws.root / "a.py").read_text()  # untouched
    assert any(e.kind == "denied" for e in ev)
    assert reply == "" and len(c.steps) == 1  # the turn ended at the denial: the model was not asked again


def test_policies(ws):
    def go(policy):
        asked = []
        c = Scripted(step("edit_file", path="a.py", old="a - b", new="a + b"),
                     step("bash", command="true"), step("finish", message="x"))
        run(Agent(c, ws, policy), approve=lambda s, p: asked.append(s) or True)
        (ws.root / "a.py").write_text("def add(a, b):\n    return a - b\n")
        return [a.split()[0] for a in asked]

    assert go(Policy.ASK) == ["edit_file", "bash:"]
    assert go(Policy.EDITS) == ["bash:"]
    assert go(Policy.YOLO) == []


def test_reads_never_ask(ws):
    c = Scripted(step("list_dir"), step("grep", pattern="def"), step("finish", message="x"))
    run(Agent(c, ws), approve=lambda s, p: pytest.fail("should not ask"))


def test_tool_errors_reach_the_model(ws):
    c = Scripted(step("read_file", path="missing.py"), step("finish", message="x"))
    _, ev = run(Agent(c, ws))
    assert "does not exist" in c.seen[1][-1]["content"]
    assert [e.ok for e in ev if e.kind == "result"] == [False]


def test_invalid_replies_retry_then_give_up(ws):
    c = Scripted(LLMError("bad json"), step("finish", message="fine"))
    assert run(Agent(c, ws))[0] == "fine"
    c = Scripted(LLMError("a"), LLMError("b"), LLMError("c"))
    reply, ev = run(Agent(c, ws))
    assert reply == "" and sum(e.kind == "error" for e in ev) == 3


def test_repeat_guard(ws):
    c = Scripted(*[step("list_dir")] * 3, step("list_dir"), step("list_dir"), step("list_dir"))
    reply, ev = run(Agent(c, ws))
    assert reply == "" and any(e.text == "model is repeating itself" for e in ev)


def test_step_limit(ws):
    c = Scripted(*[step("read_file", path=f"f{i}") for i in range(5)])
    _, ev = run(Agent(c, ws, max_steps=3))
    assert ev[-1].kind == "error" and "3 steps" in ev[-1].text


def test_should_stop(ws):
    c = Scripted(step("list_dir"))
    ev = []
    assert Agent(c, ws).run_turn("x", ev.append, lambda s, p: True, should_stop=lambda: True) == ""
    assert ev[-1].kind == "stopped" and not c.seen


def test_conversation_persists_across_turns_and_reset(ws):
    c = Scripted(step("finish", message="one"), step("finish", message="two"))
    a = Agent(c, ws)
    run(a, "first")
    run(a, "second")
    assert [m["content"] for m in c.seen[1] if m["role"] == "user"][:2] == ["first", "second"]
    a.reset()
    assert len(a.messages) == 1 and "a.py" in a.messages[0]["content"]


def test_context_trimming_keeps_the_task(ws):
    (ws.root / "big.txt").write_text("x" * 3000)
    c = Scripted(*[step("read_file", path="big.txt")] * 1, step("grep", pattern="x"),
                 step("list_dir", path="sub"), step("finish", message="done"))
    a = Agent(c, ws, context_chars=3500)
    run(a, "the original task")
    last = c.seen[-1]
    assert last[0]["role"] == "system" and "the original task" in [m["content"] for m in last]
    assert sum(len(m["content"]) for m in last) <= 3500 + 600  # budget (+ the newest observation)


def test_old_turns_dropped_before_current_task(ws):
    a = Agent(Scripted(*[step("finish", message="r" * 400) for _ in range(4)]), ws, context_chars=1900)
    for i in range(4):
        run(a, f"task {i} " + "y" * 300)
    texts = [m["content"] for m in a.client.seen[-1]]
    assert any(t.startswith("task 3") for t in texts) and not any(t.startswith("task 0") for t in texts)


def test_request_error_ends_turn_immediately_with_the_real_message(ws):
    c = Scripted(LLMRequestError("model server returned 404: model 'x' not found (try: ollama pull x)"),
                 step("finish", message="never reached"))
    reply, ev = run(Agent(c, ws))
    assert reply == "" and len(ev) == 1 and ev[0].kind == "error" and "ollama pull x" in ev[0].text
    assert len(c.steps) == 1  # no retries


def test_denial_ends_the_turn_and_the_next_message_steers_the_model(ws):
    """Denying must hand control back to the person, not let the model carry on with its next action
    (it can't ask a question mid-turn: only `finish` ends a turn)."""
    c = Scripted(step("edit_file", path="a.py", old="a - b", new="a + b"),
                 step("finish", message="ok, leaving it"))
    agent = Agent(c, ws)
    ev = []
    reply = agent.run_turn("fix it", ev.append, lambda s, p: False)
    assert reply == "" and [e.kind for e in ev] == ["thought", "denied"]
    assert len(c.steps) == 1 and not c.seen[1:]  # no second model call in the denied turn

    reply2, ev2 = run(agent, "do nothing, just explain it")  # the person's next message
    assert reply2 == "ok, leaving it"
    history = [(m["role"], m["content"][:60]) for m in c.seen[1]]
    roles = [r for r, _ in history]
    assert roles == ["system", "user", "assistant", "user", "user"]
    assert "denied" in history[3][1] and "turn has ended" in c.seen[1][3]["content"]
    assert c.seen[1][4]["content"] == "do nothing, just explain it"


def test_denial_note_does_not_tell_the_model_to_ask_a_question(ws):
    c = Scripted(step("bash", command="rm -rf x"), step("finish", message="x"))
    a = Agent(c, ws)
    a.run_turn("go", lambda e: None, lambda s, p: False)
    note = a.messages[-1]["content"]
    assert "denied" in note and "ask" not in note.lower()


def test_repeat_guard_catches_a_two_step_cycle(ws):
    """read, failing edit, read, failing edit ...: no call repeats back to back, but each one keeps returning the
    same result. (Seen on a real run: a small model looped read_file / edit_file for the whole step budget.)"""
    cycle = [step("read_file", path="a.py"), step("edit_file", path="a.py", old="nope", new="x")] * 8
    c = Scripted(*cycle)
    reply, ev = run(Agent(c, ws, max_steps=20))
    assert reply == "" and sum(e.text == "model is repeating itself" for e in ev) == 2
    assert len(c.steps) > 6  # stopped well before the script (and the step budget) ran out


def test_repeat_guard_allows_rereading_after_a_change(ws):
    """The same read three times is fine when the file changed in between: the result differs."""
    c = Scripted(
        step("read_file", path="a.py"),
        step("edit_file", path="a.py", old="a - b", new="a * b"),
        step("read_file", path="a.py"),
        step("edit_file", path="a.py", old="a * b", new="a / b"),
        step("read_file", path="a.py"),
        step("finish", message="done"),
    )
    reply, ev = run(Agent(c, ws, Policy.YOLO))
    assert reply == "done" and not any(e.kind == "error" for e in ev)
