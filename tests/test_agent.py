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


# -- a tool the model asked for is missing: Retry / Fail --------------------------------------------

from wombat_harness.agent.missing import Fix  # noqa: E402

MISSING = "exit code 1\n/usr/bin/python: No module named pytest"
WORKED = "exit code 0\n2 passed in 0.01s"


def scripted_bash(monkeypatch, *outputs):
    """Bash.run returns the given outputs in turn (the last repeats) and records the commands."""
    ran = []

    def run(self, w):
        ran.append(self.command)
        return outputs[min(len(ran) - 1, len(outputs) - 1)]

    monkeypatch.setattr(Bash, "run", run)
    return ran


def run_with(agent, resolve, text="go"):
    ev = []
    reply = agent.run_turn(text, ev.append, lambda s, p: True, resolve=resolve)
    return reply, ev


def test_retry_runs_the_command_again_and_the_model_never_sees_the_failure(ws, monkeypatch):
    ran = scripted_bash(monkeypatch, MISSING, WORKED)
    asked = []
    c = Scripted(step("bash", command="python -m pytest -q"), step("finish", message="done"))
    reply, ev = run_with(Agent(c, ws, Policy.YOLO), lambda m, n: asked.append((m.name, n)) or Fix.RETRY)
    assert reply == "done" and ran == ["python -m pytest -q"] * 2
    assert asked == [("pytest", 1)]
    blocked = [e for e in ev if e.kind == "blocked"]
    assert len(blocked) == 1 and "pytest" in blocked[0].text and "failed once" in blocked[0].detail
    seen_by_model = c.seen[1][-1]["content"]
    assert "2 passed" in seen_by_model and "No module named" not in seen_by_model


def test_fail_tells_the_model_to_suggest_the_install_steps(ws, monkeypatch):
    """Fail = "I'm not retrying": the model is told what is missing and asked for the install steps, which it can propose
    to run (the person approves) or give as text for the person to run themselves before asking it to retry."""
    scripted_bash(monkeypatch, MISSING)
    c = Scripted(step("bash", command="python -m pytest -q"), step("finish", message="run: python -m pip install pytest"))
    reply, ev = run_with(Agent(c, ws, Policy.YOLO), lambda m, n: Fix.FAIL)
    assert reply == "run: python -m pip install pytest"
    note = c.seen[1][-1]["content"]
    assert "No module named pytest" in note  # the original failure is still there
    assert "did not retry" in note and "python -m pip install pytest" in note  # told what to suggest, with the sure hint
    assert "propose running it with bash" in note and "call finish" in note and "ask you to retry" in note


def test_fail_note_for_a_plain_missing_command_does_not_guess_an_installer(ws, monkeypatch):
    scripted_bash(monkeypatch, "exit code 127\nbash: line 1: ruff: command not found")
    c = Scripted(step("bash", command="ruff check ."), step("finish", message="x"))
    run_with(Agent(c, ws, Policy.YOLO), lambda m, n: Fix.FAIL)
    note = c.seen[1][-1]["content"]
    assert "ruff" in note and "did not retry" in note and "pip install" not in note


def test_a_retry_asked_for_in_the_next_message_just_works(ws, monkeypatch):
    """The 'copy the steps, run them yourself, tell it to retry' path: the next turn re-runs the command."""
    scripted_bash(monkeypatch, MISSING, WORKED)
    c = Scripted(step("bash", command="python -m pytest -q"), step("finish", message="install pytest, then ask me to retry"),
                 step("bash", command="python -m pytest -q"), step("finish", message="passes now"))
    agent = Agent(c, ws, Policy.YOLO)
    run_with(agent, lambda m, n: Fix.FAIL)
    reply, _ = run_with(agent, None, text="installed it, retry")
    assert reply == "passes now" and "2 passed" in agent.messages[-2]["content"]


def test_failure_count_includes_the_models_own_retries(ws, monkeypatch):
    scripted_bash(monkeypatch, MISSING)
    counts = []
    c = Scripted(step("bash", command="python -m pytest -q"), step("bash", command="python -m pytest"),
                 step("finish", message="gave up"))
    reply, ev = run_with(Agent(c, ws, Policy.YOLO), lambda m, n: counts.append(n) or Fix.FAIL)
    assert counts == [1, 2] and reply == "gave up"
    assert "failed 2 times" in [e for e in ev if e.kind == "blocked"][1].detail


def test_without_a_resolver_the_failure_goes_straight_to_the_model(ws, monkeypatch):
    scripted_bash(monkeypatch, MISSING)
    c = Scripted(step("bash", command="python -m pytest -q"), step("finish", message="x"))
    reply, ev = run_with(Agent(c, ws, Policy.YOLO), None)
    assert reply == "x" and not any(e.kind == "blocked" for e in ev)
    assert "No module named pytest" in c.seen[1][-1]["content"]


def test_only_bash_failures_can_block(ws, monkeypatch):
    c = Scripted(step("read_file", path="missing.py"), step("finish", message="x"))
    run_with(Agent(c, ws), lambda m, n: pytest.fail("resolve must not be called"))


# -- actions that cannot succeed are not put to the person ---------------------------------------------------

def test_edit_that_cannot_match_is_not_put_to_the_person(ws):
    """The preview used to say "(this edit will fail...)" and still ask; people learned to press y on things that couldn't work."""
    asked = []
    c = Scripted(step("edit_file", path="a.py", old="NOT THERE", new="x"), step("finish", message="ok"))
    reply, ev = run(Agent(c, ws), approve=lambda s, p: asked.append(s) or True)
    assert asked == [] and reply == "ok"
    assert [(e.kind, e.ok) for e in ev if e.kind == "result"] == [("result", False)]
    assert "not found" in c.seen[1][-1]["content"]  # the model gets the error and can correct itself


def test_ambiguous_edit_is_not_put_to_the_person(ws):
    (ws.root / "dup.py").write_text("x\nx\n")
    asked = []
    c = Scripted(step("edit_file", path="dup.py", old="x", new="y"), step("finish", message="ok"))
    run(Agent(c, ws), approve=lambda s, p: asked.append(s) or True)
    assert asked == [] and "matches 2" in c.seen[1][-1]["content"]


def test_write_that_cannot_happen_is_not_put_to_the_person(ws):
    asked = []
    c = Scripted(step("write_file", path="../outside.py", content="x = 1\n"),
                 step("write_file", path="sub", content="x = 1\n"), step("finish", message="ok"))
    (ws.root / "sub").mkdir(exist_ok=True)
    run(Agent(c, ws), approve=lambda s, p: asked.append(s) or True)
    assert asked == []
    assert "outside the workspace" in c.seen[1][-1]["content"] and "is a directory" in c.seen[2][-1]["content"]
    assert not (ws.root.parent / "outside.py").exists()


def test_an_edit_that_can_succeed_is_still_asked(ws):
    asked = []
    c = Scripted(step("edit_file", path="a.py", old="a - b", new="a + b"), step("finish", message="ok"))
    run(Agent(c, ws), approve=lambda s, p: asked.append(s) or True)
    assert asked == ["edit_file a.py"]


# -- paths shown relative to the workspace; test-file edits flagged ------------------------------------------

def test_absolute_paths_inside_the_workspace_become_relative(ws):
    from wombat_harness.agent.tools import EditFile, relativize

    abs_path = str(ws.root / "a.py")
    assert relativize(EditFile(tool="edit_file", path=abs_path, old="x", new="y"), ws).path == "a.py"
    assert relativize(EditFile(tool="edit_file", path="a.py", old="x", new="y"), ws).path == "a.py"  # already relative
    assert relativize(EditFile(tool="edit_file", path="/etc/passwd", old="x", new="y"), ws).path == "/etc/passwd"  # outside: untouched
    assert relativize(EditFile(tool="edit_file", path="../x.py", old="x", new="y"), ws).path == "../x.py"
    assert relativize(ListDir(tool="list_dir", path=str(ws.root)), ws).path == "."
    from wombat_harness.agent.tools import Bash
    cmd = Bash(tool="bash", command="ls")
    assert relativize(cmd, ws) == cmd  # tools without a path are unchanged


def test_the_person_sees_short_paths_even_when_the_model_sends_absolute_ones(ws):
    asked = []
    c = Scripted(step("edit_file", path=str(ws.root / "a.py"), old="a - b", new="a + b"), step("finish", message="ok"))
    run(Agent(c, ws), approve=lambda s, p: asked.append((s, p)) or True)
    summary, preview = asked[0]
    assert summary == "edit_file a.py" and "--- a/a.py" in preview and str(ws.root) not in preview + summary


@pytest.mark.parametrize("path,flagged", [
    ("test_calc.py", True), ("calc_test.py", True), ("conftest.py", True), ("tests/helpers.py", True),
    ("src/tests/x.py", True), ("web/app.test.ts", True), ("web/app.spec.js", True),
    ("calc.py", False), ("latest.py", False), ("contest.py", False), ("src/attestation.py", False),
])
def test_test_file_edits_are_flagged(ws, path, flagged):
    from wombat_harness.agent.tools import EditFile, is_test_path

    assert is_test_path(path) is flagged
    (ws.root / path).parent.mkdir(parents=True, exist_ok=True)
    (ws.root / path).write_text("a - b\n")
    preview = EditFile(tool="edit_file", path=path, old="a - b", new="a + b").preview(ws)
    assert preview.startswith("⚠") is flagged


def test_the_prompts_tell_the_model_not_to_edit_tests_to_make_them_pass(ws):
    from wombat_harness.agent.loop import SYSTEM

    assert "not the tests" in SYSTEM


# -- check-first ----------------------------------------------------------------------------

def test_check_output_is_shown_to_the_model_before_its_first_step(ws):
    c = Scripted(step("finish", message="done"))
    run(Agent(c, ws, check="echo FAILED test_a.py::test_add; exit 1"), text="fix it")
    first_user = c.seen[0][-1]["content"]
    assert first_user.startswith("fix it")
    assert "FAILED test_a.py::test_add" in first_user and "exit code 1" in first_user


def test_no_check_leaves_the_message_alone(ws):
    c = Scripted(step("finish", message="done"))
    run(Agent(c, ws), text="fix it")
    assert c.seen[0][-1]["content"] == "fix it"


def test_check_is_not_an_approval_prompt(ws):
    c = Scripted(step("finish", message="done"))
    asked = []
    run(Agent(c, ws, policy=Policy.ASK, check="true"), approve=lambda s, p: asked.append(s) or True)
    assert asked == []
