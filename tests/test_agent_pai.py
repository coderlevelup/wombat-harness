import json

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from wombat_harness import LLMError, LLMRequestError  # noqa: E402
from wombat_harness.agent import Policy, Step, Workspace  # noqa: E402
from wombat_harness.agent_pai import PydanticAIAgent, PydanticAIClient  # noqa: E402
from wombat_harness.agent_pai.client import to_messages  # noqa: E402


class Script:
    """A FunctionModel that replays canned responses and records what it was sent."""

    def __init__(self, *responses):
        self.responses, self.seen = list(responses), []
        self.model = FunctionModel(self._fn)

    def _fn(self, messages, info):
        self.seen.append(messages)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name, thought="", **args):
    parts = ([TextPart(thought)] if thought else []) + [ToolCallPart(name, args, tool_call_id=f"id{abs(hash(json.dumps(args)))}")]
    return ModelResponse(parts=parts)


def say(text):
    return ModelResponse(parts=[TextPart(text)])


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "a.py").write_text("def add(a, b):\n    return a - b\n")
    return Workspace(tmp_path)


def run(script, ws, text="go", policy=Policy.ASK, approve=lambda s, p: True, **kw):
    agent = PydanticAIAgent(script.model, ws, policy, **kw)
    events = []
    reply = agent.run_turn(text, events.append, approve)
    return agent, reply, events


def tool_returns(script, i=-1):
    return [p.content for m in script.seen[i] for p in m.parts if isinstance(p, ToolReturnPart)]


# -- behaviour parity with the builtin engine -------------------------------------------------

def test_read_then_reply(ws):
    s = Script(call("read_file", "reading it", path="a.py"), say("it subtracts"))
    _, reply, ev = run(s, ws)
    assert reply == "it subtracts"
    assert [e.kind for e in ev] == ["thought", "tool", "result", "reply"]
    assert ev[0].text == "reading it" and "return a - b" in ev[2].detail
    assert "return a - b" in tool_returns(s)[0]  # the observation reached the model


def test_ask_policy_gates_edit_with_diff_preview(ws):
    asked = []
    s = Script(call("edit_file", path="a.py", old="a - b", new="a + b"), say("done"))
    _, reply, ev = run(s, ws, approve=lambda sm, pv: asked.append((sm, pv)) or True)
    assert len(asked) == 1 and asked[0][0] == "edit_file a.py"
    assert "-    return a - b" in asked[0][1] and "+    return a + b" in asked[0][1]
    assert "a + b" in (ws.root / "a.py").read_text() and reply == "done"
    assert [e.kind for e in ev] == ["tool", "result", "reply"]


def test_denial_ends_the_turn_and_the_next_message_steers_the_model(ws):
    s = Script(call("edit_file", path="a.py", old="a - b", new="a + b"), say("ok, leaving it"))
    agent = PydanticAIAgent(s.model, ws)
    ev = []
    reply = agent.run_turn("fix it", ev.append, lambda sm, pv: False)
    assert "a - b" in (ws.root / "a.py").read_text()  # untouched
    assert reply == "" and [e.kind for e in ev] == ["denied"]
    assert len(s.seen) == 1  # the model was not asked again in the denied turn

    ev2 = []
    reply2 = agent.run_turn("just explain it", ev2.append, lambda sm, pv: True)
    assert reply2 == "ok, leaving it"
    # the pending edit was closed with a denial result, and the new message rides in the same request
    last_request = s.seen[1][-1]
    kinds = [type(p).__name__ for p in last_request.parts]
    assert kinds == ["ToolReturnPart", "UserPromptPart"]
    assert "denied" in last_request.parts[0].content and "turn has ended" in last_request.parts[0].content
    assert last_request.parts[1].content == "just explain it"


def test_denial_in_a_batch_closes_every_pending_call(ws):
    """Read + edit + bash in one response: the read runs, the edit is denied, the bash never runs; none is left dangling."""
    ran = []
    from wombat_harness.agent.tools import Bash

    batch = ModelResponse(parts=[
        ToolCallPart("read_file", {"path": "a.py"}, tool_call_id="r1"),
        ToolCallPart("edit_file", {"path": "a.py", "old": "a - b", "new": "a + b"}, tool_call_id="e1"),
        ToolCallPart("bash", {"command": "pytest"}, tool_call_id="b1"),
    ])
    s = Script(batch, say("fine"))
    agent = PydanticAIAgent(s.model, ws)
    Bash.run, orig = (lambda self, w: ran.append("bash") or "x"), Bash.run
    try:
        asked = []
        agent.run_turn("go", lambda e: None, lambda sm, pv: asked.append(sm) or False)
        assert asked == ["edit_file a.py"]  # stopped asking at the first denial
        assert ran == []
        agent.run_turn("ok", lambda e: None, lambda sm, pv: True)
    finally:
        Bash.run = orig
    returns = {p.tool_call_id: p.content for p in s.seen[1][-1].parts if isinstance(p, ToolReturnPart)}
    assert set(returns) == {"r1", "e1", "b1"}  # every call has a result
    assert "return a - b" in returns["r1"] and "denied" in returns["e1"] and "denied" in returns["b1"]


@pytest.mark.parametrize("policy,expected", [
    (Policy.ASK, ["edit_file a.py", "bash: true"]),
    (Policy.EDITS, ["bash: true"]),
    (Policy.YOLO, []),
])
def test_policies(ws, policy, expected):
    asked = []
    s = Script(call("edit_file", path="a.py", old="a - b", new="a + b"), call("bash", command="true"), say("x"))
    run(s, ws, policy=policy, approve=lambda sm, pv: asked.append(sm) or True)
    assert asked == expected


def test_reads_never_ask(ws):
    s = Script(call("list_dir"), call("grep", pattern="def"), say("x"))
    run(s, ws, approve=lambda sm, pv: pytest.fail("should not ask"))


def test_tool_error_is_an_observation_not_a_crash(ws):
    s = Script(call("read_file", path="missing.py"), say("x"))
    _, _, ev = run(s, ws)
    assert [e.ok for e in ev if e.kind == "result"] == [False]
    assert "does not exist" in tool_returns(s)[0]


def test_workspace_escape_is_refused(ws):
    s = Script(call("read_file", path="../../etc/passwd"), say("x"))
    _, _, ev = run(s, ws)
    assert "outside the workspace" in tool_returns(s)[0]


def test_step_limit(ws):
    s = Script(*[call("read_file", path=f"f{i}") for i in range(10)])
    _, reply, ev = run(s, ws, max_steps=3)
    assert reply == "" and ev[-1].kind == "error" and "3 steps" in ev[-1].text


def test_stop_before_the_model_is_asked(ws):
    s = Script(say("never"))
    agent = PydanticAIAgent(s.model, ws)
    ev = []
    assert agent.run_turn("x", ev.append, lambda a, b: True, should_stop=lambda: True) == ""
    assert ev[-1].kind == "stopped"


def test_repeat_guard(ws):
    s = Script(*[call("list_dir")] * 6)
    _, reply, ev = run(s, ws)
    assert reply == "" and ev[-1].text == "model is repeating itself"


def test_history_persists_across_turns_and_reset_clears_it(ws):
    s = Script(say("one"), say("two"), say("three"))
    agent = PydanticAIAgent(s.model, ws)
    agent.run_turn("first", lambda e: None, lambda a, b: True)
    agent.run_turn("second", lambda e: None, lambda a, b: True)
    texts = [p.content for m in s.seen[1] for p in m.parts if hasattr(p, "content") and isinstance(p.content, str)]
    assert "first" in texts and "second" in texts
    agent.reset()
    agent.run_turn("third", lambda e: None, lambda a, b: True)
    texts = [p.content for m in s.seen[2] for p in m.parts if hasattr(p, "content") and isinstance(p.content, str)]
    assert "first" not in texts


def test_system_prompt_names_the_workspace(ws):
    s = Script(say("x"))
    run(s, ws)
    assert str(ws.root) in s.seen[0][0].instructions and "a.py" in s.seen[0][0].instructions


# -- model errors -----------------------------------------------------------------------------

def test_missing_model_error_is_readable(ws):
    err = ModelHTTPError(404, "nope:1b", {"error": {"message": "model 'nope:1b' not found"}})
    _, reply, ev = run(Script(err), ws)
    assert reply == "" and len(ev) == 1
    assert "404" in ev[0].text and "not found" in ev[0].text and "ollama pull nope:1b" in ev[0].text


@pytest.mark.parametrize("body", [
    {"error": {"message": "gemma3:4b does not support tools"}},          # wrapped
    {"message": "gemma3:4b does not support tools", "type": "invalid_request_error"},  # what the openai SDK gives
])
def test_no_tool_support_error_points_at_the_native_engine(ws, body):
    _, _, ev = run(Script(ModelHTTPError(400, "gemma3:4b", body)), ws)
    assert "does not support tools" in ev[0].text and "pai-native" in ev[0].text
    assert "{" not in ev[0].text  # no raw dict repr


# -- the client adapter (pai-native) ----------------------------------------------------------

def step_json(tool="list_dir", **kw):
    return say(json.dumps({"thought": "t", "action": {"tool": tool, **kw}}))


def test_to_messages_splits_system_merges_users_and_pops_the_prompt():
    system, history, prompt = to_messages([
        {"role": "system", "content": "SYS"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "{}"}, {"role": "user", "content": "obs1"},
        {"role": "user", "content": "obs2"},
    ])
    assert system == "SYS" and prompt == "obs1\n\nobs2"
    assert [type(m).__name__ for m in history] == ["ModelRequest", "ModelResponse"]
    with pytest.raises(ValueError):
        to_messages([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}])


def test_client_returns_a_validated_step_and_sends_instructions():
    s = Script(step_json("read_file", path="a.py"))
    out = PydanticAIClient(pai_model=s.model).complete(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "task"}], Step)
    assert out.action.tool == "read_file" and out.action.path == "a.py"
    assert s.seen[0][0].instructions == "SYS"


def test_client_errors_map_to_the_shared_error_types():
    with pytest.raises(LLMRequestError, match="404.*ollama pull"):
        PydanticAIClient(model="m:1b", pai_model=Script(ModelHTTPError(404, "m:1b", {"error": {"message": "model 'm:1b' not found"}})).model
                         ).complete([{"role": "user", "content": "x"}], Step)
    bad = Script(say("not json"), say("still not json"), say("nope"))
    with pytest.raises(LLMError) as e:
        PydanticAIClient(pai_model=bad.model).complete([{"role": "user", "content": "x"}], Step)
    assert not isinstance(e.value, LLMRequestError)


def test_builtin_loop_runs_on_the_pydantic_ai_client(ws):
    """The point of pai-native: the existing loop, with pydantic-ai as its model client."""
    from wombat_harness.agent import Agent

    s = Script(step_json("read_file", path="a.py"), step_json("finish", message="subtracts"))
    events = []
    reply = Agent(PydanticAIClient(pai_model=s.model), ws).run_turn("what does a.py do?", events.append, lambda a, b: True)
    assert reply == "subtracts" and [e.kind for e in events][-1] == "reply"


# -- regression: the TUI calls the engine from a different worker thread each turn -------------

@pytest.fixture
def chat_server():
    """A real keep-alive HTTP server, so httpx builds a real connection pool (mock transports don't)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    reply = {"content": "hello"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply["content"]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", reply
    server.shutdown()


def _http_model(base_url):
    import httpx
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel("m", provider=OpenAIProvider(
        base_url=base_url, api_key="k", http_client=httpx.AsyncClient()))


def _from_threads(fn, n=3):
    import threading

    results, errors = [], []

    def go():
        try:
            results.append(fn())
        except BaseException as e:  # noqa: BLE001 - we want to see any failure
            errors.append(e)

    for _ in range(n):
        t = threading.Thread(target=go)
        t.start()
        t.join()
    return results, errors


def test_client_survives_being_called_from_different_threads(chat_server):
    base_url, reply = chat_server
    reply["content"] = json.dumps({"thought": "t", "action": {"tool": "list_dir"}})
    client = PydanticAIClient(pai_model=_http_model(base_url))
    results, errors = _from_threads(lambda: client.complete([{"role": "user", "content": "x"}], Step), n=4)
    assert not errors and len(results) == 4


def test_engine_survives_being_called_from_different_threads(ws, chat_server):
    base_url, reply = chat_server
    agent = PydanticAIAgent(_http_model(base_url), ws)
    results, errors = _from_threads(lambda: agent.run_turn("hi", lambda e: None, lambda a, b: True), n=4)
    assert not errors and results == ["hello"] * 4


# --- provider selection (no network: only constructs clients) -----------------------------------

SECRET = "sk-ant-test-SECRET-123"


def test_build_model_anthropic_targets_api_host_ignoring_env_url(monkeypatch):
    from wombat_harness.agent_pai.client import build_model

    monkeypatch.setenv("WOMBAT_ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://container-proxy.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-other")
    m = build_model("anthropic:claude-haiku-5-5", None)
    assert m.model_name == "claude-haiku-5-5"
    assert str(m.client.base_url).startswith("https://api.anthropic.com")
    assert m.client.api_key == SECRET  # the WOMBAT_ variable, not ANTHROPIC_API_KEY


def test_build_model_anthropic_without_key_names_variable_not_value(monkeypatch):
    from wombat_harness.agent_pai.client import build_model

    monkeypatch.delenv("WOMBAT_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)  # must not be used as a fallback
    with pytest.raises(SystemExit) as e:
        build_model("anthropic:claude-haiku-5-5", None)
    assert "WOMBAT_ANTHROPIC_API_KEY" in str(e.value) and SECRET not in str(e.value)


def test_build_model_ollama_default_and_bare_tag():
    from wombat_harness.agent_pai.client import build_model

    m = build_model("qwen2.5-coder:7b", None)
    assert m.model_name == "qwen2.5-coder:7b" and "localhost:11434" in str(m.client.base_url)


def test_client_ollama_default_not_applied_to_anthropic(monkeypatch):
    monkeypatch.setenv("WOMBAT_ANTHROPIC_API_KEY", SECRET)
    c = PydanticAIClient("anthropic:claude-haiku-5-5")
    assert c.base_url == "https://api.anthropic.com"


def test_error_text_is_provider_aware():
    from wombat_harness.agent_pai.client import describe_model_error

    msg = describe_model_error(ConnectionError("boom"), "claude-haiku-5-5", "https://api.anthropic.com", "anthropic")
    assert "Ollama" not in msg and "api.anthropic.com" in msg


def test_factory_rejects_provider_prefix_for_builtin():
    from wombat_harness.agent.factory import make_agent

    with pytest.raises(SystemExit):
        make_agent("builtin", "anthropic:claude-haiku-5-5", None, ".", Policy.YOLO, 3)


def test_invalid_replies_counted():
    bad = Script(say("not json"), step_json("list_dir"))
    client = PydanticAIClient(pai_model=bad.model)
    client.complete([{"role": "user", "content": "x"}], Step)
    assert client.invalid_replies == 1


# -- parallel tool calls: dependent calls in one response must not race -----------------------------

def test_dependent_tool_calls_in_one_response_run_in_order(ws, monkeypatch):
    """A model may batch `edit_file` and `bash pytest` into ONE response. pydantic-ai runs a batch's tool
    calls concurrently by default, so the tests could run before the edit lands and report stale results.
    The builtin engine is one action per step; the pydantic-ai engine must not be weaker than that."""
    import time

    from wombat_harness.agent.tools import Bash, EditFile

    order = []
    real_edit = EditFile.run

    def slow_edit(self, w):
        time.sleep(0.3)  # long enough that an unordered bash call would finish first
        out = real_edit(self, w)
        order.append("edit_done")
        return out

    monkeypatch.setattr(EditFile, "run", slow_edit)
    monkeypatch.setattr(Bash, "run", lambda self, w: order.append("bash_ran") or "exit code 0")
    batch = ModelResponse(parts=[
        ToolCallPart("edit_file", {"path": "a.py", "old": "a - b", "new": "a + b"}, tool_call_id="e1"),
        ToolCallPart("bash", {"command": "python -m pytest -q"}, tool_call_id="b1"),
    ])
    run(Script(batch, say("done")), ws, policy=Policy.YOLO)
    assert order == ["edit_done", "bash_ran"]


def test_repeat_guard_catches_a_two_step_cycle(ws):
    cycle = [call("read_file", path="a.py"), call("edit_file", path="a.py", old="nope", new="x")] * 8
    s = Script(*cycle)
    _, reply, ev = run(s, ws, policy=Policy.YOLO, max_steps=20)
    assert reply == "" and ev[-1].text == "model is repeating itself"
    assert sum(e.text == "model is repeating itself" for e in ev) == 2  # warned once, then stopped; not reported thrice
    assert len(s.responses) > 4  # stopped before the script ran out


def test_repeat_guard_allows_rereading_after_a_change(ws):
    s = Script(
        call("read_file", path="a.py"),
        call("edit_file", path="a.py", old="a - b", new="a * b"),
        call("read_file", path="a.py"),
        call("edit_file", path="a.py", old="a * b", new="a / b"),
        call("read_file", path="a.py"),
        say("done"),
    )
    _, reply, ev = run(s, ws, policy=Policy.YOLO)
    assert reply == "done" and not any(e.kind == "error" for e in ev)


# -- a tool the model asked for is missing: Retry / Fail --------------------------------------------------

MISSING = "exit code 1\n/usr/bin/python: No module named pytest"
WORKED = "exit code 0\n2 passed in 0.01s"


def scripted_bash(monkeypatch, *outputs):
    from wombat_harness.agent.tools import Bash

    ran = []

    def fake(self, w):
        ran.append(self.command)
        return outputs[min(len(ran) - 1, len(outputs) - 1)]

    monkeypatch.setattr(Bash, "run", fake)
    return ran


def run_resolving(script, ws, resolve, text="go"):
    agent = PydanticAIAgent(script.model, ws, Policy.YOLO)
    ev = []
    reply = agent.run_turn(text, ev.append, lambda s, p: True, resolve=resolve)
    return reply, ev


def test_retry_runs_the_command_again_and_the_model_never_sees_the_failure(ws, monkeypatch):
    from wombat_harness.agent.missing import Fix

    ran = scripted_bash(monkeypatch, MISSING, WORKED)
    asked = []
    s = Script(call("bash", command="python -m pytest -q"), say("done"))
    reply, ev = run_resolving(s, ws, lambda m, n: asked.append((m.name, n)) or Fix.RETRY)
    assert reply == "done" and ran == ["python -m pytest -q"] * 2 and asked == [("pytest", 1)]
    blocked = [e for e in ev if e.kind == "blocked"]
    assert len(blocked) == 1 and "pytest" in blocked[0].text and "failed once" in blocked[0].detail
    returned = tool_returns(s)[0]
    assert "2 passed" in returned and "No module named" not in returned


def test_fail_gives_the_error_back_to_the_model_to_recover(ws, monkeypatch):
    from wombat_harness.agent.missing import Fix

    scripted_bash(monkeypatch, MISSING)
    s = Script(call("bash", command="python -m pytest -q"), say("no tests then"))
    reply, _ = run_resolving(s, ws, lambda m, n: Fix.FAIL)
    returned = tool_returns(s)[0]
    assert reply == "no tests then" and "No module named pytest" in returned and "chose not to resolve" in returned


def test_failure_count_includes_the_models_own_retries(ws, monkeypatch):
    from wombat_harness.agent.missing import Fix

    scripted_bash(monkeypatch, MISSING)
    counts = []
    s = Script(call("bash", command="python -m pytest -q"), call("bash", command="python -m pytest"), say("gave up"))
    reply, ev = run_resolving(s, ws, lambda m, n: counts.append(n) or Fix.FAIL)
    assert counts == [1, 2] and reply == "gave up"
    assert "failed 2 times" in [e for e in ev if e.kind == "blocked"][1].detail


def test_without_a_resolver_the_failure_goes_straight_to_the_model(ws, monkeypatch):
    scripted_bash(monkeypatch, MISSING)
    s = Script(call("bash", command="python -m pytest -q"), say("x"))
    reply, ev = run_resolving(s, ws, None)
    assert reply == "x" and not any(e.kind == "blocked" for e in ev)
    assert "No module named pytest" in tool_returns(s)[0]
