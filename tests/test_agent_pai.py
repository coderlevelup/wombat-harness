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


def test_denial_leaves_file_untouched_and_tells_the_model(ws):
    s = Script(call("edit_file", path="a.py", old="a - b", new="a + b"), say("ok, not editing"))
    _, reply, ev = run(s, ws, approve=lambda sm, pv: False)
    assert "a - b" in (ws.root / "a.py").read_text()
    assert [e.kind for e in ev] == ["denied", "reply"]
    assert "denied" in tool_returns(s)[0]


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
