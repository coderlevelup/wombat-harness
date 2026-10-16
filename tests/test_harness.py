import json

import httpx
import pytest

from wombat_harness import CodeResponse, Harness, LLMError, OpenAICompatClient, Task, run_python


class Scripted:
    """Fake client replaying canned CodeResponses / errors."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, schema):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def good(code):
    return CodeResponse(plan="p", code=code)


def test_fence_stripped_and_syntax_rejected():
    assert good("```python\nx = 1\n```").code == "x = 1"
    with pytest.raises(ValueError):
        good("def (:")


def test_retries_with_feedback_until_tests_pass():
    c = Scripted(good("def add(a, b): return a - b"), good("def add(a, b): return a + b"))
    r = Harness(c).run(Task(prompt="add", tests="assert add(1, 2) == 3"))
    assert r.ok and len(r.attempts) == 2
    assert "AssertionError" in c.calls[1][-1]["content"]


def test_llm_error_recovers_and_exhaustion_fails():
    c = Scripted(LLMError("bad json"), good("raise SystemExit(1)"), good("1/0"))
    r = Harness(c, max_attempts=3).run(Task(prompt="x"))
    assert not r.ok and r.attempts[0].error and r.code == "1/0"


def test_timeout():
    assert run_python("while True: pass", timeout=0.5).timed_out


def test_client_sends_schema_and_parses():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        content = '```json\n{"plan": "p", "code": "print(1)"}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    c = OpenAICompatClient(transport=httpx.MockTransport(handler))
    out = c.complete([{"role": "user", "content": "hi"}], CodeResponse)
    assert out.code == "print(1)"
    assert seen["response_format"]["json_schema"]["schema"]["properties"]["code"]
