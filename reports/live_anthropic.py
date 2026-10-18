"""Live run of the `pai-native` engine against Anthropic's API (claude-haiku-5-5), same task as compare_engines.py.

    WOMBAT_ANTHROPIC_API_KEY=... python reports/live_anthropic.py [runs]

Costs real money (a few cents). The key is read only from WOMBAT_ANTHROPIC_API_KEY; ANTHROPIC_API_KEY and
ANTHROPIC_BASE_URL are ignored, and the client's base URL is asserted to be api.anthropic.com before any call.
Per run we record: success (judged independently: see `judge` in compare_engines.py), tool calls, errors, invalid replies
(pydantic-ai output retries), wall time and token usage. For every invalid reply we also keep what the model
sent and the validation message pydantic-ai sent back, to see *what kind* of mistake it was.
Writes one JSON line per run to stdout.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart
from pydantic_ai.models.anthropic import AnthropicModel

from compare_engines import MAX_STEPS, ORIGINAL_TEST, PROMPT, judge  # same task, same prompt, same strict judge
from wombat_harness.agent import Policy
from wombat_harness.agent.factory import make_agent

MODEL = "anthropic:claude-haiku-5-5"
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EXPECTED_URL = "https://api.anthropic.com"

# Observation hook: wrap AnthropicModel.request to log each exchange. It changes nothing about the call.
_log: list[dict] = []
_orig_request = AnthropicModel.request


async def _logged_request(self, messages, model_settings, model_request_parameters):
    resp = await _orig_request(self, messages, model_settings, model_request_parameters)
    last = messages[-1]
    retries = [p for p in getattr(last, "parts", []) if isinstance(p, RetryPromptPart)]
    text = "".join(p.content for p in resp.parts if isinstance(p, TextPart))
    _log.append({
        "retry_prompt": [str(p.content)[:400] for p in retries],  # why the *previous* reply was rejected
        "reply": text[:400],
        "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
        "cache_read": resp.usage.cache_read_tokens, "cache_write": resp.usage.cache_write_tokens,
        "model": resp.model_name,
    })
    return resp


AnthropicModel.request = _logged_request


def run_one() -> dict:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")
        (root / "test_calc.py").write_text(ORIGINAL_TEST)
        agent = make_agent("pai-native", MODEL, None, root, Policy.YOLO, MAX_STEPS)
        client = agent.client
        assert client.base_url == EXPECTED_URL, client.base_url
        assert str(client._model.client.base_url).rstrip("/") == EXPECTED_URL, client._model.client.base_url
        _log.clear()
        events: list = []
        t0 = time.time()
        try:
            reply = agent.run_turn(PROMPT, events.append, lambda s, p: True)
            crash = None
        except Exception as e:  # an engine crash is a result too
            reply, crash = "", f"{type(e).__name__}: {e}"
        secs = round(time.time() - t0, 1)
        verdict = judge(root, ORIGINAL_TEST)
        return {
            "engine": "pai-native", "model": MODEL, "secs": secs,
            "success": all(verdict.values()) and bool(reply) and not crash,
            **verdict, "replied": bool(reply),
            "tool_calls": sum(e.kind == "tool" for e in events),
            "tools": [e.text for e in events if e.kind == "tool"],
            "errors": [e.text + (f" :: {e.detail[:300]}" if e.detail else "") for e in events if e.kind == "error"],
            "invalid_replies": client.invalid_replies,
            "api_calls": len(_log),
            "input_tokens": sum(x["input_tokens"] or 0 for x in _log),
            "output_tokens": sum(x["output_tokens"] or 0 for x in _log),
            "cache_read": sum(x["cache_read"] or 0 for x in _log),
            "invalid_detail": [x for x in _log if x["retry_prompt"]],
            "crash": crash, "reply": reply[:300],
        }


if __name__ == "__main__":
    if not os.environ.get("WOMBAT_ANTHROPIC_API_KEY"):
        sys.exit("set WOMBAT_ANTHROPIC_API_KEY")
    for _ in range(RUNS):
        print(json.dumps(run_one()), flush=True)
