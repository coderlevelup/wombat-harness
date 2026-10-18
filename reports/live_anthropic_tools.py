"""Live run of the `pai-tools` engine (pydantic-ai function tools) on Anthropic's claude-haiku-5-5.

    WOMBAT_ANTHROPIC_API_KEY=... python reports/live_anthropic_tools.py [yolo_runs]

Same task as compare_engines.py. Runs `yolo_runs` (default 3) in mode `yolo`, then ONE run in mode `ask`
with an auto-approving callback that records every approval it was asked for, so pydantic-ai's deferred
approval round trip is exercised against a real hosted model. Costs real money (a few cents).
Key only from WOMBAT_ANTHROPIC_API_KEY (ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL are ignored); the model's
base URL is asserted to be api.anthropic.com before any call. Success is judged by re-running pytest
ourselves, plus a non-empty reply. `tool_retries` counts retry prompts pydantic-ai sent back to the model
(bad tool arguments); for each we keep the message, to see what kind of mistake it was.
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

from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.anthropic import AnthropicModel

from compare_engines import MAX_STEPS, PROMPT
from wombat_harness.agent import Policy
from wombat_harness.agent.factory import make_agent

MODEL = "anthropic:claude-haiku-5-5"
EXPECTED_URL = "https://api.anthropic.com"
YOLO_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

_log: list[dict] = []
_orig_request = AnthropicModel.request


async def _logged_request(self, messages, model_settings, model_request_parameters):
    resp = await _orig_request(self, messages, model_settings, model_request_parameters)
    retries = [p for p in getattr(messages[-1], "parts", []) if isinstance(p, RetryPromptPart)]
    _log.append({
        "retry_prompt": [str(p.content)[:400] for p in retries],
        "text": "".join(p.content for p in resp.parts if isinstance(p, TextPart))[:300],
        "tool_calls": [(p.tool_name, str(p.args)[:200]) for p in resp.parts if isinstance(p, ToolCallPart)],
        "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
    })
    return resp


AnthropicModel.request = _logged_request


def run_one(policy: Policy) -> dict:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")
        (root / "test_calc.py").write_text(
            "from calc import add, mul\n\ndef test_add():\n    assert add(1, 2) == 3\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
        agent = make_agent("pai-tools", MODEL, None, root, policy, MAX_STEPS)
        model = agent._agent.model
        assert str(model.client.base_url).rstrip("/") == EXPECTED_URL, model.client.base_url
        _log.clear()
        events: list = []
        asked: list[str] = []
        t0 = time.time()
        try:
            reply = agent.run_turn(PROMPT, events.append, lambda s, p: asked.append(s) or True)
            crash = None
        except Exception as e:
            reply, crash = "", f"{type(e).__name__}: {e}"
        secs = round(time.time() - t0, 1)
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                               cwd=root, capture_output=True, text=True)
        retry_msgs = [m for x in _log for m in x["retry_prompt"]]
        return {
            "engine": "pai-tools", "model": MODEL, "mode": policy.value, "secs": secs,
            "success": tests.returncode == 0 and bool(reply) and not crash,
            "tests_pass": tests.returncode == 0, "replied": bool(reply),
            "tool_calls": sum(e.kind == "tool" for e in events),
            "tools": [e.text for e in events if e.kind == "tool"],
            "results": [(e.text, e.ok, e.detail[:160]) for e in events if e.kind == "result"],  # in completion order
            "tool_errors": [e.detail[:200] for e in events if e.kind == "result" and not e.ok],
            "tool_retries": len(retry_msgs), "retry_messages": retry_msgs,
            "approvals_asked": asked,
            "errors": [e.text + (f" :: {e.detail[:300]}" if e.detail else "") for e in events if e.kind == "error"],
            "api_calls": len(_log),
            "input_tokens": sum(x["input_tokens"] or 0 for x in _log),
            "output_tokens": sum(x["output_tokens"] or 0 for x in _log),
            "crash": crash, "reply": reply[:300],
        }


if __name__ == "__main__":
    if not os.environ.get("WOMBAT_ANTHROPIC_API_KEY"):
        sys.exit("set WOMBAT_ANTHROPIC_API_KEY")
    for policy in [Policy.YOLO] * YOLO_RUNS + [Policy.ASK]:
        print(json.dumps(run_one(policy)), flush=True)
