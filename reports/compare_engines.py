"""Run each engine x model on the same bug-fix task and judge the result independently.

    python reports/compare_engines.py builtin:qwen2.5-coder:7b pai-tools:qwen2.5-coder:7b ...

Each spec is ENGINE:MODEL. Task: calc.py has `add` returning `a - b`; the agent is asked to fix the
failing test and run the tests. Success = calc.py now adds AND pytest passes when *we* run it AND the
agent produced a reply. The agent runs in `yolo` mode (nothing to approve), max 12 steps.
Writes one JSON line per run to stdout. Needs a running Ollama with the models pulled.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from wombat_harness.agent import Policy
from wombat_harness.agent.factory import DEFAULT_BASE_URL, make_agent

PROMPT = "The test in test_calc.py is failing. Find the bug in calc.py and fix it, then run the tests."
MAX_STEPS = 12


def warm(model: str) -> None:
    httpx.post("http://localhost:11434/api/generate", timeout=900, json={
        "model": model, "prompt": "hi", "stream": False, "keep_alive": "30m", "options": {"num_predict": 1}})


def run_one(engine: str, model: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")
        (root / "test_calc.py").write_text(
            "from calc import add, mul\n\ndef test_add():\n    assert add(1, 2) == 3\n\ndef test_mul():\n    assert mul(2, 3) == 6\n")
        events: list = []
        agent = make_agent(engine, model, DEFAULT_BASE_URL, root, Policy.YOLO, MAX_STEPS)
        t0 = time.time()
        try:
            reply = agent.run_turn(PROMPT, events.append, lambda s, p: True)
            crash = None
        except Exception as e:  # an engine crash is a result too
            reply, crash = "", f"{type(e).__name__}: {e}"
        secs = round(time.time() - t0, 1)
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                               cwd=root, capture_output=True, text=True)
        errors = [e.text + (f" :: {e.detail[:300]}" if e.detail else "") for e in events if e.kind == "error"]
        return {
            "engine": engine, "model": model, "secs": secs,
            "success": tests.returncode == 0 and bool(reply) and not crash,
            "tests_pass": tests.returncode == 0, "replied": bool(reply),
            "tool_calls": sum(e.kind == "tool" for e in events),
            "tools": [e.text for e in events if e.kind == "tool"],
            "errors": errors, "crash": crash, "reply": reply[:300],
        }


if __name__ == "__main__":
    warmed = set()
    for spec in sys.argv[1:]:
        engine, model = spec.split(":", 1)
        if model not in warmed:
            warm(model)
            warmed.add(model)
        print(json.dumps(run_one(engine, model)), flush=True)
