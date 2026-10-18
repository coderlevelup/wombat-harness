"""Reproduce a small-model run: the author's exact project, current code, builtin and pai-native engines.

    python reports/repro_gemma3_1b.py [model] [runs_per_engine]      # defaults: gemma3:1b, 5

The project is the one the author set up by hand on a Mac: `calc.py` with only `add` returning `a - b`, and a
`test_calc.py` with the single test `assert add(1, 2) == 3`. (compare_engines.py's project also has `mul`.)
Success uses the strict `judge` from compare_engines.py: hidden checks that `add` really adds, the visible test
file unmodified, visible pytest passing, and a reply. Passing pytest alone is not enough: this model edits the test.
Writes one JSON line per run, including the full event sequence, to stdout. Needs Ollama with the model pulled and
pytest installed in the Python that runs the agent's `bash` tool.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_engines import PROMPT, judge
from wombat_harness.agent import Policy
from wombat_harness.agent.factory import make_agent

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma3:1b"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TEST = "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def one(engine: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (root / "test_calc.py").write_text(TEST)
        agent = make_agent(engine, MODEL, None, root, Policy.YOLO, 12)
        events, t0 = [], time.time()
        try:
            reply, crash = agent.run_turn(PROMPT, events.append, lambda s, p: True), None
        except Exception as e:
            reply, crash = "", f"{type(e).__name__}: {e}"
        secs = round(time.time() - t0, 1)
        verdict = judge(root, TEST)
        names = [e.text.split()[1].replace(str(root), "<ws>") if e.text.startswith(("edit_file", "read_file", "write_file")) else e.text
                 for e in events if e.kind == "tool"]
        return {
            "engine": engine, "model": MODEL, "secs": secs,
            "success": all(verdict.values()) and bool(reply) and not crash, **verdict, "replied": bool(reply),
            "tool_summaries": [e.text.replace(str(root), "<ws>") for e in events if e.kind == "tool"],
            "errors": [e.text for e in events if e.kind == "error"],
            "touched_test_file": any(e.kind == "tool" and e.text.startswith(("edit_file", "write_file")) and "test_calc" in e.text for e in events),
            "touched_calc_py": any(e.kind == "tool" and e.text.startswith(("edit_file", "write_file")) and "calc.py" in e.text and "test_calc" not in e.text for e in events),
            "calc_py_final": (root / "calc.py").read_text(), "crash": crash, "reply": reply[:200],
        }


if __name__ == "__main__":
    for engine in ("builtin", "pai-native"):
        for _ in range(RUNS):
            print(json.dumps(one(engine)), flush=True)
