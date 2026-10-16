from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .client import OpenAICompatClient
from .harness import Harness
from .models import Task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wombat", description="Generate Python with a local model.")
    ap.add_argument("prompt", help="What to build")
    ap.add_argument("--tests", type=Path, help="File of assert-style tests the code must pass")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    task = Task(prompt=a.prompt, tests=a.tests.read_text() if a.tests else None)
    res = Harness(OpenAICompatClient(a.model, a.base_url), a.attempts).run(task)
    if a.verbose:
        for i, at in enumerate(res.attempts, 1):
            status = "ok" if at.run and at.run.ok else (at.error or at.run.feedback() if at.run else "no run")
            print(f"[attempt {i}] {status}", file=sys.stderr)
    if res.code:
        print(res.code)
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
