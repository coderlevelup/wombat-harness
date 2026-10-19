"""python -m wombat_harness.tasks [--engine E] [--model M] [task ...]: run acceptance tasks, one JSON line each."""
from __future__ import annotations

import argparse
import sys

from ..agent import Policy
from ..agent.factory import DEFAULT_MODEL, ENGINES, make_agent
from . import TASKS, run_task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m wombat_harness.tasks")
    p.add_argument("names", nargs="*", help=f"tasks to run (default all): {', '.join(TASKS)}")
    p.add_argument("--engine", choices=ENGINES, default="builtin")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-steps", type=int, default=15)
    args = p.parse_args(argv)
    unknown = [n for n in args.names if n not in TASKS]
    if unknown:
        p.error(f"unknown task(s): {', '.join(unknown)}")
    ok = True
    for name in args.names or TASKS:
        result = run_task(TASKS[name], lambda root: make_agent(
            args.engine, args.model, args.base_url, root, Policy.YOLO, args.max_steps))
        print(result.model_dump_json(), flush=True)
        ok &= result.done
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
