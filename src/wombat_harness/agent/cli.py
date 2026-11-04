"""Headless agent: `wombat-agent "fix the failing test"`. Prompts y/n/a on stdin for approvals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .factory import DEFAULT_MODEL, ENGINES, HELP, make_agent
from .loop import Event, Policy
from .missing import Fix, Missing

DIM, RED, GREEN, YELLOW, BOLD, RESET = "\x1b[2m", "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[1m", "\x1b[0m"


def show(e: Event) -> None:
    if e.kind == "thought":
        print(f"{DIM}  {e.text}{RESET}")
    elif e.kind == "tool":
        print(f"{BOLD}{YELLOW}● {e.text}{RESET}")
    elif e.kind == "result":
        body = (e.detail or e.text).strip().splitlines()
        shown = body[:8] + ([f"... {len(body) - 8} more lines"] if len(body) > 8 else [])
        print(f"{DIM if e.ok else RED}" + "\n".join("    " + ln for ln in shown) + RESET)
    elif e.kind == "blocked":
        print(f"{BOLD}{YELLOW}⚠ {e.text}{RESET}")
        print(f"{DIM}" + "\n".join("    " + ln for ln in e.detail.splitlines()) + RESET)
    elif e.kind == "denied":
        print(f"{YELLOW}✗ denied: {e.text}{RESET}")
    elif e.kind == "reply":
        print(f"\n{e.text}\n")
    elif e.kind in ("error", "stopped"):
        print(f"{RED}! {e.text}{RESET}" + (f"\n{DIM}{e.detail}{RESET}" if e.detail else ""))


def resolve_missing(missing: Missing, failures: int, *, tty: bool | None = None, ask=input) -> Fix:
    """Ask on a terminal whether to Retry (after fixing it) or Fail (tell the model); without one, Fail."""
    if not (sys.stdin.isatty() if tty is None else tty):
        print("  (no tty: failing)")
        return Fix.FAIL
    while True:
        ans = ask("  fix it in another terminal, then: [r]etry / [f]ail (tell the model): ").strip().lower()
        if ans in ("r", "retry"):
            return Fix.RETRY
        if ans in ("f", "fail", ""):
            return Fix.FAIL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wombat-agent", description="Run the coding agent once, non-interactively.")
    ap.add_argument("prompt")
    ap.add_argument("--workspace", type=Path, default=Path("."))
    ap.add_argument("--mode", choices=[p.value for p in Policy], default="ask")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None, help="override the provider's default endpoint")
    ap.add_argument("--engine", choices=ENGINES, default="builtin", help=HELP)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--check-first", metavar="CMD", help="run this command (e.g. 'python -m pytest -q') before each turn and show the model its output")
    a = ap.parse_args(argv)

    agent = make_agent(a.engine, a.model, a.base_url, a.workspace, Policy(a.mode), a.max_steps, check=a.check_first)

    def approve(summary: str, preview: str) -> bool:
        print(f"{BOLD}{YELLOW}? {summary}{RESET}")
        for ln in preview.splitlines():
            color = GREEN if ln.startswith("+") else RED if ln.startswith("-") else DIM
            print(f"{color}    {ln}{RESET}")
        if not sys.stdin.isatty():
            print("  (no tty: denied)")
            return False
        ans = input("  allow? [y]es / [n]o / [a]ll this session: ").strip().lower()
        if ans == "a":
            agent.policy = Policy.YOLO
        return ans in ("y", "a")

    reply = agent.run_turn(a.prompt, show, approve, resolve=resolve_missing)
    return 0 if reply else 1


if __name__ == "__main__":
    raise SystemExit(main())
