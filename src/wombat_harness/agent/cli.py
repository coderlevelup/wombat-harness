"""Headless agent: `wombat-agent "fix the failing test"`. Prompts y/n/a on stdin for approvals."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..client import OpenAICompatClient
from .loop import Agent, Event, Policy
from .workspace import Workspace

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
    elif e.kind == "denied":
        print(f"{YELLOW}✗ denied: {e.text}{RESET}")
    elif e.kind == "reply":
        print(f"\n{e.text}\n")
    elif e.kind in ("error", "stopped"):
        print(f"{RED}! {e.text}{RESET}" + (f"\n{DIM}{e.detail}{RESET}" if e.detail else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wombat-agent", description="Run the coding agent once, non-interactively.")
    ap.add_argument("prompt")
    ap.add_argument("--workspace", type=Path, default=Path("."))
    ap.add_argument("--mode", choices=[p.value for p in Policy], default="ask")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--max-steps", type=int, default=25)
    a = ap.parse_args(argv)

    agent = Agent(OpenAICompatClient(a.model, a.base_url), Workspace(a.workspace),
                  Policy(a.mode), a.max_steps)

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

    reply = agent.run_turn(a.prompt, show, approve)
    return 0 if reply else 1


if __name__ == "__main__":
    raise SystemExit(main())
