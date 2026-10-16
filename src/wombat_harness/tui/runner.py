"""The port between the TUI and the engine. Nothing here imports Textual.

The UI talks only to `Runner`; adapters (`DemoRunner`, `HarnessRunner`) plug the engine in.
Adapters are plain blocking code: the app runs them in a worker thread, they push output
through `emit`, and they poll `cancel` to stop cooperatively.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from ..client import ChatClient
from ..harness import Attempt, Harness
from ..models import Task

Emit = Callable[[str], None]  # one line of (possibly ANSI-coloured) text

RED, GREEN, YELLOW, DIM, RESET = "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[2m", "\x1b[0m"


class Target(BaseModel):
    """One selectable entry in the sidebar."""

    name: str
    description: str = ""
    task: Task | None = None  # None for demo targets


class Runner(Protocol):
    def run(self, target: Target, emit: Emit, cancel: threading.Event) -> bool:
        """Run to completion (or until `cancel` is set); return True on success."""


class DemoRunner:
    """Pretends to be a harness run: sequential coloured lines with a delay."""

    def __init__(self, delay: float = 0.3, steps: int = 6):
        self.delay, self.steps = delay, steps

    def run(self, target: Target, emit: Emit, cancel: threading.Event) -> bool:
        emit(f"{DIM}starting{RESET} {target.name}")
        for i in range(1, self.steps + 1):
            if cancel.wait(self.delay):
                emit(f"{YELLOW}cancelled after step {i - 1}{RESET}")
                return False
            emit(f"step {i}/{self.steps}: {GREEN}ok{RESET}")
        emit(f"{GREEN}done{RESET}")
        return True


class HarnessRunner:
    """Runs a target's `Task` through the real `Harness`, streaming per-attempt progress."""

    def __init__(self, client: ChatClient, max_attempts: int = 3):
        self.harness = Harness(client, max_attempts)

    def run(self, target: Target, emit: Emit, cancel: threading.Event) -> bool:
        if target.task is None:
            emit(f"{RED}target {target.name!r} has no task{RESET}")
            return False
        emit(f"{DIM}running{RESET} {target.name}")

        def on_attempt(n: int, a: Attempt) -> None:
            if a.error:
                emit(f"{YELLOW}attempt {n}: invalid reply{RESET}: {_last_line(a.error)}")
            elif a.check and not a.check.ok:
                emit(f"{RED}attempt {n}: check failed{RESET}: {_last_line(a.check.feedback())}")
            elif a.run and not a.run.ok:
                emit(f"{RED}attempt {n}: failed{RESET}: {_last_line(a.run.feedback())}")
            else:
                emit(f"{GREEN}attempt {n}: ok{RESET}")

        result = self.harness.run(target.task, on_attempt, cancel.is_set)
        if result.ok:
            emit(f"{GREEN}passed{RESET}")
            for line in (result.code or "").splitlines():
                emit(f"  {line}")
        elif cancel.is_set():
            emit(f"{YELLOW}cancelled{RESET}")
        else:
            emit(f"{RED}gave up after {len(result.attempts)} attempts{RESET}")
        return result.ok


def _last_line(text: str) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


def demo_targets() -> list[Target]:
    return [
        Target(name="demo: quick", description="Six fast steps"),
        Target(name="demo: slow", description="Same, to try Stop"),
    ]


def sample_targets() -> list[Target]:
    """Built-in live targets; each pairs a prompt with asserts the code must pass."""
    return [
        Target(
            name="is_prime",
            description="Primality test",
            task=Task(
                prompt="Write a function is_prime(n) that returns True if n is prime.",
                tests="assert is_prime(2) and is_prime(97)\n"
                "assert not is_prime(0) and not is_prime(1) and not is_prime(91) and not is_prime(-7)",
            ),
        ),
        Target(
            name="lru_cache",
            description="LRU cache class",
            task=Task(
                prompt="Implement class LRUCache(capacity) with get(key) returning -1 if "
                "missing and put(key, value), evicting the least recently used item.",
                tests="c = LRUCache(2)\nc.put(1, 1); c.put(2, 2)\nassert c.get(1) == 1\n"
                "c.put(3, 3)\nassert c.get(2) == -1\nc.put(4, 4)\n"
                "assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4",
            ),
        ),
    ]
