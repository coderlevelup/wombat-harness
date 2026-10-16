from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .client import OpenAICompatClient
from .harness import Harness
from .models import Task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wombat", description="Generate Python with a local model.")
    ap.add_argument("prompt", help="What to build")
    ap.add_argument("--tests", type=Path, help="File of assert-style tests the code must pass")
    ap.add_argument("--check", help="Shell command run after the code passes, e.g. 'python -m pytest -q'. "
                    "Runs in a scratch dir with the code as solution.py (no shell; args split like a shell would)")
    ap.add_argument("--file", type=Path, action="append", default=[], metavar="PATH",
                    help="File copied next to solution.py for --check (repeatable), e.g. test_solution.py")
    ap.add_argument("--check-cwd", type=Path, metavar="DIR",
                    help="Run --check inside this project directory instead of a scratch dir. "
                    "Each candidate is written to --out (which must be inside DIR) for the check "
                    "and the file is restored afterwards; the final result is then saved to --out")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, help="Write the final code here (successful runs only)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files with --out")
    ap.add_argument("--write-tests", action="store_true", help="Also write test_<name>.py next to --out")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if (a.force or a.write_tests) and not a.out:
        ap.error("--force and --write-tests require --out")
    if a.write_tests and not a.tests:
        ap.error("--write-tests requires --tests")

    if a.check_cwd:
        if not (a.check and a.out):
            ap.error("--check-cwd requires --check and --out")
        if a.file:
            ap.error("--file cannot be combined with --check-cwd")
        if a.out.exists() and not a.force:  # fail now, not after minutes of generation
            ap.error(f"{a.out} already exists (pass --force to overwrite)")

    try:
        task = Task(
            prompt=a.prompt,
            tests=a.tests.read_text() if a.tests else None,
            check=a.check,
            files={f.name: f.read_text() for f in a.file},
            check_cwd=a.check_cwd,
            check_target=a.out if a.check_cwd else None,
        )
    except ValidationError as e:
        ap.error("; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors()))
    res = Harness(OpenAICompatClient(a.model, a.base_url), a.attempts).run(task)
    if a.verbose:
        for i, at in enumerate(res.attempts, 1):
            if at.check and not at.check.ok:
                status = "check failed: " + (at.check.feedback().strip().splitlines() or ["(no output)"])[-1]
            elif at.error:
                status = f"invalid reply: {at.error}"
            elif at.run and at.run.ok:
                status = "ok"
            else:
                status = at.run.feedback().strip().splitlines()[-1] if at.run else "no run"
            print(f"[attempt {i}] {status}", file=sys.stderr)
    if not res.ok and res.attempts and not res.code and res.attempts[-1].error:
        print(f"error: {res.attempts[-1].error}", file=sys.stderr)
    if res.code:
        print(res.code)
    if a.out and res.ok:
        try:
            for p in res.save(a.out, force=a.force, write_tests=a.write_tests):
                print(f"wrote {p}", file=sys.stderr)
        except FileExistsError as e:
            print(f"error: {e} (pass --force to overwrite)", file=sys.stderr)
            return 2
        except (ValueError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
