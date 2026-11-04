"""TDD as a game state the harness holds, not something the model is asked to remember.

RED       write a failing test. Only test files may be edited; production code is locked.
GREEN     make the failing test pass. Test files are locked, so the test can't be edited to pass.
REFACTOR  tidy up with the tests green. Anything may be edited; a red check is flagged.

The harness decides the phase from the project's check command (the tests): failing at the start of a turn means GREEN
(there is a test to satisfy), passing means RED (there is nothing to satisfy yet), unless the person says otherwise.
Moves happen after an edit, from the check result: RED to GREEN when the new test fails, GREEN to REFACTOR when it passes.
"""
from __future__ import annotations

from enum import Enum

from .tools import EditFile, WriteFile, is_test_path


class Phase(str, Enum):
    RED = "red"
    GREEN = "green"
    REFACTOR = "refactor"


GUIDE = {
    Phase.RED: "RED: write a failing test for the change. You can only edit test files; production code is locked until a test fails.",
    Phase.GREEN: "GREEN: make the failing test pass by changing the code. Test files are locked: do not edit them.",
    Phase.REFACTOR: "REFACTOR: the tests pass. Tidy up if it helps, keep them passing, then call finish.",
}


def start_phase(check_passed: bool) -> Phase:
    return Phase.RED if check_passed else Phase.GREEN


def refusal(phase: Phase, call) -> str | None:
    """Why this call isn't allowed in this phase, or None. Only file edits are policed (bash can't be)."""
    if not isinstance(call, (EditFile, WriteFile)):
        return None
    test = is_test_path(call.path)
    if phase is Phase.RED and not test:
        return f"Refused. {GUIDE[Phase.RED]} `{call.path}` is not a test file."
    if phase is Phase.GREEN and test:
        return f"Refused. {GUIDE[Phase.GREEN]} `{call.path}` is a test file; change the code under test instead."
    return None


def finish_refusal(phase: Phase) -> str | None:
    if phase is Phase.RED:
        return f"Not finished. {GUIDE[Phase.RED]}"
    if phase is Phase.GREEN:
        return f"Not finished: the tests still fail. {GUIDE[Phase.GREEN]}"
    return None


def advance(phase: Phase, call, check_passed: bool) -> tuple[Phase, str]:
    """The phase after `call` (an edit that ran) and the check result, with a note for the model ('' if none)."""
    if not isinstance(call, (EditFile, WriteFile)):
        return phase, ""
    test = is_test_path(call.path)
    if phase is Phase.RED and test:
        if check_passed:
            return phase, "That test passes already, so it doesn't describe a change yet. Make it fail first."
        return Phase.GREEN, f"The test fails, as it should. Now {GUIDE[Phase.GREEN]}"
    if phase is Phase.GREEN and not test and check_passed:
        return Phase.REFACTOR, f"The tests pass. {GUIDE[Phase.REFACTOR]}"
    if phase is Phase.REFACTOR and not check_passed:
        return phase, "The tests fail after that edit. Fix it, or undo it."
    return phase, ""
