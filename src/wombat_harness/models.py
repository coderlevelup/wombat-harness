"""Pydantic models: the task going in, and the structured response coming out."""
from __future__ import annotations

import ast
import re
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator

_FENCE = re.compile(r"^```(?:python|py)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


class Task(BaseModel):
    prompt: str = Field(description="What the code should do.")
    tests: str | None = Field(
        default=None,
        description="Optional Python test code (plain asserts) run after the "
        "generated code, in the same namespace.",
    )
    timeout: float = 10.0
    check: str | None = Field(
        default=None,
        description="Optional shell command (e.g. 'python -m pytest -q') run after the "
        "code passes. The code is written to solution.py in a scratch directory that "
        "is the command's cwd. Supplied by the caller; the model never chooses it.",
    )
    files: dict[str, str] = Field(
        default_factory=dict,
        description="Extra files (relative path -> content) written next to solution.py "
        "for the check command, e.g. a test_solution.py.",
    )
    check_timeout: float = 60.0
    check_cwd: Path | None = Field(
        default=None,
        description="Run the check inside this existing project directory instead of a "
        "scratch dir. Requires check and check_target; files must be empty (the tests "
        "already live in the project).",
    )
    check_target: Path | None = Field(
        default=None,
        description="With check_cwd: where, inside that directory, each candidate is "
        "written for the check. The file is restored (or removed) after every check.",
    )

    @model_validator(mode="after")
    def _check_cwd_consistent(self) -> "Task":
        if self.check_cwd is None and self.check_target is None:
            return self
        if self.check_cwd is None or self.check_target is None or not self.check:
            raise ValueError("check_cwd, check_target and check must be given together")
        if self.files:
            raise ValueError("files cannot be combined with check_cwd")
        cwd, target = self.check_cwd.resolve(), self.check_target.resolve()
        if not cwd.is_dir():
            raise ValueError(f"check_cwd {cwd} is not a directory")
        if cwd not in target.parents:
            raise ValueError(f"check_target {target} must be inside check_cwd {cwd}")
        return self

    @field_validator("files")
    @classmethod
    def _files_stay_inside(cls, v: dict[str, str]) -> dict[str, str]:
        for name in v:
            parts = PurePosixPath(name).parts
            if PurePosixPath(name).is_absolute() or ".." in parts or not parts:
                raise ValueError(f"file path must be relative and inside the workdir: {name!r}")
            if name == "solution.py":
                raise ValueError("solution.py is reserved for the generated code")
        return v


class CodeResponse(BaseModel):
    """Schema the model must fill in. Kept deliberately small for small models."""

    plan: str = Field(description="One or two sentences on the approach.")
    code: str = Field(description="Complete Python source, no markdown fences.")

    @field_validator("code")
    @classmethod
    def _clean_and_check(cls, v: str) -> str:
        v = v.strip()
        if m := _FENCE.match(v):  # small models love adding fences anyway
            v = m.group(1).strip()
        if not v:
            raise ValueError("code is empty")
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"code has a syntax error: {e.msg} (line {e.lineno})") from e
        return v
