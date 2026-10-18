"""Tools the model can call, as pydantic models.

Wire format: each step is a `Step` whose `action` is a discriminated union of these models.
The JSON schema sent for constrained decoding then forces the right arguments for each tool
(measured on a 1.5b model: an all-optional flat schema let it call edit_file with no
arguments, and an all-required flat one made it invent values for every unused field).
"""
from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, Field

from ..sandbox import run_command
from .workspace import ToolError, Workspace

MAX_FILE_BYTES = 1_000_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache"}


def clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _read_text(p) -> str:
    if not p.is_file():
        raise ToolError(f"{p.name} is not a file" if p.exists() else f"{p.name} does not exist")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"{p.name} is larger than {MAX_FILE_BYTES} bytes")
    try:
        return p.read_text()
    except UnicodeDecodeError:
        raise ToolError(f"{p.name} is not a UTF-8 text file") from None


def _diff(old: str, new: str, name: str, limit: int = 80) -> str:
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=2))
    if len(lines) > limit:
        lines = lines[:limit] + [f"... [{len(lines) - limit} more diff lines]"]
    return "\n".join(lines) or "(no changes)"


class _Tool(BaseModel):
    needs_approval: ClassVar[bool] = False

    def summary(self) -> str:
        raise NotImplementedError

    def preview(self, ws: Workspace) -> str | None:
        return None

    def preflight(self, ws: Workspace) -> str | None:
        """Why this call cannot succeed (the same message `run` would give), or None if it might.

        Used to skip asking a person to approve something that is certain to fail.
        """
        return None

    def run(self, ws: Workspace) -> str:
        raise NotImplementedError


def _iter_files(base):
    if base.is_file():
        yield base
        return
    for d, dirs, names in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for n in sorted(names):
            yield Path(d) / n


class ListDir(_Tool):
    tool: Literal["list_dir"]
    path: str = "."

    def summary(self): return f"list_dir {self.path}"

    def run(self, ws):
        p = ws.resolve(self.path)
        if not p.is_dir():
            raise ToolError(f"{self.path} is not a directory")
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir() if c.name != ".git")
        return "\n".join(names[:200]) + (f"\n... [{len(names) - 200} more]" if len(names) > 200 else "") \
            if names else "(empty)"


class ReadFile(_Tool):
    tool: Literal["read_file"]
    path: str = Field(min_length=1, description="File to read, relative to the project.")

    def summary(self): return f"read_file {self.path}"

    def run(self, ws):
        text = _read_text(ws.resolve(self.path))
        numbered = "\n".join(f"{i:4}  {ln}" for i, ln in enumerate(text.splitlines(), 1))
        return clip(numbered or "(empty file)", 6000)


class Grep(_Tool):
    tool: Literal["grep"]
    pattern: str = Field(min_length=1, description="Regular expression.")
    path: str = Field(default=".", description="File or directory to search.")

    def summary(self): return f"grep {self.pattern!r} {self.path}"

    def run(self, ws):
        try:
            rx = re.compile(self.pattern)
        except re.error as e:
            raise ToolError(f"bad regex: {e}") from None
        files = _iter_files(ws.resolve(self.path))
        hits: list[str] = []
        for f in files:
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    continue
                for n, line in enumerate(f.read_text().splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{ws.rel(f)}:{n}: {line.strip()[:200]}")
                        if len(hits) >= 50:
                            return "\n".join(hits) + "\n... [stopped at 50 matches]"
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(hits) or "no matches"


class WriteFile(_Tool):
    tool: Literal["write_file"]
    needs_approval: ClassVar[bool] = True
    path: str = Field(min_length=1, description="File to create or overwrite.")
    content: str = Field(description="The COMPLETE new content of the file.")

    def summary(self): return f"write_file {self.path} ({len(self.content.splitlines())} lines)"

    def preview(self, ws):
        p = ws.resolve(self.path)
        if p.is_file():
            try:
                return _diff(p.read_text(), self.content, self.path)
            except UnicodeDecodeError:
                return "(overwrites a binary file)"
        return _diff("", self.content, self.path)

    def preflight(self, ws):
        try:
            if ws.resolve(self.path).is_dir():
                return f"{self.path} is a directory"
        except ToolError as e:
            return str(e)
        return None

    def run(self, ws):
        p = ws.resolve(self.path)
        if p.is_dir():
            raise ToolError(f"{self.path} is a directory")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.content)
        return f"wrote {self.path} ({len(self.content.splitlines())} lines)"


class EditFile(_Tool):
    tool: Literal["edit_file"]
    needs_approval: ClassVar[bool] = True
    path: str = Field(min_length=1, description="File to edit.")
    old: str = Field(min_length=1, description="Exact existing text; must occur once in the file.")
    new: str = Field(description="Replacement text.")

    def summary(self): return f"edit_file {self.path}"

    def _apply(self, ws) -> tuple[str, str]:
        text = _read_text(ws.resolve(self.path))
        n = text.count(self.old)
        if n == 0:
            raise ToolError(f"`old` text not found in {self.path}; read_file it and copy the exact text")
        if n > 1:
            raise ToolError(f"`old` text matches {n} places in {self.path}; include more surrounding lines")
        return text, text.replace(self.old, self.new, 1)

    def preflight(self, ws):
        try:
            self._apply(ws)
        except ToolError as e:
            return str(e)
        return None

    def preview(self, ws):
        try:
            before, after = self._apply(ws)
        except ToolError as e:
            return f"(this edit will fail: {e})"
        return _diff(before, after, self.path)

    def run(self, ws):
        _, after = self._apply(ws)
        ws.resolve(self.path).write_text(after)
        return f"edited {self.path}"


class Bash(_Tool):
    tool: Literal["bash"]
    needs_approval: ClassVar[bool] = True
    command: str = Field(min_length=1, description="Shell command to run in the project directory.")
    timeout: ClassVar[float] = 60.0

    def summary(self): return f"bash: {self.command}"

    def preview(self, ws): return f"$ {self.command}\n(in {ws.root})"

    def run(self, ws):
        r = run_command(["bash", "-c", self.command], ws.root, self.timeout, max_output=6000)
        out = "\n".join(x for x in (r.stdout, r.stderr) if x.strip())
        head = f"timed out after {self.timeout:.0f}s" if r.timed_out else f"exit code {r.returncode}"
        return f"{head}\n{out}".rstrip()


class Finish(_Tool):
    tool: Literal["finish"]
    message: str = Field(description="Your final answer to the user.")

    def summary(self): return "finish"


ToolCall = Annotated[
    Union[ListDir, ReadFile, Grep, WriteFile, EditFile, Bash, Finish], Field(discriminator="tool")
]


class Step(BaseModel):
    """What the model outputs each step."""

    thought: str = Field(description="One short sentence: what you will do next and why.")
    action: ToolCall
