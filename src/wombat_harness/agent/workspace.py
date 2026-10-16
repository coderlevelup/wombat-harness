"""The directory the agent may touch. Every path from the model goes through `resolve`."""
from __future__ import annotations

from pathlib import Path


class ToolError(Exception):
    """A failure the model should see (and can react to), not a crash."""


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace {self.root} is not a directory")

    def resolve(self, path: str) -> Path:
        """Absolute path for `path`, which must land inside the workspace.

        Symlinks are followed before the check, so a link pointing outside is rejected.
        """
        p = (self.root / (path or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path {path!r} is outside the workspace")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."
