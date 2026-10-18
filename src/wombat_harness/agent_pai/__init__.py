"""Engines built on pydantic-ai (optional: pip install 'wombat-harness[pydantic-ai]').

pydantic-ai prints a startup banner to the terminal, which would corrupt a TUI. Disable it
before anything imports pydantic_ai.
"""
import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from .client import PydanticAIClient  # noqa: E402
from .engine import PydanticAIAgent  # noqa: E402

__all__ = ["PydanticAIAgent", "PydanticAIClient"]
