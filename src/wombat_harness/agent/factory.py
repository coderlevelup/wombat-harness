"""Pick an engine by name. The pydantic-ai ones are optional and imported lazily."""
from __future__ import annotations

from pathlib import Path

from ..client import OpenAICompatClient
from .loop import Agent, AgentPort, Policy
from .workspace import Workspace

ENGINES = ("builtin", "pai-native", "pai-tools")
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

HELP = (
    "builtin: own loop + own client (default). "
    "pai-native: own loop, pydantic-ai as the model client (NativeOutput; works without tool support). "
    "pai-tools: pydantic-ai Agent with native function tools and deferred approvals (needs a model with tool calling)."
)


def make_agent(engine: str, model: str, base_url: str, workspace: str | Path,
               policy: Policy, max_steps: int) -> AgentPort:
    ws = Workspace(workspace)
    if engine == "builtin":
        return Agent(OpenAICompatClient(model, base_url), ws, policy, max_steps)
    try:
        from ..agent_pai import PydanticAIAgent, PydanticAIClient
        from ..agent_pai.client import ollama_model
    except ImportError:
        raise SystemExit(f"--engine {engine} needs pydantic-ai: pip install -e '.[pydantic-ai]'") from None
    if engine == "pai-native":
        return Agent(PydanticAIClient(model, base_url), ws, policy, max_steps)
    if engine == "pai-tools":
        return PydanticAIAgent(ollama_model(model, base_url), ws, policy, max_steps)
    raise ValueError(f"unknown engine {engine!r}; choose from {', '.join(ENGINES)}")
