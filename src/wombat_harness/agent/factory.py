"""Pick an engine by name. The pydantic-ai ones are optional and imported lazily."""
from __future__ import annotations

from pathlib import Path

from ..client import OpenAICompatClient
from .loop import Agent, AgentPort, Policy
from .workspace import Workspace

ENGINES = ("builtin", "pai-native", "pai-tools")
from .models import OLLAMA_BASE_URL as DEFAULT_BASE_URL, parse_model_spec, resolve_base_url

DEFAULT_MODEL = "qwen2.5-coder:7b"

HELP = (
    "builtin: own loop + own client (default). "
    "pai-native: own loop, pydantic-ai as the model client (NativeOutput; works without tool support). "
    "pai-tools: pydantic-ai Agent with native function tools and deferred approvals (needs a model with tool calling)."
)


def make_agent(engine: str, model: str, base_url: str | None, workspace: str | Path,
               policy: Policy, max_steps: int) -> AgentPort:
    ws = Workspace(workspace)
    provider, name = parse_model_spec(model)
    if engine == "builtin":
        if provider != "ollama":
            raise SystemExit(f"--engine builtin only talks to Ollama-style servers; use pai-native or pai-tools for {provider}")
        return Agent(OpenAICompatClient(name, resolve_base_url(provider, base_url)), ws, policy, max_steps)
    try:
        from ..agent_pai import PydanticAIAgent, PydanticAIClient
        from ..agent_pai.client import build_model
    except ImportError:
        raise SystemExit(f"--engine {engine} needs pydantic-ai: pip install -e '.[pydantic-ai]'") from None
    if engine == "pai-native":
        return Agent(PydanticAIClient(model, base_url), ws, policy, max_steps)
    if engine == "pai-tools":
        return PydanticAIAgent(build_model(model, base_url), ws, policy, max_steps)
    raise ValueError(f"unknown engine {engine!r}; choose from {', '.join(ENGINES)}")
