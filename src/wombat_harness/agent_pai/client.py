"""A `ChatClient` backed by pydantic-ai, replacing the hand-written httpx client.

Same contract as `OpenAICompatClient.complete(messages, schema)`: the schema is sent as a
structured-output request (`NativeOutput`) and the reply is validated into it. Because the
schema rides on the response format rather than tool calling, this also works with models
Ollama will not give tools to (e.g. Gemma 3).
"""
from __future__ import annotations

import asyncio
import threading
from typing import TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, RetryPromptPart, SystemPromptPart, TextPart, UserPromptPart,
)
from pydantic_ai.models import Model

from ..agent.models import parse_model_spec, resolve_base_url
from ..client import LLMError, LLMRequestError

T = TypeVar("T", bound=BaseModel)


def ollama_model(model: str, base_url: str) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(model, provider=OllamaProvider(base_url=base_url))


def build_model(spec: str, base_url: str | None = None) -> Model:
    """`provider:name` (or a bare Ollama tag) -> a pydantic-ai model.

    Anthropic credentials come only from WOMBAT_ANTHROPIC_API_KEY, and the endpoint is passed
    explicitly: the SDK would otherwise pick up ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY, which can
    belong to some other tool in the same environment.
    """
    provider, name = parse_model_spec(spec)
    url = resolve_base_url(provider, base_url)
    if provider == "anthropic":
        import os

        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        key = os.environ.get("WOMBAT_ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("anthropic models need the WOMBAT_ANTHROPIC_API_KEY environment variable")
        return AnthropicModel(name, provider=AnthropicProvider(api_key=key, base_url=url))
    return ollama_model(name, url)


def describe_model_error(e: Exception, model: str, base_url: str, provider: str = "ollama") -> str:
    """One readable line for a failed model call (shared by both pydantic-ai engines)."""
    if isinstance(e, ModelHTTPError):
        body = e.body
        err = body.get("error", body) if isinstance(body, dict) else body  # the openai SDK unwraps "error"
        detail = (err.get("message", "") if isinstance(err, dict) else str(err or "")).strip()
        msg = f"model server returned {e.status_code}" + (f": {detail}" if detail else "")
        model = e.model_name or model
        low = detail.lower()
        if e.status_code == 404 and "not found" in low and provider == "ollama":
            msg += f" (try: ollama pull {model})"
        elif "does not support tools" in low:
            msg += " (this model cannot do native tool calling; use --engine pai-native or builtin)"
        return msg
    hint = " Is Ollama running?" if provider == "ollama" else ""
    return f"cannot reach the model server at {base_url}: {e}.{hint}"


def to_messages(messages: list[dict]) -> tuple[str | None, list[ModelMessage], str]:
    """Chat-style dicts -> (system prompt, pydantic-ai history, final user prompt).

    The system prompt goes through pydantic-ai's per-run `instructions`. Consecutive user
    messages are merged and the last user message becomes the prompt.
    """
    system: str | None = None
    turns: list[tuple[str, str]] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            system = content
        elif turns and turns[-1][0] == role == "user":
            turns[-1] = ("user", turns[-1][1] + "\n\n" + content)
        else:
            turns.append((role, content))
    if not turns or turns[-1][0] != "user":
        raise ValueError("the last message must be from the user")
    prompt = turns.pop()[1]
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(c)]) if r == "user" else ModelResponse(parts=[TextPart(c)])
        for r, c in turns
    ]
    return system, history, prompt


class PydanticAIClient:
    """Drop-in `ChatClient`. `model` is `provider:name` or a bare Ollama tag; `base_url=None` means
    that provider's own default (Ollama's OpenAI-compatible endpoint, or api.anthropic.com)."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: float = 120.0,
        *,
        pai_model: Model | None = None,  # tests inject a FunctionModel here
    ):
        self.provider, self.model_name = parse_model_spec(model)
        self.base_url = resolve_base_url(self.provider, base_url)
        self._model = pai_model if pai_model is not None else build_model(model, base_url)
        self.invalid_replies = 0  # replies pydantic-ai had to send back for another try
        self._settings = {"temperature": temperature, "timeout": timeout}
        # One event loop for this client's lifetime. `run_sync` makes a fresh loop per call, but the
        # HTTP client's pooled connections and locks are bound to the loop that first used them, so a
        # second call (e.g. from the next TUI worker thread) fails with "bound to a different event loop".
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()

    def complete(self, messages: list[dict], schema: type[T]) -> T:
        system, history, prompt = to_messages(messages)
        agent = Agent(self._model, output_type=NativeOutput(schema), model_settings=self._settings)
        try:
            with self._lock, capture_run_messages() as sent:
                try:
                    result = self._loop.run_until_complete(
                        agent.run(prompt, message_history=history, instructions=system))
                finally:  # counted even when the run ends in a validation failure
                    self.invalid_replies += sum(
                        isinstance(p, RetryPromptPart) for m in sent for p in m.parts)
            return result.output
        except UnexpectedModelBehavior as e:  # reply arrived but would not validate: retryable
            raise LLMError(f"model output failed validation: {e}") from None
        except (ModelHTTPError, ModelAPIError) as e:
            raise LLMRequestError(describe_model_error(e, self.model_name, self.base_url, self.provider)) from None
