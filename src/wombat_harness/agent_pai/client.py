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
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart,
)
from pydantic_ai.models import Model

from ..client import LLMError, LLMRequestError

T = TypeVar("T", bound=BaseModel)


def ollama_model(model: str, base_url: str) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(model, provider=OllamaProvider(base_url=base_url))


def describe_model_error(e: Exception, model: str, base_url: str) -> str:
    """One readable line for a failed model call (shared by both pydantic-ai engines)."""
    if isinstance(e, ModelHTTPError):
        body = e.body
        err = body.get("error", body) if isinstance(body, dict) else body  # the openai SDK unwraps "error"
        detail = (err.get("message", "") if isinstance(err, dict) else str(err or "")).strip()
        msg = f"model server returned {e.status_code}" + (f": {detail}" if detail else "")
        model = e.model_name or model
        low = detail.lower()
        if e.status_code == 404 and "not found" in low:
            msg += f" (try: ollama pull {model})"
        elif "does not support tools" in low:
            msg += " (this model cannot do native tool calling; use --engine pai-native or builtin)"
        return msg
    return f"cannot reach the model server at {base_url}: {e}. Is Ollama running?"


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
    """Drop-in `ChatClient`. Defaults point at Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.2,
        timeout: float = 120.0,
        *,
        pai_model: Model | None = None,  # tests inject a FunctionModel here
    ):
        self.model_name, self.base_url = model, base_url
        self._model = pai_model if pai_model is not None else ollama_model(model, base_url)
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
            with self._lock:
                result = self._loop.run_until_complete(
                    agent.run(prompt, message_history=history, instructions=system))
            return result.output
        except UnexpectedModelBehavior as e:  # reply arrived but would not validate: retryable
            raise LLMError(f"model output failed validation: {e}") from None
        except (ModelHTTPError, ModelAPIError) as e:
            raise LLMRequestError(describe_model_error(e, self.model_name, self.base_url)) from None
