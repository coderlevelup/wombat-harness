"""Thin client for any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio, vLLM)."""
from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The model's reply was unusable (bad JSON, failed validation). Worth retrying."""


class LLMRequestError(LLMError):
    """The server could not be reached or refused the request. Retrying won't help."""


def _describe_http_error(r: httpx.Response, model: str) -> str:
    """Prefer the server's own message (Ollama: "model 'x' not found") over httpx's generic one."""
    try:
        err = r.json().get("error")
        detail = err.get("message", "") if isinstance(err, dict) else str(err or "")
    except (ValueError, AttributeError):
        detail = r.text.strip()[:300]
    msg = f"model server returned {r.status_code}" + (f": {detail}" if detail else "")
    if r.status_code == 404 and "not found" in detail.lower():
        msg += f" (try: ollama pull {model})"
    return msg


class ChatClient(Protocol):
    def complete(self, messages: list[dict], schema: type[T]) -> T: ...


def extract_json(text: str) -> str:
    """Pull a JSON object out of text that may be wrapped in fences/prose."""
    text = text.strip()
    if m := re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL):
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class OpenAICompatClient:
    """Default points at Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "local",
        temperature: float = 0.2,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model, self.temperature, self.base_url, self.timeout = model, temperature, base_url, timeout
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def complete(self, messages: list[dict], schema: type[T]) -> T:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            # Server-side constrained decoding from the pydantic schema.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        try:
            r = self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException:
            raise LLMRequestError(
                f"no reply from {self.base_url} within {self.timeout:.0f}s (slow model or CPU?)") from None
        except httpx.TransportError as e:
            raise LLMRequestError(
                f"cannot reach the model server at {self.base_url}: {e}. Is Ollama running?") from None
        if r.status_code >= 400:
            raise LLMRequestError(_describe_http_error(r, self.model))
        try:
            content = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError(f"unexpected response from the model server: {e!r}") from None
        try:
            return schema.model_validate_json(extract_json(content))
        except (ValidationError, json.JSONDecodeError) as e:
            raise LLMError(f"model output failed validation: {e}") from e
