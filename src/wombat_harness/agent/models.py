"""Model specs (`provider:name`) and per-provider endpoint defaults. Pure: no SDK, no network."""
from __future__ import annotations

PROVIDERS = ("ollama", "anthropic")
OLLAMA_BASE_URL = "http://localhost:11434/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_DEFAULT_URLS = {"ollama": OLLAMA_BASE_URL, "anthropic": ANTHROPIC_BASE_URL}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """`anthropic:claude-haiku-5-5` -> ("anthropic", "claude-haiku-5-5").

    Ollama tags also contain a colon (`qwen2.5-coder:7b`), so only a known provider prefix is split
    off; anything else is a bare Ollama model name.
    """
    provider, sep, rest = spec.partition(":")
    if sep and provider in PROVIDERS:
        spec_provider, name = provider, rest
    else:
        spec_provider, name = "ollama", spec
    if not name:
        raise ValueError(f"model spec {spec!r} has no model name")
    return spec_provider, name


def resolve_base_url(provider: str, base_url: str | None) -> str:
    """An explicit URL wins; otherwise the provider's own default. Never read from the environment."""
    return base_url or _DEFAULT_URLS[provider]
