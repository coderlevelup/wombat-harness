import pytest

from wombat_harness.agent.models import (
    ANTHROPIC_BASE_URL, OLLAMA_BASE_URL, parse_model_spec, resolve_base_url,
)


@pytest.mark.parametrize("spec,expected", [
    ("anthropic:claude-haiku-5-5", ("anthropic", "claude-haiku-5-5")),
    ("ollama:qwen2.5-coder:7b", ("ollama", "qwen2.5-coder:7b")),
    ("qwen2.5-coder:7b", ("ollama", "qwen2.5-coder:7b")),  # an Ollama tag, not a provider
    ("gemma3", ("ollama", "gemma3")),
    ("mystery:thing", ("ollama", "mystery:thing")),  # unknown prefix stays part of the name
])
def test_parse_model_spec(spec, expected):
    assert parse_model_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "anthropic:", "ollama:"])
def test_parse_model_spec_rejects_empty_name(spec):
    with pytest.raises(ValueError):
        parse_model_spec(spec)


def test_base_url_defaults_are_per_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://container-proxy.invalid")
    assert resolve_base_url("ollama", None) == OLLAMA_BASE_URL
    assert resolve_base_url("anthropic", None) == ANTHROPIC_BASE_URL == "https://api.anthropic.com"


def test_explicit_base_url_wins():
    assert resolve_base_url("anthropic", "http://x.test") == "http://x.test"
    assert resolve_base_url("ollama", "http://y.test/v1") == "http://y.test/v1"
