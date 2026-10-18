"""What request does each provider get for our action schema? (No network, no API keys.)

Builds OpenAI / Anthropic / Google models from pydantic-ai with a dummy key and a transport that records the
request body and fails, then reports how the provider-specific schema transformer rewrote `Step`.
Findings (2026-10-18, pydantic-ai-slim 2.57): Anthropic gets `anyOf` and per-variant required args, but
`const` (the tool name) and `minLength` are moved into description text, not enforced; OpenAI gets
`strict: false` because the schema isn't strict-compatible; Google keeps `oneOf` and the structure.

    pip install -e '.[pydantic-ai]' anthropic google-genai   # the Anthropic SDK needs httpx2 clients
    python reports/schema_probe_providers.py
"""
"""Capture the exact request each provider would send for NativeOutput(Step), with no network."""
import json, os, sys
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import httpx
from pydantic_ai import Agent, NativeOutput
from wombat_harness.agent.tools import Step

captured = {}
def make_client(name):
    lib = __import__("httpx2") if name == "anthropic" else httpx   # the anthropic SDK wants httpx2
    def handler(request):
        captured[name] = json.loads(request.content or b"{}")
        return lib.Response(500, json={"error": {"message": "probe: stop here"}})
    return lib.AsyncClient(transport=lib.MockTransport(handler))

def models():
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
    yield "openai-chat", OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(api_key="k", http_client=make_client("openai-chat")))
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        yield "anthropic", AnthropicModel("claude-haiku-5-5", provider=AnthropicProvider(api_key="k", http_client=make_client("anthropic")))
    except Exception as e:
        print("anthropic model setup failed:", type(e).__name__, e)
    try:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        yield "google", GoogleModel("gemini-2.5-flash", provider=GoogleProvider(api_key="k", http_client=make_client("google")))
    except Exception as e:
        print("google model setup failed:", type(e).__name__, str(e)[:200])

for name, m in models():
    try:
        Agent(m, output_type=NativeOutput(Step)).run_sync("hi")
    except Exception as e:
        print(f"[{name}] run ended with {type(e).__name__}: {str(e)[:120]}")
    body = captured.get(name)
    if not body:
        print(f"[{name}] no request captured"); continue
    text = json.dumps(body)
    print(f"[{name}] request keys: {sorted(body)}")
    # find the schema wherever the provider put it
    def find(o, path=""):
        if isinstance(o, dict):
            if "$defs" in o or ("properties" in o and "thought" in json.dumps(o)[:4000]):
                return path, o
            for k, v in o.items():
                r = find(v, f"{path}/{k}")
                if r: return r
        elif isinstance(o, list):
            for i, v in enumerate(o):
                r = find(v, f"{path}[{i}]")
                if r: return r
    r = find(body)
    if r:
        path, sch = r; s = json.dumps(sch)
        print(f"   schema at {path}: {len(s)} chars | $defs={'$defs' in s} oneOf={'oneOf' in s} anyOf={'anyOf' in s} "
              f"const={'const' in s} minLength={'minLength' in s} additionalProperties={'additionalProperties' in s} strict={body.get('response_format', {}).get('json_schema', {}).get('strict')}")
        print("   tool-name constants kept:", sorted(set(__import__('re').findall(r'"(?:const|enum)": \[?"(\w+)"', s)))[:10])
    else:
        print("   (schema not found in request)", text[:300])

print("\n=== anthropic schema, as sent ===")
sch = captured["anthropic"]["output_config"]["format"]
print({k: v for k, v in sch.items() if k != "schema"})
print(json.dumps(sch["schema"], indent=1)[:2600])
