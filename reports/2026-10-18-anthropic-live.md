# pai-native on Anthropic (claude-haiku-5-5), live, 2026-10-18

Issue #9 follow-up. Question: the offline probe (`reports/schema_probe_providers.py`) showed pydantic-ai
rewrites our `Step` schema for Anthropic, moving the tool-name `const` and `minLength` into description
text instead of enforcing them. Does Haiku follow the hint? One small live run, **n=3, not a benchmark**.

## Summary

- **3 of 3 runs succeeded, with 0 invalid replies.** No wrong tool names, no missing arguments, no retries.
- Every run took the same path: `read_file calc.py`, `edit_file calc.py`, `bash: python -m pytest -q`,
  then a plain-text reply. 4 API calls each, 5 to 8 s wall time.
- So on this task Haiku followed the schema as rewritten. That says nothing yet about harder tasks.

## Setup

| | |
|---|---|
| Engine / model | `pai-native`, `anthropic:claude-haiku-5-5`, via `make_agent("pai-native", ...)` |
| Task | same as `compare_engines.py` (`calc.py` has `add` returning `a - b`; fix it and run the tests) |
| Success | judged by the driver re-running pytest on the workspace, plus a non-empty reply; not the agent's claim |
| Run | mode `yolo`, max 12 steps, 3 runs, `invalid_replies` read from the client |
| Endpoint | asserted to be `https://api.anthropic.com` (both the client's setting and the SDK client's base URL) before any call; key only from `WOMBAT_ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL` unset |
| Versions | pydantic-ai-slim 2.57.0 |
| Driver / data | `reports/live_anthropic.py`, `reports/data/live_anthropic.jsonl` |

## Results

| Run | Success | Tool calls | API calls | Invalid replies | Wall time | Input / output tokens |
|---|---|---|---|---|---|---|
| 1 | yes | 3 | 4 | 0 | 8.2 s | 8,989 / 303 |
| 2 | yes | 3 | 4 | 0 | 5.2 s | 9,012 / 309 |
| 3 | yes | 3 | 4 | 0 | 5.3 s | 8,987 / 294 |

No errors, no crashes. No prompt caching was used (cache reads 0).

## The schema question

The probe predicted a risk: for Anthropic the tool name is a free string with a description hint, so
a model could send an unknown tool name or omit `old`/`new`. We counted this through `invalid_replies`
(pydantic-ai's output retries) and logged the validation message for any retry. **There were none to
classify**, so I cannot say what kind of mistakes Haiku makes here, only that it made none in 12 calls.

Reminder of the limit: pydantic-ai's default is 1 output retry, so a second invalid reply in one step
would have failed that step. That did not happen.

## Other findings

- **`temperature` is ignored for this model.** pydantic-ai warns `Sampling parameters ['temperature'] are
  not supported by 'claude-haiku-5-5'` and drops it. Our client always sends 0.2, so the setting does
  nothing here. Harmless for the run, but the warning is printed on every call.

## Cost (rough)

About 27,000 input and 900 output tokens across the 3 runs. At an assumed $1 per million input and $5 per
million output tokens (**I did not verify the price of this model**; check current pricing), that is about
$0.03 in total, about a cent per run. Treat as an order of magnitude.

## Caveats

- n=3 on one tiny task with a one-line fix; identical behaviour three times says the failure rate is
  probably not high, not that it is low. No confidence interval is claimed.
- I did not test *whether the hint is what made it work*. The system prompt also lists the tools, and
  Haiku may succeed regardless of the schema rewrite. A control would swap in a weaker prompt or
  inspect the raw requests; not done.
- Only `pai-native`. `pai-tools` and `builtin` were not run on Anthropic (`builtin` cannot).
- Only Haiku. Larger tasks, longer histories (context trimming), multi-file edits, and approval flows
  with a hosted model were **not verified**.
- Token counts come from a wrapper around `AnthropicModel.request` in the driver, not from the engine.
- Wall times include network latency from the cloud container.

## Reproduce

```
pip install -e '.[pydantic-ai]'
WOMBAT_ANTHROPIC_API_KEY=... python reports/live_anthropic.py 3
wombat-agent "..." --engine pai-native --model anthropic:claude-haiku-5-5
```
