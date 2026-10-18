# pydantic-ai engines vs the hand-rolled loop, 2026-10-18

Issue #9. Question: should the agent be built on pydantic-ai? Two engines were added behind the same
`AgentPort` and run on the same task with three local models. Single hand-run experiments on one
tiny task, **not a benchmark**.

## Summary

- **`builtin` and `pai-native` were equally reliable: 6 of 6 runs succeeded** (3 models each).
- **`pai-tools`, pydantic-ai's idiomatic agent with native function tools, succeeded in 0 of 7
  recorded runs** on the one model it could use properly (`llama3.1:8b`), and could not run at all
  on the other two: Gemma 3 is refused by Ollama (`400: does not support tools`), and
  `qwen2.5-coder:7b` returned its tool call as plain text.
- pydantic-ai **saved essentially no code** (table below), and `pai-tools` lacks the context trimming
  the builtin engine has.
- Recommendation against the decision rule in #9: **keep `builtin` as the default**, keep `pai-native`
  as an optional engine, don't recommend `pai-tools` for small local models. Details and the
  limits of this evidence are below.

## Setup

| | |
|---|---|
| Engines | `builtin`: own loop + own httpx client, action = pydantic discriminated union with constrained decoding. `pai-native`: the same loop, but pydantic-ai is the model client (`NativeOutput` with the same schema). `pai-tools`: a pydantic-ai `Agent` with native function tools, `requires_approval`/`DeferredToolRequests` for approvals, `UsageLimits` for the step cap |
| Reused by all | `Workspace`, the tool implementations and diff previews, the approval policies, the `Event` stream, the TUI |
| Task | `calc.py` has `add(a, b)` returning `a - b`; "The test in test_calc.py is failing. Find the bug in calc.py and fix it, then run the tests." |
| Success | judged **independently of the agent**: `calc.py` passes `pytest` when the driver runs it, **and** the agent produced a reply. The agent's own claim is not used |
| Run | mode `yolo` (no approvals), max 12 steps, temperature 0.2, each model warmed before timing |
| Server | Ollama 0.40.5, CPU only, 4 vCPU |
| Versions | pydantic-ai-slim 2.57.0, pydantic 2.13.5 |
| Driver | `reports/compare_engines.py`; raw rows in `reports/data/*.jsonl` |

## Results

| Engine | `gemma3:4b` | `qwen2.5-coder:7b` | `llama3.1:8b` |
|---|---|---|---|
| `builtin` | pass, 43 s, 3 tool calls | pass, 82 s, 5 | pass, 59 s, 4 |
| `pai-native` | pass, 39 s, 3 | pass, 89 s, 4 | pass, 72 s, 4 |
| `pai-tools` | **cannot run**: Ollama returns 400 (model has no tool support); the engine reports it in one readable line | **fail**: replied `{"name": "list_dir", "arguments": {...}}` as plain text, 0 tool calls | **fail, 0 of 7 runs** (see below) |

One run per cell except where stated. Times are CPU wall-clock and noisy; the two passing engines are
indistinguishable on speed (39 vs 43 s, 89 vs 82 s, 72 vs 59 s).

### How `pai-tools` failed on `llama3.1:8b`

llama3.1 does make real structured tool calls (every run began with a proper `read_file`), so this is not
"the model can't do tools at all". Of 7 recorded runs (1 + 3 with pydantic-ai's default of 1 tool
retry, then 3 with `retries=3` to match the builtin engine's tolerance), plus one ad hoc repro:

- **Prose instead of an edit** (about 3 runs): after reading `calc.py` the model *explains* the fix in
  text, sometimes with tool-call-looking JSON in the text (including an invented `pytest` tool), and
  never calls `edit_file`. That is a valid final answer to pydantic-ai, so the run "completes" with the
  file unchanged.
- **Bad `edit_file` arguments** (about 3 runs): the model calls `edit_file` with arguments that fail
  validation and `Tool 'edit_file' exceeded max retries count` ends the turn, at 1 retry and again at 3.
  In the repro it was passing `content` (a `write_file` argument) instead of `old`/`new`.
- One early run recorded only "model reply was invalid" without detail (the driver now keeps it).

Raising retries from 1 to 3 did not change the outcome (0 of 3).

## Why? (hypothesis, not tested)

Native function calling asks a small model to emit exactly the right call format, and on Ollama
also depends on the model's chat template being able to parse the call back out. These models
slip: they write the call as prose, or put arguments in the wrong places. The builtin and
`pai-native` engines don't ask the model to call a function. They constrain the *output schema*, so
the grammar makes a malformed action impossible, and that failure class disappears. This matches the
earlier schema experiment (`reports/2026-10-16-local-model-results.md`). I did not test it
further, for instance by seeing whether a larger or tool-tuned model closes the gap.

## Code size (non-blank, non-comment lines)

| Hand-rolled | Lines | pydantic-ai version | Lines |
|---|---|---|---|
| `client.py` (httpx client + error mapping) | 82 | `agent_pai/client.py` (adapter + message conversion + error mapping) | 89 |
| `agent/loop.py` (loop, guards, trimming) | 184 | `agent_pai/engine.py` (agent, tools, approval round-trip) | 165 |

No meaningful reduction. The adapter is not smaller than the client it replaces, because message
conversion and error mapping take as much code as the HTTP call did. `pai-tools` is about 10% shorter
than the loop but **does not trim context**, so it is not feature-equivalent, and both reuse the same
`tools.py` (164 lines).

## Problems found along the way

- **pydantic-ai's `run_sync` breaks when called from a different thread each time**, which is how the
  TUI calls it: the second call fails with `asyncio ... is bound to a different event loop` because
  pooled HTTP connections stay bound to the first loop. Both engines now keep one event loop for their
  lifetime. A regression test reproduces the failure on a real local HTTP server and was confirmed to
  fail without the fix; a first version using a mock transport passed without the fix and was
  replaced.
- pydantic-ai prints a **startup banner**, which would corrupt a TUI; disabled with
  `PYDANTIC_AI_NO_BANNER`.
- The error body from the OpenAI SDK is the error object itself (not wrapped in `"error"`); the first
  version printed a raw dict. Fixed and tested for both shapes.

## Against the decision rule in #9

> Adopt pydantic-ai as the default engine only if it is as reliable on the same tasks and replaces a
> meaningful amount of hand-written code.

- `pai-native`: as reliable (6 of 6 against 6 of 6), but replaces no meaningful code. **Not adopted
  as default.** Kept as an optional engine.
- `pai-tools`: not as reliable on these models (0 of 7 where it can run at all), and doesn't trim context.
  **Not adopted.**

A reason to keep `pai-native` anyway: pydantic-ai supports many providers, so the same loop could drive a
hosted model. I have not tested that.

## Caveats

- One task, tiny, with a one-line fix; one run per cell except `pai-tools` on llama. Success rates are
  not estimated. Don't rank `builtin` against `pai-native` from this.
- **Only small local models were tried (4b-8b).** Larger or better tool-tuned models, or hosted
  ones, may make `pai-tools` work well. This result is about small models on Ollama.
- `pai-tools` was not tuned beyond `retries=3`: no prompt changes, no per-tool description work.
- Approvals in `pai-tools` were only exercised against scripted test models and in `yolo` runs, never
  with a real model asking for approval.
- Sampling variance is real (the same `pai-tools` configuration failed in different ways).

## Reproduce

```
pip install -e '.[pydantic-ai]'
ollama pull gemma3:4b qwen2.5-coder:7b llama3.1:8b
python reports/compare_engines.py builtin:llama3.1:8b pai-native:llama3.1:8b pai-tools:llama3.1:8b
wombat-agent "..." --engine pai-native --model gemma3:4b      # or wombat-tui --engine ...
```
