# pai-tools on Anthropic (claude-haiku-5-5), 2026-10-18

Issue #9 follow-up. The comparison report found that `pai-tools` (pydantic-ai's native function tools)
failed on every small Ollama model it could run on (`0 of 7` on `llama3.1:8b`). Was that the approach, or the
models? One short live test on a strong tool-caller. **10 runs on one tiny task; not a benchmark.**

## Summary

- **10 of 10 runs succeeded** (judged by re-running pytest ourselves, plus a non-empty reply), with **0 tool-argument
  retries, 0 tool errors, 0 crashes.** So the earlier `pai-tools` failures were about the small models, not the approach.
- **Approvals work against a real hosted model.** In the 3 `ask`-mode runs (auto-approving callback), pydantic-ai's
  deferred-approval round trip asked for exactly the edit and the pytest command, every time, and the run
  continued correctly.
- **Found and fixed a race:** the model batches dependent calls (an edit and the test run) into one response, and
  pydantic-ai runs a batch concurrently by default, so the tests could start before the edit landed. All six tools
  are now `sequential=True`. A deterministic unit test fails without the fix and passes with it.
- This **does not** overturn "small local models do better with schema-constrained output". It adds that on a capable
  tool-caller, `pai-tools` works.

## Setup

| | |
|---|---|
| Engine / model | `pai-tools`, `anthropic:claude-haiku-5-5`, `retries=3` |
| Task | same as the other reports (`calc.py` `add` returns `a - b`; fix it and run the tests) |
| Success | the driver re-runs pytest on the workspace after the agent finishes, and requires a reply |
| Runs | 3 batches: 4 (3 `yolo` + 1 `ask`), 4 (3 + 1), 2 (1 + 1); `max_steps` 12 |
| Safeguards | key only from `WOMBAT_ANTHROPIC_API_KEY` (`ANTHROPIC_API_KEY` unset); the model's base URL is asserted to be `https://api.anthropic.com` before every run, even though `ANTHROPIC_BASE_URL` is set in this container for other purposes |
| Driver / data | `reports/live_anthropic_tools.py`; `reports/data/live_anthropic_tools_run{1,2,3_sequential}.jsonl` |

## Results

| | |
|---|---|
| Success | 10 / 10 (7 `yolo`, 3 `ask`) |
| Tool-argument retries / tool errors / crashes | 0 / 0 / 0 |
| Tool calls per run | 4 (`read_file calc.py`, `read_file test_calc.py`, `edit_file`, `bash pytest`) |
| API calls per run | 3 or 4 (the model batches several tool calls per response) |
| Wall time per run | 2.8 to 3.7 s |
| Approvals asked in the 3 `ask` runs | `edit_file calc.py` then `bash: python -m pytest -q`, every time |
| Tokens (all 10 runs) | 41,916 in / 3,493 out, about $0.06 at an assumed $1 / $5 per million tokens (**price not verified**) |

## The race

Haiku usually sends `edit_file` and `bash pytest` in the **same response**. pydantic-ai executes the tool calls of one
response concurrently unless a tool is marked sequential, and our tools are synchronous functions run in threads. So the
test run can start, or even finish, before the edit is written.

- **Observed:** in the first batch, 2 of the 4 runs show `bash: pytest` *starting before* `edit_file` in the event
  order. Their final state was still correct (our pytest passed), but that batch did not record the tool outputs, so I
  cannot say what the in-run pytest reported. A second batch (outputs recorded, all `edit` before `bash`) showed correct
  `2 passed` results.
- **Confirmed deterministically:** `test_dependent_tool_calls_in_one_response_run_in_order` runs a slow edit alongside a
  fast `bash`; without the fix the order is `['bash_ran', 'edit_done']`, with it `['edit_done', 'bash_ran']`.
- **Fix:** `@agent.tool(sequential=True)` on all six tools, not only the mutating ones, so a read placed after an
  edit in the same batch also sees the edited file. This is the same one-action-at-a-time ordering the builtin engine
  has by construction. A last batch of 2 runs after the fix behaved correctly (edit, then pytest, 2 passed).

The race existed with any model that batches calls; it simply hadn't surfaced in the small-model runs because those
models never got that far.

## Other

- pydantic-ai warns `Sampling parameters ['temperature'] are not supported by 'claude-haiku-5-5'` on every call and
  drops it, as the `pai-native` Anthropic report found. Not handled yet.
- Unlike the small models, Haiku always called a tool for the edit and never answered in prose.

## Caveats

- **10 runs, one tiny task, one model.** Identical behaviour 10 times says failures are probably not common here, not that
  they are rare. No rates or intervals are claimed. Don't use this to rank `pai-tools` against `pai-native` or `builtin`.
- Haiku is a strong tool-caller; this says nothing about mid-sized models between 8b and Haiku.
- The race fix is verified by the deterministic unit test and 2 live runs, not by a long soak.
- `ask` mode used an auto-approving callback; nobody was at a terminal and no approval was ever denied in these runs
  (denial is covered by the scripted unit tests only).
- Larger tasks, longer histories and context trimming (`pai-tools` has none) were not tested.
- Cost figures assume prices I did not check.

## Reproduce

```
pip install -e '.[pydantic-ai]'
WOMBAT_ANTHROPIC_API_KEY=... python reports/live_anthropic_tools.py 3
```
