# Local-model results, 2026-10-16

What happened when the harness and the coding agent were run against real small models. These are
single hand-run experiments, **not a benchmark**: one run per configuration unless stated, so read
them as "this worked / this failed once", not as rates.

## Setup

| | |
|---|---|
| Server | Ollama 0.40.5, OpenAI-compatible `/v1` endpoint |
| Hardware | 4 vCPU, 15 GB RAM, **CPU only** (cloud container) |
| Models | `qwen2.5-coder` 0.5b, 1.5b, 7b (default Ollama quantisation) |
| Sampling | temperature 0.2 (the harness default) |
| Structured output | pydantic JSON schema sent as `response_format: json_schema` |

Timings include other processes sharing the same 4 CPUs (tests, a second model download), so treat
them as rough.

## 1. One-shot harness (generate, verify, retry)

`wombat "<prompt>" --tests ...` / `--check ...`: the model returns `{plan, code}`, which is
validated, run, and retried with the error fed back.

| Task | Model | Result |
|---|---|---|
| `is_prime(n)`, asserts incl. 0, 1, 91, -7 | 1.5b | pass, attempt 1 (~21 s) |
| `LRUCache` with eviction | 1.5b | pass, attempt 1 (~15 s) |
| `is_prime` via `--check "python -m pytest -q"` (plain, plus `--out`, plus `--check-cwd`) | 1.5b | 5 runs: first **failed** (all 3 replies rejected as syntax errors, see notes); the next 4 passed on attempt 1 |
| Recursive-descent expression evaluator, 5 asserts, 4 attempts | 0.5b | **fail**, 4/4 attempts: split the string on spaces, never parsed |
| same | 1.5b | **fail**: attempt 1 rejected by the syntax check, attempt 2 `NameError`, attempts 3-4 `TypeError: 'str' object is not callable` |

Notes:

- The retry loop and the pydantic syntax rejection both ran as designed on a real model; neither
  model converged on the harder task. A traceback in the prompt did not help models this small.
- Sampling variance is real: one `is_prime` run had all 3 replies rejected as syntax errors
  (newlines collapsed to spaces, identical each retry), while resampling the same prompt gave
  clean code. "invalid syntax (line 1)" is not useful feedback for a 1.5b model.

## 2. Which wire schema? (1.5b, agent prompt, 6 samples each)

Situation: the model has just read `calc.py` (`add` returns `a - b`); the next step should be an edit.

| Schema | What the model emitted |
|---|---|
| **Flat, all fields optional** (first version) | In the full agent run it called `edit_file` with **no arguments** on every step, because a reply with just `thought` and `tool` is valid. See section 3. |
| **Flat, all fields required** | 6/6 filled *every* field with invented text (`"content": "... (output truncated) ..."`, `"command": "bash ./test_calc.py"`) and chose `read_file` again instead of editing. Unusable. |
| **Discriminated union** (`action` is one of the tool models) | 6/6 clean, tool-specific objects: `{"thought": ..., "action": {"tool": "read_file", "path": "test_calc.py"}}` |

The union schema is what the agent uses. Ollama honours `oneOf`/discriminated unions in
constrained decoding, and each tool's required arguments are then enforced by the grammar.
(The model still chose `read_file` over `edit_file` in those samples: schema fixes structure, not
judgement.) Script: `reports/schema_experiment.py` (recorded with an earlier wording of the system
prompt; exact outputs will differ).

## 3. Coding agent on a tiny repo

Task, identical each time: *"The test in test_calc.py is failing. Find the bug in calc.py and fix it,
then run the tests."* `add(a, b)` returns `a - b`; `mul` is correct.

| Run | Model / schema | Outcome | Time |
|---|---|---|---|
| A | 1.5b, flat optional | **Failed.** Called `edit_file` with no arguments 25 times in a row until the step limit. File untouched. (This also exposed that the loop had no give-up guard for repeated bad calls.) | 3 m 14 s |
| B | 1.5b, union | **Found and fixed the bug** (`a - b` to `a + b`) on the first edit, then failed to verify: ran `bash test_calc.py` instead of pytest, mis-diagnosed the shell errors, and looped until the 10-step cap. | 1 m 58 s |
| C | 7b, union, headless `wombat-agent --mode yolo` | **Success.** read test, ran `python -m pytest -q` (saw `assert -1 == 3`), read `calc.py`, edited, re-ran tests (`2 passed`), finished with a correct summary. | 2 m 55 s |
| D | 7b, union, **chat TUI** (`ChatApp`, mode `ask`, approvals answered `y` by a script) | **Success.** Same sequence; 2 approvals (the edit with a diff preview, the pytest command); UI returned to ready. | 2 m 52 s |

After run B the system prompt got one line saying to run Python tests with `python -m pytest -q`
and never to run a `.py` file with `bash`. Runs C and D used that prompt; B did not, so B and C are
not a clean model-only comparison.

## What this suggests

- **Use 7b or larger for the agent.** 1.5b can sometimes locate and fix a one-line bug but cannot
  reliably drive the verify step, and loops. 0.5b was only tried in the one-shot harness, where it
  failed the harder task.
- **Constrain the output with a per-tool schema (union).** It was the difference between the model
  being unable to produce a valid edit and producing one.
- **Guard loops.** Identical repeated calls and repeated invalid replies are cut off after a few
  tries; before that guard, run A spent 25 steps doing nothing.
- **Speed is the limit on CPU.** About 3 minutes for a six-step task on 7b. A GPU or Apple silicon
  would change that substantially; I have not measured either.

## Caveats

- n = 1 per configuration; one tiny task; one model family. Do not read rates into any of this.
- Ollama's default context window is small and truncates silently. The agent keeps prompts under
  ~16k characters by trimming, but `OLLAMA_CONTEXT_LENGTH` was not raised for these runs. Raising it
  is recommended for anything longer.
- The TUI run (D) was driven by a script pressing keys through Textual's test harness, not by a
  person at a terminal.

## Reproduce

```
ollama pull qwen2.5-coder:7b
printf 'def add(a, b):\n    return a - b\n' > proj/calc.py   # plus a test_calc.py with add(1,2)==3
wombat-agent "The test in test_calc.py is failing. Find the bug in calc.py and fix it, then run the tests." \
    --workspace proj --model qwen2.5-coder:7b --mode yolo
wombat-tui --workspace proj --model qwen2.5-coder:7b          # interactive
python reports/schema_experiment.py                            # schema comparison (1.5b)
```
