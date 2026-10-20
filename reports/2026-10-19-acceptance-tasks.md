# Acceptance tasks, first runs, 2026-10-19

The five tasks in `wombat_harness.tasks` (see `ROADMAP.md`, #18), run once each. One run per cell, so read these as "did / didn't this time", not rates.
"Done" means the visible pytest passes, a hidden check passes, protected test files are unmodified, and the agent replied.
Hardware for the local models: 4 vCPU CPU only, so times are slow. Raw results: `data/2026-10-19-tasks-haiku-pai-tools.jsonl`, `data/2026-10-19-tasks-local.jsonl`.

| Task | Haiku 5.5, `pai-tools` | `qwen2.5-coder:7b`, `builtin` | `gemma3:4b`, `builtin` |
|---|---|---|---|
| fix-add | done (4 s) | done on rerun (133 s)* | done (61 s) |
| fix-across-files | done (6 s) | done (232 s) | done (113 s) |
| add-feature | done (6 s) | done (370 s) | **not done** (117 s) |
| rename-function | done (6 s) | done (240 s) | **not done** (226 s) |
| unfamiliar-package | done (7 s) | **not done** (360 s) | **not done** (344 s) |

\* The first `qwen2.5-coder:7b` run of fix-add hit the client's 120 s no-reply timeout with 0 tool calls (a slow first request on a CPU shared with the other model), so it is not a result for the model. Rerun alone: done. The timeout is not exposed on the CLI.

## What the failures show
(From the recorded tool-call lists; I did not read the model's full replies.)

- **`gemma3:4b`, add-feature:** the visible tests passed but the hidden check failed. The hidden check is what separates "tests green" from "did the task"; the visible tests alone would have called it a pass. I have not checked which requirement it missed.
- **`gemma3:4b`, rename-function:** 15 tool calls alternating `edit_file` and `grep` on the same two files, then stopped by the repeat guard ("model is repeating itself"). It never finished the rename.
- **`gemma3:4b`, unfamiliar-package:** it went looking for `inventory.py` (a file that doesn't exist; the code is a package), wrote a new `inventory.py`, then edited `tests/test_inventory.py` twice. The protected-file check caught that. Ran out of the 15 steps.
- **`qwen2.5-coder:7b`, unfamiliar-package:** ran pytest, then edited `tests/test_inventory.py` four times (each followed by pytest) before it listed the package directory and read `inventory/loader.py`; one edit to the loader came right at the step limit. Same pattern as the gemma run: the failure was in the test's expectation, and the model changed the test rather than look at the code.

## What this suggests for §2
These are hypotheses from five tasks and two small models:
- Editing the test instead of the code showed up again in both small models on the one task whose bug isn't in the file named by the test. "Run the failing test first and show the failure, with the files in the traceback" (#19) is the first affordance to try: the traceback would have pointed at `inventory/loader.py`.
- Repeated `edit_file`/`grep` loops on a multi-file rename: a rename-style tool, or a guard that names the loop, are candidates.
- The capable model finished everything in seconds, so these five tasks do not separate capable models from each other. They do separate the small ones.

## Experiment: `--check-first` (show the test output before the first step) did not help

Hypothesis from the failures above: if the harness runs the project's tests and puts the output in front of the model, it will start from the real failure and stop editing tests. The flag is `--check-first` (builtin and pai-native engines). Same tasks, one run per cell, data in `data/2026-10-19-tasks-local-check-first.jsonl`.

| Task | gemma3:4b without | with | qwen2.5-coder:7b without | with |
|---|---|---|---|---|
| fix-add | done | done | done | done* |
| fix-across-files | done | not done | done | not done |
| add-feature | not done | not done | done | not done |
| rename-function | not done | not done | done | done |
| unfamiliar-package | not done | not done | not done | not done |
| **done** | **2 of 5** | **1 of 5** | **4 of 5** | **2 of 5** |

\* First attempt died with an Ollama 500 (`signal: killed`; I had two 4b/7b models warm at once, probably memory), rerun alone passed. Not counted as a model failure.

- Not supported: no improvement on either model, and both did worse on `fix-across-files`, where the failing output names `pricing.py` yet gemma edited `checkout.py` and then edited the protected `test_checkout.py` five times. Both models still edited tests on `unfamiliar-package`.
- n = 1 per cell, so a drop from 4 to 2 could be noise. I can't say it hurts; I can say there is no sign it helps.
- What I haven't ruled out: that the way the output is worded or its length (pytest output in a small context) is the problem, rather than showing the failure at all. I have not read the models' replies to see how they used it.
- Decision: the flag stays, off by default. Not recommended on the strength of this run. Next candidates in #19 are different in kind: stopping the test edit itself (e.g. refusing or flagging an edit to a test file before the model has touched the code) rather than hoping the model reads better.
