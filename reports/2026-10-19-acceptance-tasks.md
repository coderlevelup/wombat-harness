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
