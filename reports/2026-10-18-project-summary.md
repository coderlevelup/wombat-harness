# wombat-harness: where things stand, 2026-10-18

One page that pulls the other reports together: what exists, what the evidence supports, what we still
don't know, and what is open. Every claim points at the report or data behind it. The experiments behind it
are small hand-run tests on one tiny task, so **none of this is a benchmark**.

## 1. What exists

A coding agent for small local models (and, optionally, hosted ones): you chat, the model reads, searches and edits
files and runs commands in a workspace directory, and you approve the risky steps.

| Piece | What it is |
|---|---|
| Engine (`wombat_harness/agent`) | tool loop: `list_dir`, `read_file`, `grep`, `write_file`, `edit_file`, `bash`, `finish`; workspace confinement (paths and symlinks); `ask` / `edits` / `yolo` approval policies with diff previews; guards for repeats, invalid replies and step limits; context trimming |
| Wire format | each step is `{thought, action}` where `action` is a pydantic discriminated union, sent as a JSON schema for constrained decoding |
| Front ends | `wombat-agent` (headless, y/n/a on stdin) and `wombat-tui` (Textual chat UI, inline approval, Esc to interrupt), both over one `AgentPort` protocol |
| Engines | `builtin` (default), `pai-native` (pydantic-ai as the model client), `pai-tools` (pydantic-ai `Agent` with native function tools); `--engine` selects |
| Providers | `--model provider:name`: `ollama` (default) and `anthropic` (key only from `WOMBAT_ANTHROPIC_API_KEY`, endpoint passed explicitly) |
| One-shot generator | `wombat "<prompt>"`: generate, validate, run with asserts, retry; `--check` / `--check-cwd` run a project command as the final gate; `--out` saves the result |
| Tests | 107 pass; the UI is tested headlessly with Textual's `run_test()` and a fake agent; the pydantic-ai engines with `FunctionModel` |

Issues and PRs: #1 harness (PR #2), #3 agent + TUI (PR #8), #4 write to disk (PR #6), #5 shell checks (PR #7) are merged.
#9 pydantic-ai is in **PR #10, open**. Open follow-ups: #11 control run, #12 context trimming for `pai-tools`,
#13 temperature warning.

## 2. What the evidence supports

Strength: **solid** = reproduced several times or enforced by a test; **suggestive** = a few runs; **single** = one run.

| Finding | Strength | Where |
|---|---|---|
| Constraining the *output schema* beats asking a small model to fill a flat schema: an all-optional flat schema let a 1.5b model call `edit_file` with no arguments 25 times; an all-required one made it invent values; a discriminated union gave clean, tool-specific output | suggestive (6 samples per schema, plus the agent runs) | `2026-10-16-local-model-results.md` §2-3 |
| Model size matters a lot: 1.5b finds the bug but cannot verify and loops; 4b-8b models complete the task | single run per model | `2026-10-16-local-model-results.md`, `2026-10-18-pydantic-ai-engines.md` |
| Gemma 3 4b works with the schema-constrained engines; Ollama refuses tools for it outright (`400`) | solid for the refusal; suggestive for the rest | `2026-10-18-pydantic-ai-engines.md` |
| `builtin` and `pai-native` were equally reliable on 3 local models (6 of 6) | suggestive | `2026-10-18-pydantic-ai-engines.md` |
| `pai-tools` (native tool calling) failed on small Ollama models: prose instead of an edit, bad tool arguments, or the call returned as plain text (0 of 7 on Llama 3.1 8b) | suggestive | `2026-10-18-pydantic-ai-engines.md` |
| ...but `pai-tools` succeeded 10 of 10 on Claude Haiku 5.5, and deferred approvals worked against a hosted model | suggestive (10 runs) | `2026-10-18-pai-tools-haiku.md` |
| `pai-native` works on Anthropic: 3 of 3, no invalid replies | single batch of 3 | `2026-10-18-anthropic-live.md` |
| pydantic-ai saved almost no code (client 89 vs 82 lines, loop 165 vs 184 without trimming) | solid (counted) | `2026-10-18-pydantic-ai-engines.md` |
| On Anthropic, pydantic-ai rewrites our schema so tool names and `minLength` become description text, not enforced | solid (captured requests, offline) | `reports/schema_probe_providers.py` |

What these add up to: for small local models, **schema-constrained output is the reliable path**; native tool calling
works once the model is a capable tool-caller. Provider switching is real, but the strength of the guarantees depends
on the provider's structured-output mode.

## 3. Bugs that real runs found and unit tests did not

This is the most reusable lesson. Each of these passed tests, or would have, until a real model or a real server was involved:

- **Stale bytecode:** rewriting a same-size candidate within a second made Python run the old `.pyc`; found as a flaky `--check-cwd` test.
- **Cross-thread event loop:** the pydantic-ai client failed on its second call from a different thread (the TUI's pattern). A first test using a mock transport **passed without the fix**; only a real keep-alive local server reproduced it.
- **Tool-ordering race:** a model batched `edit_file` and `bash pytest` into one response and pydantic-ai ran them concurrently. It surfaced only on a model that batches calls.
- **A missing model looked like a broken app:** three retries labelled "reply was invalid" with an MDN link, found when it hit a real machine.
- **A display tidy-up squashed code indentation**, found in a screenshot, not a test.

Mistakes of mine worth recording: I pushed a failing test once (piping pytest into `tail` hid its exit code); and
several statements in reports and issues went beyond the evidence and were corrected after checking (the edit
arguments in the Llama failures, "failed 0 of 7", the effect of a warning on the TUI).

## 4. What we don't know

- **How well `wombat-tui` works for a person on a real project.** The author reports a **successful run of the client on their
  Mac** (self-reported, after the missing-model error was fixed; which model, engine and task, how long it took, and anything
  that felt rough were not recorded). Everything else about the UI with a real model comes from a script driving
  Textual's test harness, not from a person at a terminal. One reported run on a tiny task says it can work, not that it is pleasant.
- **Anything larger than a one-line bug fix.** Multi-file edits, long sessions, long histories (context trimming has never been
  exercised against a real long session), and bigger models.
- **Whether Anthropic's schema rewrite matters** (#11): the system prompt also names the tools, so Haiku may succeed regardless.
- **Other providers.** OpenAI and Google schemas were only inspected offline; no live call.
- **Approval *denial* with a real model.** Covered by scripted tests only.
- **Speed and cost on real hardware.** Everything local ran on 4 CPUs without a GPU; cost figures assume unverified prices.
- **Sandboxing.** Approved `bash` runs as the user; path confinement and approvals are not a sandbox.

## 5. Recommendations (current)

- Default engine: `builtin`. Use `pai-native` for another provider or when you want pydantic-ai in the model seat.
  Use `pai-tools` only with a capable tool-calling model.
- Use a 4b-8b local model or larger for the agent; 1.5b is too small. Raise Ollama's context
  (`OLLAMA_CONTEXT_LENGTH`) or it truncates silently.
- Treat all numbers above as order-of-magnitude. Before relying on any claim for a decision, rerun it on a task like yours.

## 6. Next steps (none started)

1. Merge PR #10 after review.
2. Write down what the Mac run actually was (model, engine, task, timing, rough edges), then try `wombat-tui` on a real
   project, not a toy, and report what feels wrong.
3. #11, #12, #13 (control run, context trimming for `pai-tools`, temperature warning).
4. A broader task set (several tasks, several sizes) so that rates, not single runs, can be reported.
5. Streaming output, and sandboxing beyond path confinement.

## 7. Where things are

See `reports/README.md` for the index of reports, drivers and data.
