# Reports

Hand-run experiments and write-ups, newest first within each theme. All are small (one tiny task, few runs): read the
caveats section of each before relying on a number.

| Report | What it covers |
|---|---|
| `2026-10-19-acceptance-tasks.md` | the five acceptance tasks (`wombat_harness.tasks`) run once on Haiku `pai-tools`, `qwen2.5-coder:7b` and `gemma3:4b`; where the small models fail |
| `2026-10-18-project-summary.md` | **Start here.** What exists, what the evidence supports, what is unknown, what is open |
| `2026-10-18-gemma3-1b.md` | the author's `gemma3:1b` Mac run, a 20-run reproduction (0 genuine fixes), a flaw in the success criterion (audited), and the harness gaps it exposed (#14), including the denial fix |
| `2026-10-18-pai-tools-haiku.md` | `pai-tools` on Claude Haiku 5.5 (10 of 10), deferred approvals, the tool-ordering race and its fix |
| `2026-10-18-anthropic-live.md` | `pai-native` on Claude Haiku 5.5 (3 of 3); the schema-rewrite question |
| `2026-10-18-pydantic-ai-engines.md` | `builtin` vs `pai-native` vs `pai-tools` on Gemma 3 4b, Qwen2.5-Coder 7b, Llama 3.1 8b; code size; decision against #9 |
| `2026-10-16-tui-screenshots.md` | what `wombat-tui` looks like on a real run, and the bugs those runs exposed |
| `2026-10-16-local-model-results.md` | first agent and one-shot results on small local models; the schema experiment |

| Script | Purpose |
|---|---|
| `compare_engines.py` | engine x model comparison on the bug-fix task, judged by re-running pytest |
| `live_anthropic.py`, `live_anthropic_tools.py` | live Anthropic runs for `pai-native` and `pai-tools` (need `WOMBAT_ANTHROPIC_API_KEY`; cost a few cents) |
| `schema_experiment.py` | flat vs union schema on a small model (needs Ollama) |
| `schema_probe_providers.py` | what each provider does to our schema (offline, no key) |
| `repro_gemma3_1b.py` | the author's exact project on a small model, builtin and pai-native, strict judge |

`data/` holds the raw JSONL rows behind the tables. `screenshots/` has the TUI images.
