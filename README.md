# wombat-harness
Python harness for small models, named for the poetry inspector, Wombat Harness, in the Monty Python sketch "Poets". 

> "Oh, WOMBAT! Wombat Harness! Take me to that place where eternity knows no bounds!"
> - Monty Python S2EP17 [Watch on YouTube](https://youtu.be/gAobK3fyGDI?si=gSAyMI0Up1p2kinY&t=223) [Script](http://www.montypython.50webs.com/scripts/Series_2/24.htm) 

**Where this is going:** see [ROADMAP.md](ROADMAP.md): first, a quick proof of possibility, completing real coding tasks with a capable cloud model and the same tasks with a tiny model and finessing finagling, keeping the harness plain Python that is easy to inspect and adapt to other small models and other worlds. Then exploring compositions comprising multiple domain specific small models. 

Generates Python with a small local model (Ollama, llama.cpp, LM Studio — anything
OpenAI-compatible). The pydantic `CodeResponse` schema is sent as a JSON schema for
constrained decoding, the reply is validated (JSON + `ast.parse`), executed in a
subprocess with optional asserts, and failures are fed back for retry.

```
pip install -e '.[dev]'
ollama pull qwen2.5-coder:1.5b
wombat "write is_prime(n)" --tests tests_prime.py -v
pytest
```

Run a project check (e.g. pytest) as the final gate. The code is written to
`solution.py` in a scratch dir; extra files go beside it. The command is yours,
never the model's, and runs without a shell (`shlex`-split) with a timeout:

```
wombat "write is_prime(n)" --file test_solution.py --check "python -m pytest -q"
```

Save the result with `--out is_prime.py` (successful runs only; refuses to overwrite
without `--force`; `--write-tests` also writes `test_is_prime.py` beside it). From Python:
`result.save(path, force=False, write_tests=False)`.

To check against a real project (its own tests, imports, config), point the check at it.
Each candidate is dropped at `--out` for the check and the file is restored afterwards, so
failed attempts never leave junk behind; only the final success is saved:

```
wombat "implement add(a, b)" --check-cwd ./myproj --check "python -m pytest -q" --out myproj/calc.py
```

Execution is process-isolated only; use a container for untrusted output.

## Coding agent (chat)

An opencode / Claude Code style loop on a local model: you chat, the model reads, searches and
edits files and runs commands in a workspace directory, and you approve the risky steps.

```
pip install -e '.[tui]'
wombat-tui --workspace ./myproj            # chat UI
wombat-agent "fix the failing test" --workspace ./myproj   # one-shot, prompts y/n on stdin
```

- **Model:** defaults to `qwen2.5-coder:7b`; run `ollama pull qwen2.5-coder:7b` first (a missing model
  gives `model server returned 404 ... (try: ollama pull ...)`). Gemma 3 4b also worked
  (`--model gemma3:4b`); 1.5b did not. See `reports/`.
- **Tools:** `list_dir`, `read_file`, `grep`, `write_file`, `edit_file`, `bash`, `finish`. File tools
  cannot leave the workspace (paths and symlinks are resolved and checked).
- **Approvals:** reads are automatic. In `ask` mode every write, edit and command shows a diff or the
  command and waits for `y` / `n` / `a` (allow everything this session). `--mode edits` auto-approves
  file edits but still asks for commands; `--mode yolo` asks for nothing. `/mode` changes it live.
- **You are only asked about actions that could happen.** An edit whose text isn't found or matches several places, or a write outside the workspace or onto a directory, is not put to you: it fails and the model gets the error.
- **Test-file edits are flagged.** An approval for a change to a test file (`test_*.py`, `*_test.py`, `conftest.py`, `tests/`, `*.test.*`, `*.spec.*`) shows a warning in the bar and the diff, and the model is told to fix the code, not the tests. Paths in summaries and diffs are shown relative to the workspace.
- **Denying (`n`) ends the turn** and returns you to the input box, with the denial kept in the conversation, so what you type next steers the agent instead of it guessing its next action.
- **Missing tools: Retry / Fail.** If the model runs a command that isn't installed (`command not found`, or `python -m X` with no module `X`),
  wombat pauses and tells you what is missing, for which Python (its path), how to install it when that is certain, and how many times it has
  failed this turn. Press **`r`** to retry once you've fixed it in another terminal (the model never sees the failure), or **`f`** to not retry:
  the model is told what is missing and asked for the install steps. It can propose running them with `bash` (you approve it as usual), or give
  you the steps to run yourself, after which you just tell it to retry. Esc still interrupts. The model is told not to install packages
  without asking. Headless `wombat-agent` asks on a terminal and fails the call without one.
- **Keys:** Esc interrupts (cooperatively: after the current model call or command), `/clear` starts
  over, `/help`, `Ctrl+Q` quits.
- **Why a union schema:** each step is `{thought, action}` where `action` is a pydantic discriminated
  union of the tools, sent to the server as a JSON schema for constrained decoding. That makes the
  model supply the right arguments for each tool; flat optional/required schemas did not (1.5b model).
- **Context:** old tool output is elided and old turns dropped to stay under `context_chars` (16k).
  Ollama's default context window is small; raise it (e.g. `OLLAMA_CONTEXT_LENGTH=8192`) or the
  server will silently truncate the prompt.

Architecture: `wombat_harness/agent` is the engine (no UI imports); `wombat_harness/tui` talks to it only
through the `AgentPort` protocol. A turn runs in a Textual thread worker that only posts messages.
Tests drive the UI headlessly with `App.run_test()` and a fake agent (`tests/test_tui.py`).

## Engines and providers

Three engines run the same agent behind one interface (`--engine` on `wombat-agent` and `wombat-tui`):

| Engine | What it is | Use it for |
|---|---|---|
| `builtin` (default) | own loop + own HTTP client; the action is a pydantic discriminated union sent as a JSON schema | local models on Ollama; no extra dependencies |
| `pai-native` | the same loop, with pydantic-ai as the model client (`NativeOutput`) | switching providers; works on models without tool support |
| `pai-tools` | a pydantic-ai `Agent` with native function tools and deferred approvals | models that are good at tool calling (worked on Haiku; failed on 4b-8b Ollama models) |

```
pip install -e '.[pydantic-ai]'
wombat-agent "fix the failing test" --engine pai-native --model gemma3:4b                 # Ollama (default)
WOMBAT_ANTHROPIC_API_KEY=... wombat-agent "..." --engine pai-tools --model anthropic:claude-haiku-5-5
```

`--model` takes `provider:name` (`ollama` or `anthropic`); a bare name is an Ollama tag. The Anthropic key is read
**only** from `WOMBAT_ANTHROPIC_API_KEY`, and the endpoint is always passed explicitly, so `ANTHROPIC_API_KEY` /
`ANTHROPIC_BASE_URL` set for some other tool in your environment are ignored. Results, and how much to trust
them, are in `reports/` (start with `2026-10-18-pydantic-ai-engines.md`).
