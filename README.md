# wombat-harness
Python harness, not a novelist or a nude man.

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
