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

Execution is process-isolated only; use a container for untrusted output.
