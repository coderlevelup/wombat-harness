# Roadmap

Where wombat-harness is going, in order. Nothing here is a promise: each item says what motivates it, and most are ideas to try. The
evidence so far is in `reports/` (small hand-run experiments on one tiny task, not a benchmark).

## 1. Complete real coding tasks with a capable model

**Goal:** give the harness a coding task and have it reliably finish it, using a capable model (a hosted one such as Claude Haiku/Sonnet,
or a larger local model).

**Where we are:** on the one task we have used (a one-line bug fix in a two-file project) Claude Haiku 5.5 finished every run (13 of 13 across
`pai-native` and `pai-tools`), and the harness's approvals, retry/fail on missing tools, loop guard and denial handling work against real
models. That task is far too easy to say the goal is met.

**To do:**
- [~] (#18, task set and `python -m wombat_harness.tasks` runner in place; not yet run against a model) A small set of **realistic tasks** (fix a bug across files, add a feature with tests, a small refactor, work in an unfamiliar repo) with a
  clear "done" check that a model can't satisfy by editing the test. These are the harness's acceptance tests, not a leaderboard.
- [ ] Long sessions: context trimming for `pai-tools` (#12), and a real long-history run on either engine
- [ ] Hosted-model rough edges: the `temperature` warning (#13), whether Anthropic's schema rewrite matters (#11)
- [ ] A policy decision: should `edits` mode still ask before changing a test file?
- [ ] Safety beyond approvals: `bash` runs as the user once approved, and path confinement is not a sandbox
- [ ] Streaming output, so a long model call isn't a silent wait

## 2. Try the same tasks with a tiny model, with whatever affordances it needs

**Goal:** find out how far a small local model (1b to 8b) can get on the same tasks if the *harness* does more of the work: better scaffolding,
not a better model. Models around 1b did not finish even the easy task (0 of 20 genuine fixes for `gemma3:1b`); 4b to 8b models did.

**Affordances to try** (each comes from something we saw go wrong; all are hypotheses, cheapest first):
- The small model read the test, decided the code was fine, and edited the test: **run the failing test first and show the failure** (traceback,
  files named in it) so it starts from the real problem. *Tried (`--check-first`, one run per task on 2 models): no improvement, see `reports/2026-10-19-acceptance-tasks.md`.* Already flagged: test-file edits, and the prompt says fix the code, not the tests.
- It forgot to verify, or ran the wrong command: **the harness runs the project's check itself after an edit** and feeds the result back
  (the one-shot generator already has `--check` / `--check-cwd`).
- `edit_file` fails when the `old` text is missing or ambiguous: an easier edit primitive (by line range, or replace a function) for small files.
- Too many tools to choose from: **staged tools** (investigate, then edit, then verify) or a smaller tool set per step.
- Loops and bad JSON: already handled (constrained output, a guard that spots the same call returning the same result).
- **Per-model profiles**: prompt wording, tool subset, context budget, temperature, and which engine suits a model (a 4b to 8b model that can't call tools
  works with constrained output; a strong tool-caller works with native tool calling).

**How we'll know:** the same acceptance tasks as in section 1. The question is whether the harness gets a tiny model to *done*, not which
model is best.

## 3. Pythonic, easy to modify: other small models, and other worlds

**Direction:** the harness should be plain Python that someone can read in an afternoon and change to fit **a different small model** or **a
different world**: a virtual-world platform such as WorkAdventure, or a robot harness. Plain pydantic models, small Protocols, no framework to
fight, and extension by adding code rather than editing the loop.

**What already helps:** the pieces are ports with fakes in the tests: the model client (`ChatClient`), the agent behind the UI (`AgentPort`),
approvals and "something is missing" callbacks, and the UI is just one front end. A tool is a pydantic model with `run`, `preview`, `preflight`
and `needs_approval`.

**What gets in the way today** (found by reading the code):
- Adding a tool means editing **three places**: the tool union in `agent/tools.py`, the tool list in the system prompt, and a wrapper in the
  pydantic-ai engine. A single tool registry would let a tool be defined once.
- The system prompt, the tools and the `Workspace` are all about *files and a shell*. A different world needs its own observations and actions.
- Model-specific behaviour has no home: there are options (`--model`, `--engine`, step and context limits) but no per-model profile.

**To do:**
- [ ] One tool registry: define a tool once and derive its schema, prompt text and engine wrappers from it
- [ ] Per-model profiles (see section 2)
- [ ] Generalise `Workspace` into a pluggable "world": what the agent can observe, what it can do, and what needs approval
- [ ] A second, non-coding world, as a small sample, to prove the seams (most likely a simulated one first)

**Honest unknowns for world interfaces:** I have not looked at WorkAdventure's or any robot's interface, and nothing here is designed. They probably
differ from coding in ways that matter: real-time events rather than turns, perception beyond text, and, for a robot, physical safety. Approvals
would then act as safety interlocks and need an immediate stop, and testing against a simulator before real hardware. Each needs its own design before
any commitment.

## Not on the roadmap (yet)
Ranking models, publishing benchmark numbers, a plugin marketplace, removing the `builtin` engine.
