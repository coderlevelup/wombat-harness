import pytest

from wombat_harness.agent.missing import Fix, Missing, describe, detect_missing


def out(code, text):
    return f"exit code {code}\n{text}"


@pytest.mark.parametrize("command,output,kind,name", [
    ("pytest -q", out(127, "bash: line 1: pytest: command not found"), "command", "pytest"),
    ("pytest -q", out(127, "bash: pytest: command not found"), "command", "pytest"),
    ("ruff check .", out(127, "zsh: command not found: ruff"), "command", "ruff"),
    ("pytest", out(127, "/bin/sh: 1: pytest: not found"), "command", "pytest"),
    ("cd src && npm test", out(127, "bash: line 1: npm: command not found"), "command", "npm"),
    ("python -m pytest -q", out(1, "/usr/bin/python: No module named pytest"), "module", "pytest"),
    ("python3 -m pytest", out(1, "/usr/bin/python3: No module named pytest"), "module", "pytest"),
    ("python -m foo.bar", out(1, "/usr/bin/python: No module named 'foo'"), "module", "foo"),
])
def test_detects_a_missing_tool(command, output, kind, name):
    m = detect_missing(command, output)
    assert m is not None and (m.kind, m.name, m.command) == (kind, name, command)


def test_module_hint_uses_the_interpreter_the_model_asked_for():
    assert detect_missing("python3 -m pytest", out(1, "No module named pytest")).install_hint == "python3 -m pip install pytest"
    assert detect_missing("pytest", out(127, "bash: pytest: command not found")).install_hint is None  # no guessing an installer


@pytest.mark.parametrize("command,output", [
    ("python -m pytest -q", out(1, "1 failed in 0.02s")),                                  # a normal test failure
    ("python -m pytest -q", out(1, "ModuleNotFoundError: No module named 'requests'")),   # a missing import in the project, not the tool
    ("python script.py", out(1, "ModuleNotFoundError: No module named 'requests'")),      # not `-m`: could be the model's own bug
    ("grep -r 'not found' .", out(1, "docs.md: error text: not found")),                  # the words appear in ordinary output
    ("python -m pytest -q", out(0, "2 passed")),                                          # success
    ("ls", out(2, "ls: cannot access 'x': No such file or directory")),                   # a missing *file* is not a missing tool
])
def test_does_not_misfire(command, output):
    assert detect_missing(command, output) is None


def test_describe_counts_failures_and_names_the_python():
    m = Missing(kind="module", name="pytest", command="python -m pytest", install_hint="python -m pip install pytest",
                interpreter="/opt/py/bin/python")
    first, third = describe(m, 1), describe(m, 3)
    assert "pytest" in first and "/opt/py/bin/python" in first and "python -m pip install pytest" in first
    assert "failed once" in first and "failed 3 times" in third
    cmd = describe(Missing(kind="command", name="ruff", command="ruff .", install_hint=None, interpreter=None), 2)
    assert "ruff" in cmd and "PATH" in cmd and "failed 2 times" in cmd


def test_fix_choices():
    assert {f.value for f in Fix} == {"retry", "fail"}


def test_headless_resolver_asks_on_a_terminal_and_fails_without_one():
    from wombat_harness.agent.cli import resolve_missing

    m = Missing(kind="command", name="ruff", command="ruff .")
    assert resolve_missing(m, 1, tty=False) is Fix.FAIL  # nobody to ask: never hang
    answers = iter(["huh?", "r"])
    assert resolve_missing(m, 1, tty=True, ask=lambda prompt: next(answers)) is Fix.RETRY  # re-asks on junk input
    assert resolve_missing(m, 1, tty=True, ask=lambda prompt: "f") is Fix.FAIL
    assert resolve_missing(m, 1, tty=True, ask=lambda prompt: "") is Fix.FAIL  # bare Enter = the safe choice
