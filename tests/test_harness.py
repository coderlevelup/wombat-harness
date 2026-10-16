import json

import httpx
import pytest

from wombat_harness import CodeResponse, Harness, LLMError, OpenAICompatClient, Task, run_python


class Scripted:
    """Fake client replaying canned CodeResponses / errors."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, schema):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def good(code):
    return CodeResponse(plan="p", code=code)


def test_fence_stripped_and_syntax_rejected():
    assert good("```python\nx = 1\n```").code == "x = 1"
    with pytest.raises(ValueError):
        good("def (:")


def test_retries_with_feedback_until_tests_pass():
    c = Scripted(good("def add(a, b): return a - b"), good("def add(a, b): return a + b"))
    r = Harness(c).run(Task(prompt="add", tests="assert add(1, 2) == 3"))
    assert r.ok and len(r.attempts) == 2
    assert "AssertionError" in c.calls[1][-1]["content"]


def test_llm_error_recovers_and_exhaustion_fails():
    c = Scripted(LLMError("bad json"), good("raise SystemExit(1)"), good("1/0"))
    r = Harness(c, max_attempts=3).run(Task(prompt="x"))
    assert not r.ok and r.attempts[0].error and r.code == "1/0"


def test_timeout():
    assert run_python("while True: pass", timeout=0.5).timed_out


def test_client_sends_schema_and_parses():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        content = '```json\n{"plan": "p", "code": "print(1)"}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    c = OpenAICompatClient(transport=httpx.MockTransport(handler))
    out = c.complete([{"role": "user", "content": "hi"}], CodeResponse)
    assert out.code == "print(1)"
    assert seen["response_format"]["json_schema"]["schema"]["properties"]["code"]


def test_run_command_ok_fail_timeout_missing(tmp_path):
    import sys
    from wombat_harness import run_command

    py = sys.executable
    assert run_command([py, "-c", "print('hi')"], tmp_path).stdout.strip() == "hi"
    r = run_command([py, "-c", "import sys; sys.exit(3)"], tmp_path)
    assert not r.ok and r.returncode == 3
    assert run_command([py, "-c", "import time; time.sleep(30)"], tmp_path, timeout=0.5).timed_out
    assert not run_command("definitely-not-a-command", tmp_path).ok
    assert not run_command("", tmp_path).ok
    assert len(run_command([py, "-c", "print('x'*500)"], tmp_path, max_output=50).stdout) == 50


def test_check_command_failure_is_fed_back_then_passes():
    import sys

    test = "from solution import add\ndef test_add():\n    assert add(1, 2) == 3\n"
    c = Scripted(good("def add(a, b): return a - b"), good("def add(a, b): return a + b"))
    task = Task(prompt="add", check=f"{sys.executable} -m pytest -q",
                files={"test_solution.py": test})
    r = Harness(c).run(task)
    assert r.ok and len(r.attempts) == 2
    assert not r.attempts[0].check.ok and r.attempts[1].check.ok
    assert "pytest -q" in c.calls[1][-1]["content"] and "assert" in c.calls[1][-1]["content"]


def test_check_not_run_if_code_fails():
    c = Scripted(good("1/0"))
    r = Harness(c, max_attempts=1).run(Task(prompt="x", check="true"))
    assert not r.ok and r.attempts[0].check is None


@pytest.mark.parametrize("bad", ["../x.py", "/etc/passwd", "a/../../x", "solution.py"])
def test_files_rejects_escaping_paths(bad):
    with pytest.raises(ValueError):
        Task(prompt="x", files={bad: ""})


# --- saving to disk (issue #4) ---

def _ok_result(tests=None):
    return Harness(Scripted(good("x = 1"))).run(Task(prompt="p", tests=tests))


def test_save_writes_code(tmp_path):
    out = tmp_path / "foo.py"
    assert _ok_result().save(out) == [out]
    assert out.read_text() == "x = 1\n"


def test_save_refuses_failed_run_and_existing_file(tmp_path):
    failed = Harness(Scripted(good("1/0")), max_attempts=1).run(Task(prompt="p"))
    with pytest.raises(ValueError, match="did not succeed"):
        failed.save(tmp_path / "a.py")
    assert not (tmp_path / "a.py").exists()

    out = tmp_path / "b.py"
    out.write_text("keep")
    with pytest.raises(FileExistsError):
        _ok_result().save(out)
    assert out.read_text() == "keep"
    _ok_result().save(out, force=True)
    assert out.read_text() == "x = 1\n"


def test_save_rejects_bad_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ok_result().save(tmp_path / "missing" / "foo.py")
    with pytest.raises(ValueError, match="directory"):
        _ok_result().save(tmp_path)
    target = tmp_path / "real.py"
    target.write_text("keep")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        _ok_result().save(link, force=True)
    assert target.read_text() == "keep"


def test_save_writes_tests_alongside(tmp_path):
    out = tmp_path / "foo.py"
    paths = _ok_result(tests="assert x == 1").save(out, write_tests=True)
    assert paths == [out, tmp_path / "test_foo.py"]
    assert (tmp_path / "test_foo.py").read_text() == "assert x == 1\n"


def test_save_write_tests_without_tests_is_error(tmp_path):
    with pytest.raises(ValueError, match="no tests"):
        _ok_result().save(tmp_path / "foo.py", write_tests=True)
    assert not (tmp_path / "foo.py").exists()


def test_save_checks_all_targets_before_writing(tmp_path):
    (tmp_path / "test_foo.py").write_text("keep")
    with pytest.raises(FileExistsError):
        _ok_result(tests="assert x == 1").save(tmp_path / "foo.py", write_tests=True)
    assert not (tmp_path / "foo.py").exists()


def _cli(monkeypatch, *replies):
    from wombat_harness import cli

    monkeypatch.setattr(cli, "OpenAICompatClient", lambda *a, **k: Scripted(*replies))
    return cli.main


def test_cli_out_writes_file_and_still_prints(tmp_path, monkeypatch, capsys):
    out = tmp_path / "foo.py"
    assert _cli(monkeypatch, good("x = 1"))(["p", "--out", str(out)]) == 0
    assert out.read_text() == "x = 1\n"
    assert "x = 1" in capsys.readouterr().out


def test_cli_out_refuses_overwrite_without_force(tmp_path, monkeypatch, capsys):
    out = tmp_path / "foo.py"
    out.write_text("keep")
    assert _cli(monkeypatch, good("x = 1"))(["p", "--out", str(out)]) != 0
    assert out.read_text() == "keep"
    assert "--force" in capsys.readouterr().err
    assert _cli(monkeypatch, good("x = 1"))(["p", "--out", str(out), "--force"]) == 0
    assert out.read_text() == "x = 1\n"


def test_cli_does_not_write_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "foo.py"
    assert _cli(monkeypatch, good("1/0"))(["p", "--out", str(out), "--attempts", "1"]) == 1
    assert not out.exists()


def test_cli_write_tests_requires_out_and_tests(tmp_path, monkeypatch):
    main = _cli(monkeypatch)
    with pytest.raises(SystemExit):
        main(["p", "--write-tests"])
    with pytest.raises(SystemExit):
        main(["p", "--force"])
    t = tmp_path / "t.py"
    t.write_text("assert x == 1")
    with pytest.raises(SystemExit):
        main(["p", "--out", str(tmp_path / "foo.py"), "--write-tests"])
    out = tmp_path / "foo.py"
    assert _cli(monkeypatch, good("x = 1"))(
        ["p", "--tests", str(t), "--out", str(out), "--write-tests"]
    ) == 0
    assert (tmp_path / "test_foo.py").read_text() == "assert x == 1\n"


def _project(tmp_path):
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    return tmp_path


def test_check_cwd_runs_in_project_and_restores_original(tmp_path):
    import sys

    proj = _project(tmp_path)
    target = proj / "calc.py"
    target.write_text("ORIGINAL\n")
    c = Scripted(good("def add(a, b): return a - b"), good("def add(a, b): return a + b"))
    task = Task(prompt="add", check=f"{sys.executable} -m pytest -q -p no:cacheprovider",
                check_cwd=proj, check_target=target)
    r = Harness(c).run(task)
    assert r.ok and len(r.attempts) == 2
    assert not r.attempts[0].check.ok and r.attempts[1].check.ok
    assert target.read_text() == "ORIGINAL\n"  # check never leaves a candidate behind
    r.save(target, force=True)
    assert "a + b" in target.read_text()


def test_check_cwd_removes_candidate_when_target_was_new(tmp_path):
    import sys

    proj = _project(tmp_path)
    task = Task(prompt="add", check=f"{sys.executable} -m pytest -q -p no:cacheprovider",
                check_cwd=proj, check_target=proj / "calc.py")
    r = Harness(Scripted(good("def add(a, b): return a - b")), max_attempts=1).run(task)
    assert not r.ok and not (proj / "calc.py").exists()


def test_check_cwd_restores_on_interrupt(tmp_path, monkeypatch):
    from wombat_harness import harness

    target = tmp_path / "calc.py"
    target.write_text("ORIGINAL\n")

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(harness, "run_command", boom)
    task = Task(prompt="x", check="true", check_cwd=tmp_path, check_target=target)
    with pytest.raises(KeyboardInterrupt):
        Harness(Scripted(good("x = 1"))).run(task)
    assert target.read_text() == "ORIGINAL\n"


def test_check_cwd_validation(tmp_path):
    with pytest.raises(ValueError):  # target outside cwd
        Task(prompt="x", check="true", check_cwd=tmp_path, check_target=tmp_path.parent / "x.py")
    with pytest.raises(ValueError):  # missing check_target
        Task(prompt="x", check="true", check_cwd=tmp_path)
    with pytest.raises(ValueError):  # no check
        Task(prompt="x", check_cwd=tmp_path, check_target=tmp_path / "x.py")
    with pytest.raises(ValueError):  # files not allowed
        Task(prompt="x", check="true", check_cwd=tmp_path, check_target=tmp_path / "x.py",
             files={"a.py": ""})
    link = tmp_path / "link.py"
    link.symlink_to(tmp_path / "real.py")
    task = Task(prompt="x", check="true", check_cwd=tmp_path, check_target=link)
    with pytest.raises(ValueError, match="symlink"):
        Harness(Scripted(good("x = 1"))).run(task)
