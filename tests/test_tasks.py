"""The acceptance tasks must be sound: unsolved they are not done, solved they are, and cheating does not count."""
import pytest

from wombat_harness.agent.loop import Event
from wombat_harness.tasks import TASKS, check_task, run_task


@pytest.fixture(params=list(TASKS.values()), ids=list(TASKS))
def task(request):
    return request.param


def test_unsolved_task_is_not_done(task, tmp_path):
    task.write(tmp_path)
    assert not check_task(task, tmp_path).done


def test_reference_solution_is_done(task, tmp_path):
    task.write(tmp_path)
    task.write(tmp_path, task.solution)
    assert check_task(task, tmp_path).done


def test_every_task_has_a_failing_visible_test_or_a_hidden_check(task, tmp_path):
    task.write(tmp_path)
    verdict = check_task(task, tmp_path)
    assert not (verdict.tests_pass and verdict.hidden_pass)


def test_editing_a_protected_test_is_not_done(tmp_path):
    task = TASKS["fix-add"]
    task.write(tmp_path)
    (tmp_path / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == -1\n")
    verdict = check_task(task, tmp_path)
    assert verdict.tests_pass and not verdict.protected_intact and not verdict.done


def test_run_task_judges_the_result_not_the_reply(tmp_path):
    task = TASKS["fix-add"]

    class Cheater:
        def run_turn(self, text, on_event, approve, *a, **k):
            (tmp_path / "test_calc.py").write_text("def test_nothing():\n    pass\n")
            on_event(Event(kind="tool", text="write_file test_calc.py"))
            return "Fixed it!"

    result = run_task(task, lambda root: Cheater(), root=tmp_path)
    assert result.replied and result.verdict.tests_pass and not result.done
