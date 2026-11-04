"""Acceptance tasks: a small project, a prompt, and a check a model cannot satisfy by editing the tests."""
from .task import Task, Verdict, run_task, check_task
from .catalog import TASKS

__all__ = ["Task", "Verdict", "TASKS", "run_task", "check_task"]
