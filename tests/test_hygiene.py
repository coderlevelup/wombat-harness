"""Cheap structural checks that tests of behaviour won't catch."""
import ast
import collections
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_no_duplicate_definitions_in_a_module_or_class():
    """A scripted edit once duplicated half of loop.py: two `class Agent`, the first cut off mid-method. Python quietly
    used the second, so every behavioural test passed. Duplicates are never intentional here."""
    problems = []
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text())
        scopes = [(tree.body, "module")] + [(n.body, f"class {n.name}") for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        for body, where in scopes:
            names = collections.Counter(n.name for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            problems += [f"{name} defined {c} times in {f.relative_to(SRC)} ({where})" for name, c in names.items() if c > 1]
    assert not problems, problems
