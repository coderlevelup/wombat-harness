"""The task set. Each task's `hidden` check asserts behaviour the visible tests do not, so a model that edits a
test, or special-cases it, does not pass. `tests/test_tasks.py` proves every task fails as given and passes with its `solution`."""
from __future__ import annotations

from .task import Task

FIX_ADD = Task(
    name="fix-add",
    prompt="The test in test_calc.py is failing. Find the bug in calc.py and fix it, then run the tests.",
    files={
        "calc.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
        "test_calc.py": "from calc import add, mul\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n",
    },
    protected=["test_calc.py"],
    hidden="from calc import add; assert add(1, 2) == 3 and add(2, 3) == 5 and add(0, 0) == 0",
    solution={"calc.py": "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"},
)

# The failing test is in checkout.py's tests, but the bug is in pricing.py (percent divided by 10).
FIX_ACROSS_FILES = Task(
    name="fix-across-files",
    prompt="test_checkout.py fails. Work out why and fix it, then run the tests.",
    files={
        "pricing.py": "def apply_discount(price, percent):\n    return price - price * percent / 10\n",
        "checkout.py": (
            "from pricing import apply_discount\n\n\n"
            "def total(prices, discount_percent=0):\n"
            "    return round(apply_discount(sum(prices), discount_percent), 2)\n"),
        "test_checkout.py": (
            "from checkout import total\n\n\n"
            "def test_no_discount():\n    assert total([10, 5]) == 15\n\n\n"
            "def test_ten_percent():\n    assert total([10, 10], 10) == 18\n"),
    },
    protected=["test_checkout.py"],
    hidden=(
        "from pricing import apply_discount; from checkout import total\n"
        "assert apply_discount(200, 25) == 150 and apply_discount(80, 0) == 80\n"
        "assert total([50, 50], 50) == 50 and total([19.99], 15) == 16.99"),
    solution={"pricing.py": "def apply_discount(price, percent):\n    return price - price * percent / 100\n"},
)

ADD_FEATURE = Task(
    name="add-feature",
    prompt=(
        "Add a function `word_count(text)` to textutil.py that returns how many words are in the text "
        "(words are separated by any whitespace; empty text has 0). Add a test for it to test_textutil.py and run the tests."),
    files={
        "textutil.py": "def shout(text):\n    return text.upper() + \"!\"\n",
        "test_textutil.py": "from textutil import shout\n\n\ndef test_shout():\n    assert shout(\"hi\") == \"HI!\"\n",
    },
    hidden=(
        "from textutil import word_count, shout\n"
        "assert word_count('') == 0 and word_count('   ') == 0 and word_count('one') == 1\n"
        "assert word_count('a  b\\tc\\nd') == 4 and shout('hi') == 'HI!'\n"
        "t = open('test_textutil.py').read(); assert 'word_count' in t and 'test_shout' in t"),
    solution={
        "textutil.py": "def shout(text):\n    return text.upper() + \"!\"\n\n\ndef word_count(text):\n    return len(text.split())\n",
        "test_textutil.py": (
            "from textutil import shout, word_count\n\n\ndef test_shout():\n    assert shout(\"hi\") == \"HI!\"\n\n\n"
            "def test_word_count():\n    assert word_count(\"a b  c\") == 3\n"),
    },
)

RENAME_ACROSS = Task(
    name="rename-function",
    prompt=(
        "Rename the function `calc_total` to `compute_total` everywhere it is used in this project "
        "(definition, callers and tests), without changing behaviour, then run the tests."),
    files={
        "orders.py": "def calc_total(items):\n    return sum(price * qty for price, qty in items)\n",
        "report.py": (
            "from orders import calc_total\n\n\n"
            "def summary(items):\n    return f\"total={calc_total(items)}\"\n"),
        "test_orders.py": (
            "from orders import calc_total\nfrom report import summary\n\n\n"
            "def test_total():\n    assert calc_total([(2, 3), (1, 4)]) == 10\n\n\n"
            "def test_summary():\n    assert summary([(2, 3)]) == \"total=6\"\n"),
    },
    hidden=(
        "import pathlib\n"
        "from orders import compute_total; from report import summary\n"
        "assert compute_total([(2, 3), (1, 4)]) == 10 and summary([(5, 1)]) == 'total=5'\n"
        "assert not any('calc_total' in p.read_text() for p in pathlib.Path('.').glob('*.py'))"),
    solution={
        "orders.py": "def compute_total(items):\n    return sum(price * qty for price, qty in items)\n",
        "report.py": (
            "from orders import compute_total\n\n\n"
            "def summary(items):\n    return f\"total={compute_total(items)}\"\n"),
        "test_orders.py": (
            "from orders import compute_total\nfrom report import summary\n\n\n"
            "def test_total():\n    assert compute_total([(2, 3), (1, 4)]) == 10\n\n\n"
            "def test_summary():\n    assert summary([(2, 3)]) == \"total=6\"\n"),
    },
)

# A small package the model has not seen: the loader splits on every comma, so quoted fields break.
UNFAMILIAR_REPO = Task(
    name="unfamiliar-package",
    prompt=(
        "This is a small inventory package I haven't looked at in a while. `python -m pytest` fails. "
        "Find out why and fix the problem, then run the tests."),
    files={
        "inventory/__init__.py": "from .loader import load_items\nfrom .stock import low_stock\n",
        "inventory/loader.py": (
            "from .model import Item\n\n\n"
            "def parse_line(line):\n    return line.rstrip(\"\\n\").split(\",\")\n\n\n"
            "def load_items(text):\n"
            "    rows = [parse_line(line) for line in text.splitlines()[1:] if line.strip()]\n"
            "    return [Item(name=r[0], qty=int(r[1]), note=r[2]) for r in rows]\n"),
        "inventory/model.py": (
            "from dataclasses import dataclass\n\n\n"
            "@dataclass\nclass Item:\n    name: str\n    qty: int\n    note: str = \"\"\n"),
        "inventory/stock.py": "def low_stock(items, limit=5):\n    return [i.name for i in items if i.qty < limit]\n",
        "tests/test_inventory.py": (
            "from inventory import load_items, low_stock\n\n"
            "DATA = 'name,qty,note\\nbolt,3,\"small, steel\"\\nnut,10,plain\\n'\n\n\n"
            "def test_quoted_note():\n    items = load_items(DATA)\n    assert items[0].note == \"small, steel\"\n\n\n"
            "def test_low_stock():\n    assert low_stock(load_items(DATA)) == [\"bolt\"]\n"),
    },
    protected=["tests/test_inventory.py"],
    hidden=(
        "from inventory import load_items, low_stock\n"
        "d = 'name,qty,note\\nwasher,2,\"a, b, c\"\\nscrew,9,\"x\"\\ncap,1,\\n'\n"
        "i = load_items(d); assert [x.note for x in i] == ['a, b, c', 'x', ''] and low_stock(i) == ['washer', 'cap']"),
    solution={
        "inventory/loader.py": (
            "import csv\n\nfrom .model import Item\n\n\n"
            "def parse_line(line):\n    return next(csv.reader([line.rstrip(\"\\n\")]))\n\n\n"
            "def load_items(text):\n"
            "    rows = [parse_line(line) for line in text.splitlines()[1:] if line.strip()]\n"
            "    return [Item(name=r[0], qty=int(r[1]), note=r[2]) for r in rows]\n"),
    },
)

TASKS: dict[str, Task] = {t.name: t for t in (FIX_ADD, FIX_ACROSS_FILES, ADD_FEATURE, RENAME_ACROSS, UNFAMILIAR_REPO)}
