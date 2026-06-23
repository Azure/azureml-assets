# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Task Efficiency Evaluator - realistic multi-turn coding agent trajectories.

Each test crafts a GitHub-Copilot-style coding agent transcript (using realistic tools
such as ``read_file``, ``grep_search``, ``list_dir``, ``replace_string_in_file``,
``run_in_terminal``, ``get_errors``) and asserts that the LLM judge places it in the
expected efficiency band.

Default threshold is 3, so:
- Scores 4-5 (highly / mostly efficient) -> ``pass``
- Score 3 (moderately efficient) -> borderline ``pass``
- Scores 1-2 (inefficient / highly inefficient) -> ``fail``
- No assistant tool calls -> ``skipped`` (short-circuit, no LLM call)
"""

import pytest

from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.task_efficiency.evaluator._task_efficiency import TaskEfficiencyEvaluator


# region Reusable tool definitions for a coding agent

CODING_AGENT_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or workspace-relative path"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search the workspace for a regex pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the entries in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "replace_string_in_file",
        "description": "Replace an exact string in a file with new text.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "run_in_terminal",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "get_errors",
        "description": "Return compile/lint errors for the given files.",
        "parameters": {
            "type": "object",
            "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
            "required": ["paths"],
        },
    },
]


def _tc(call_id: str, name: str, arguments: dict) -> dict:
    """Build an assistant tool_call content block."""
    return {"type": "tool_call", "tool_call_id": call_id, "name": name, "arguments": arguments}


def _tr(call_id: str, result) -> dict:
    """Build a tool message wrapping a tool_result."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [{"type": "tool_result", "tool_result": result}],
    }


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _assistant_calls(*calls) -> dict:
    return {"role": "assistant", "content": list(calls)}


# endregion


@pytest.mark.quality
class TestTaskEfficiencyEvaluatorQualityMultiTurn(BaseQualityEvaluatorRunner):
    """Multi-turn quality tests for Task Efficiency Evaluator.

    Realistic GitHub-Copilot-style coding agent sessions across the full 1-5 rubric.
    """

    evaluator_type = TaskEfficiencyEvaluator

    # ==================== PASS: Highly Efficient (expect 4-5) ====================

    def test_pass_highly_efficient_add_cli_flag(self) -> None:
        """PASS - direct, minimal trajectory: locate, read once, edit once, verify once.

        The agent adds a ``--verbose`` flag to a CLI. Each tool call yields new
        information or makes concrete progress; nothing is read twice, no command
        is repeated, no dead-end exploration. Expected score: 4 or 5.
        """
        self.run_quality_test(
            test_label="PASS-highly-efficient-add-cli-flag",
            expected=ExpectedResult.PASS,
            messages=[
                _user("Add a `--verbose` flag to our CLI in src/cli.py that enables debug logging."),
                _assistant_calls(
                    _tc("c1", "grep_search", {"pattern": "argparse.ArgumentParser", "path": "src/"}),
                ),
                _tr("c1", "src/cli.py:14: parser = argparse.ArgumentParser(description='myapp')"),
                _assistant_calls(
                    _tc("c2", "read_file", {"path": "src/cli.py", "start_line": 1, "end_line": 60}),
                ),
                _tr(
                    "c2",
                    (
                        "1  import argparse\n"
                        "2  import logging\n"
                        "...\n"
                        "14 parser = argparse.ArgumentParser(description='myapp')\n"
                        "15 parser.add_argument('--config', type=str, required=True)\n"
                        "16 args = parser.parse_args()\n"
                        "17 logging.basicConfig(level=logging.INFO)\n"
                    ),
                ),
                _assistant_calls(
                    _tc(
                        "c3",
                        "replace_string_in_file",
                        {
                            "path": "src/cli.py",
                            "old": "parser.add_argument('--config', type=str, required=True)\nargs = parser.parse_args()\nlogging.basicConfig(level=logging.INFO)",  # noqa: E501
                            "new": "parser.add_argument('--config', type=str, required=True)\nparser.add_argument('--verbose', action='store_true', help='Enable debug logging')\nargs = parser.parse_args()\nlogging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)",  # noqa: E501
                        },
                    ),
                ),
                _tr("c3", "ok"),
                _assistant_calls(
                    _tc("c4", "run_in_terminal", {"command": "pytest tests/test_cli.py -q"}),
                ),
                _tr("c4", "42 passed in 0.8s"),
                _assistant_text(
                    "Added the `--verbose` flag in src/cli.py. It toggles the root logger between INFO "
                    "and DEBUG. All 42 CLI tests pass."
                ),
            ],
            tool_definitions=CODING_AGENT_TOOL_DEFINITIONS,
        )

    def test_pass_mostly_efficient_fix_off_by_one(self) -> None:
        """PASS - mostly efficient: one acceptable post-edit verification read.

        The agent fixes an off-by-one bug in a pagination helper. It performs one
        extra read of the file after the edit to verify the diff, which is legitimate
        verification, not redundancy. Expected score: 4 (occasionally 5).
        """
        self.run_quality_test(
            test_label="PASS-mostly-efficient-fix-off-by-one",
            expected=ExpectedResult.PASS,
            messages=[
                _user("test_pagination is failing - the last page is missing one row. Fix it in src/pagination.py."),
                _assistant_calls(
                    _tc("c1", "run_in_terminal", {"command": "pytest tests/test_pagination.py -q"}),
                ),
                _tr(
                    "c1",
                    (
                        "FAILED tests/test_pagination.py::test_paginate_last_page\n"
                        "AssertionError: expected 5 items on last page, got 4\n"
                    ),
                ),
                _assistant_calls(
                    _tc("c2", "read_file", {"path": "src/pagination.py", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c2",
                    (
                        "def paginate(items, page_size):\n"
                        "    total = len(items)\n"
                        "    pages = total // page_size  # bug: drops the partial last page\n"
                        "    for i in range(pages):\n"
                        "        yield items[i * page_size:(i + 1) * page_size]\n"
                    ),
                ),
                _assistant_calls(
                    _tc(
                        "c3",
                        "replace_string_in_file",
                        {
                            "path": "src/pagination.py",
                            "old": "pages = total // page_size  # bug: drops the partial last page",
                            "new": "pages = (total + page_size - 1) // page_size  # ceil to include the partial last page",  # noqa: E501
                        },
                    ),
                ),
                _tr("c3", "ok"),
                _assistant_calls(
                    _tc("c4", "read_file", {"path": "src/pagination.py", "start_line": 1, "end_line": 10}),
                ),
                _tr(
                    "c4",
                    (
                        "def paginate(items, page_size):\n"
                        "    total = len(items)\n"
                        "    pages = (total + page_size - 1) // page_size  # ceil to include the partial last page\n"
                        "    for i in range(pages):\n"
                        "        yield items[i * page_size:(i + 1) * page_size]\n"
                    ),
                ),
                _assistant_calls(
                    _tc("c5", "run_in_terminal", {"command": "pytest tests/test_pagination.py -q"}),
                ),
                _tr("c5", "8 passed in 0.3s"),
                _assistant_text(
                    "Fixed the off-by-one by switching the page count to ceiling division. All 8 "
                    "pagination tests now pass."
                ),
            ],
            tool_definitions=CODING_AGENT_TOOL_DEFINITIONS,
        )

    # ==================== BORDERLINE: Moderately Efficient (expect ~3) ====================

    def test_borderline_moderately_efficient_add_method(self) -> None:
        """BORDERLINE - moderately efficient: a few redundant reads/searches, no loops.

        Agent adds a ``to_dict`` method to a ``User`` class. It re-reads the same
        file twice with no edit between, runs two nearly-identical greps that return
        the same hit, and lists ``src/`` twice. Roughly a third of steps are wasted.
        Expected score: 3 (sometimes 2 or 4 depending on judge).
        """
        self.run_quality_test(
            test_label="BORDERLINE-moderately-efficient-add-to_dict",
            expected=ExpectedResult.PASS_OR_FAIL,
            messages=[
                _user("Add a `to_dict()` method to the User class so we can JSON-serialize it."),
                _assistant_calls(
                    _tc("c1", "list_dir", {"path": "src/"}),
                ),
                _tr("c1", "models.py  views.py  utils.py  __init__.py"),
                _assistant_calls(
                    _tc("c2", "grep_search", {"pattern": "class User", "path": "src/"}),
                ),
                _tr("c2", "src/models.py:12: class User:"),
                _assistant_calls(
                    _tc("c3", "grep_search", {"pattern": "class User\\b", "path": "src/"}),
                ),
                _tr("c3", "src/models.py:12: class User:"),
                _assistant_calls(
                    _tc("c4", "list_dir", {"path": "src/"}),
                ),
                _tr("c4", "models.py  views.py  utils.py  __init__.py"),
                _assistant_calls(
                    _tc("c5", "read_file", {"path": "src/models.py", "start_line": 1, "end_line": 60}),
                ),
                _tr(
                    "c5",
                    (
                        "class User:\n"
                        "    def __init__(self, id, name, email):\n"
                        "        self.id = id\n"
                        "        self.name = name\n"
                        "        self.email = email\n"
                    ),
                ),
                _assistant_calls(
                    _tc("c6", "read_file", {"path": "src/models.py", "start_line": 1, "end_line": 60}),
                ),
                _tr(
                    "c6",
                    (
                        "class User:\n"
                        "    def __init__(self, id, name, email):\n"
                        "        self.id = id\n"
                        "        self.name = name\n"
                        "        self.email = email\n"
                    ),
                ),
                _assistant_calls(
                    _tc(
                        "c7",
                        "replace_string_in_file",
                        {
                            "path": "src/models.py",
                            "old": "        self.email = email\n",
                            "new": (
                                "        self.email = email\n\n"
                                "    def to_dict(self):\n"
                                "        return {'id': self.id, 'name': self.name, 'email': self.email}\n"
                            ),
                        },
                    ),
                ),
                _tr("c7", "ok"),
                _assistant_calls(
                    _tc("c8", "run_in_terminal", {"command": "pytest tests/test_models.py -q"}),
                ),
                _tr("c8", "15 passed in 0.2s"),
                _assistant_text("Added `to_dict()` to the User class. Tests pass."),
            ],
            tool_definitions=CODING_AGENT_TOOL_DEFINITIONS,
        )

    # ==================== FAIL: Inefficient (expect 1-2) ====================

    def test_fail_inefficient_repeated_reads_and_retries(self) -> None:
        """FAIL - inefficient: file read 4+ times, unchanged retries on failing tests.

        The agent's actual fix is one line. Around it, the agent reads ``auth.py``
        four separate times with no edit between, lists ``src/`` twice in immediate
        succession, and runs ``pytest`` twice unchanged after the same failure
        before doing any work. Roughly half the trajectory is removable. Expected
        score: 1 or 2.
        """
        self.run_quality_test(
            test_label="FAIL-inefficient-repeated-reads-and-retries",
            expected=ExpectedResult.FAIL,
            messages=[
                _user("test_login is failing - the user lookup returns None. Fix it in src/auth.py."),
                _assistant_calls(_tc("c1", "list_dir", {"path": "src/"})),
                _tr("c1", "auth.py  cli.py  models.py"),
                _assistant_calls(_tc("c2", "list_dir", {"path": "src/"})),
                _tr("c2", "auth.py  cli.py  models.py"),
                _assistant_calls(
                    _tc("c3", "read_file", {"path": "src/auth.py", "start_line": 1, "end_line": 80}),
                ),
                _tr(
                    "c3",
                    (
                        "def lookup_user(username):\n"
                        "    rows = db.query('SELECT * FROM users WHERE name = ?', username)\n"
                        "    if rows:\n"
                        "        return rows[0]\n"
                        "    return None\n"
                        "\n"
                        "def login(username, password):\n"
                        "    user = lookup_user(username.lower())  # bug: db stores names case-sensitively\n"
                        "    if user and check_password(user, password):\n"
                        "        return user\n"
                        "    return None\n"
                    ),
                ),
                _assistant_calls(
                    _tc("c4", "run_in_terminal", {"command": "pytest tests/test_auth.py -q"}),
                ),
                _tr("c4", "FAILED tests/test_auth.py::test_login - AssertionError: expected user, got None"),
                _assistant_calls(
                    _tc("c5", "run_in_terminal", {"command": "pytest tests/test_auth.py -q"}),
                ),
                _tr("c5", "FAILED tests/test_auth.py::test_login - AssertionError: expected user, got None"),
                _assistant_calls(
                    _tc("c6", "read_file", {"path": "src/auth.py", "start_line": 1, "end_line": 80}),
                ),
                _tr(
                    "c6",
                    (
                        "def lookup_user(username):\n"
                        "    rows = db.query('SELECT * FROM users WHERE name = ?', username)\n"
                        "    ...\n"
                        "def login(username, password):\n"
                        "    user = lookup_user(username.lower())\n"
                        "    ...\n"
                    ),
                ),
                _assistant_calls(
                    _tc("c7", "grep_search", {"pattern": "lookup_user", "path": "src/"}),
                ),
                _tr("c7", "src/auth.py:1: def lookup_user(username):\nsrc/auth.py:8: user = lookup_user(username.lower())"),  # noqa: E501
                _assistant_calls(
                    _tc("c8", "read_file", {"path": "src/auth.py", "start_line": 1, "end_line": 80}),
                ),
                _tr(
                    "c8",
                    (
                        "def lookup_user(username):\n"
                        "    rows = db.query('SELECT * FROM users WHERE name = ?', username)\n"
                        "    ...\n"
                        "def login(username, password):\n"
                        "    user = lookup_user(username.lower())\n"
                        "    ...\n"
                    ),
                ),
                _assistant_calls(
                    _tc("c9", "read_file", {"path": "src/auth.py", "start_line": 1, "end_line": 80}),
                ),
                _tr(
                    "c9",
                    (
                        "def lookup_user(username):\n"
                        "    ...\n"
                        "    user = lookup_user(username.lower())\n"
                        "    ...\n"
                    ),
                ),
                _assistant_calls(
                    _tc(
                        "c10",
                        "replace_string_in_file",
                        {
                            "path": "src/auth.py",
                            "old": "user = lookup_user(username.lower())  # bug: db stores names case-sensitively",
                            "new": "user = lookup_user(username)",
                        },
                    ),
                ),
                _tr("c10", "ok"),
                _assistant_calls(
                    _tc("c11", "run_in_terminal", {"command": "pytest tests/test_auth.py -q"}),
                ),
                _tr("c11", "12 passed in 0.4s"),
                _assistant_text("Removed the unintended `.lower()` so the lookup uses the original casing. Tests pass."),
            ],
            tool_definitions=CODING_AGENT_TOOL_DEFINITIONS,
        )

    def test_fail_highly_inefficient_thrashing(self) -> None:
        """FAIL - highly inefficient: severe thrashing, repeated identical failed retries, oscillation.

        The agent re-reads ``models.py`` six times with no edits, retries the same
        failing ``npm test`` command four times unchanged, and oscillates between
        ``grep "User"`` and ``grep "user"`` that return the same hit. The final fix
        is a one-line change identifiable after the first read. Expected score: 1.
        """
        self.run_quality_test(
            test_label="FAIL-highly-inefficient-thrashing",
            expected=ExpectedResult.FAIL,
            messages=[
                _user("Fix the failing User test in src/models.js - it's throwing a TypeError."),
                _assistant_calls(
                    _tc("c1", "read_file", {"path": "src/models.js", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c1",
                    (
                        "class User {\n"
                        "  constructor(id, name) {\n"
                        "    this.id = id;\n"
                        "    this.name = name;\n"
                        "  }\n"
                        "  greet() {\n"
                        "    return 'Hello, ' + this.Name;  // bug: should be this.name\n"
                        "  }\n"
                        "}\n"
                    ),
                ),
                _assistant_calls(_tc("c2", "run_in_terminal", {"command": "npm test"})),
                _tr("c2", "FAIL  models.test.js > greet > TypeError: Cannot read properties of undefined"),
                _assistant_calls(
                    _tc("c3", "read_file", {"path": "src/models.js", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c3",
                    (
                        "class User {\n"
                        "  constructor(id, name) { this.id = id; this.name = name; }\n"
                        "  greet() { return 'Hello, ' + this.Name; }\n"
                        "}\n"
                    ),
                ),
                _assistant_calls(_tc("c4", "run_in_terminal", {"command": "npm test"})),
                _tr("c4", "FAIL  models.test.js > greet > TypeError: Cannot read properties of undefined"),
                _assistant_calls(_tc("c5", "grep_search", {"pattern": "User", "path": "src/"})),
                _tr("c5", "src/models.js:1: class User {"),
                _assistant_calls(
                    _tc("c6", "read_file", {"path": "src/models.js", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c6",
                    (
                        "class User {\n"
                        "  constructor(id, name) { this.id = id; this.name = name; }\n"
                        "  greet() { return 'Hello, ' + this.Name; }\n"
                        "}\n"
                    ),
                ),
                _assistant_calls(_tc("c7", "grep_search", {"pattern": "user", "path": "src/"})),
                _tr("c7", "src/models.js:1: class User {"),
                _assistant_calls(_tc("c8", "run_in_terminal", {"command": "npm test"})),
                _tr("c8", "FAIL  models.test.js > greet > TypeError: Cannot read properties of undefined"),
                _assistant_calls(
                    _tc("c9", "read_file", {"path": "src/models.js", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c9",
                    (
                        "class User {\n"
                        "  constructor(id, name) { this.id = id; this.name = name; }\n"
                        "  greet() { return 'Hello, ' + this.Name; }\n"
                        "}\n"
                    ),
                ),
                _assistant_calls(_tc("c10", "grep_search", {"pattern": "User", "path": "src/"})),
                _tr("c10", "src/models.js:1: class User {"),
                _assistant_calls(
                    _tc("c11", "read_file", {"path": "src/models.js", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c11",
                    (
                        "class User {\n"
                        "  constructor(id, name) { this.id = id; this.name = name; }\n"
                        "  greet() { return 'Hello, ' + this.Name; }\n"
                        "}\n"
                    ),
                ),
                _assistant_calls(_tc("c12", "run_in_terminal", {"command": "npm test"})),
                _tr("c12", "FAIL  models.test.js > greet > TypeError: Cannot read properties of undefined"),
                _assistant_calls(
                    _tc("c13", "read_file", {"path": "src/models.js", "start_line": 1, "end_line": 40}),
                ),
                _tr(
                    "c13",
                    (
                        "class User {\n"
                        "  constructor(id, name) { this.id = id; this.name = name; }\n"
                        "  greet() { return 'Hello, ' + this.Name; }\n"
                        "}\n"
                    ),
                ),
                _assistant_calls(
                    _tc(
                        "c14",
                        "replace_string_in_file",
                        {
                            "path": "src/models.js",
                            "old": "return 'Hello, ' + this.Name;",
                            "new": "return 'Hello, ' + this.name;",
                        },
                    ),
                ),
                _tr("c14", "ok"),
                _assistant_calls(_tc("c15", "run_in_terminal", {"command": "npm test"})),
                _tr("c15", "PASS  models.test.js (1 test)"),
                _assistant_text("Fixed - `this.Name` should have been `this.name`. Tests pass."),
            ],
            tool_definitions=CODING_AGENT_TOOL_DEFINITIONS,
        )

    # ==================== SKIPPED: No tool-use trajectory ====================

    def test_skipped_no_tool_calls(self) -> None:
        """SKIPPED - a text-only chat has no trajectory to score for efficiency.

        Should short-circuit to ``not_applicable`` / ``skipped`` without making an
        LLM call.
        """
        self.run_quality_test(
            test_label="SKIPPED-no-tool-calls",
            expected=ExpectedResult.SKIPPED,
            messages=[
                _user("What's the difference between `is` and `==` in Python?"),
                _assistant_text(
                    "`is` checks identity (same object in memory); `==` checks equality (values compare "
                    "equal via `__eq__`). For small ints and interned strings they often coincide, but "
                    "you should generally use `==` for value comparison and `is` only for identity checks "
                    "such as `x is None`."
                ),
            ],
            tool_definitions=CODING_AGENT_TOOL_DEFINITIONS,
        )
