# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Coherence Evaluator — multi-turn conversations."""

import pytest
from typing import Any, Dict, List
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator


@pytest.mark.quality
class TestCoherenceEvaluatorQualityMultiTurn(BaseQualityEvaluatorRunner):
    """
    Multi-turn quality tests for Coherence Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    Each trace was validated across 7 judge models over 3 runs (see unit_test_suite.md).

    Coherence measures the logical and orderly presentation of ideas across
    multiple conversation turns, evaluating whether the agent maintains context,
    avoids contradictions, and builds on prior exchanges.

    Rating scale (default threshold=3):
    - 1: Incoherent — ignores context, contradicts itself
    - 3: Partially coherent — minor tangents but recovers
    - 5: Highly coherent — perfect logical flow across turns
    - null/skipped: Conversation not evaluable (e.g., user derails)
    """

    evaluator_type = CoherenceEvaluator

    @property
    def expected_result_fields(self) -> List[str]:
        """Coherence multi-turn _build_result omits _passed from output."""
        return [
            self._result_prefix,
            f"{self._result_prefix}_score",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_status",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_properties",
        ]

    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Derive ``passed`` from ``label`` since coherence multi-turn omits ``_passed``."""
        result = super()._extract_and_print_result(results, test_label)
        if result.get("passed") is None and result.get("label") in ("pass", "fail"):
            result["passed"] = result["label"] == "pass"
        return result

    def assert_not_applicable(self, result_data: Dict[str, Any]):
        """Coherence multi-turn skip reason comes from the LLM, not a fixed string."""
        assert result_data["label"] == "not_applicable", \
            f"Expected 'not_applicable' but got '{result_data['label']}'"
        assert result_data["passed"] is None, \
            f"Expected passed=None but got {result_data['passed']}"
        assert result_data["status"] == "skipped", \
            f"Expected status 'skipped' but got '{result_data['status']}'"
        assert result_data["score"] is None, \
            f"Expected score to be None for not-applicable result but got '{result_data['score']}'"

    # ==================== FAIL CASES ====================

    def test_fail_agent_ignores_context_contradicts(self) -> None:
        """Test case: FAIL — Agent ignores context, gives contradictory and unrelated responses.

        Trace: coherence_score1. The user asks for a vegetarian dinner plan,
        then the shopping list for mushroom risotto. The agent responds with
        programming advice, then steak ingredients — completely incoherent.
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-incoherent-ignores-context",
            expected=ExpectedResult.FAIL,
            messages=[
                {"role": "user", "content": "Help me plan a vegetarian dinner menu for 6 guests this Saturday."},
                {"role": "assistant", "content": (
                    "For a vegetarian dinner party, I'd suggest starting with a caprese salad, "
                    "followed by mushroom risotto as the main course, and tiramisu for dessert."
                )},
                {"role": "user", "content": (
                    "That sounds great. Can you give me a shopping list for the mushroom risotto?"
                )},
                {"role": "assistant", "content": (
                    "The best programming language for web development is JavaScript. You should "
                    "learn React and Node.js to build modern applications. Python is also useful "
                    "for backend work."
                )},
                {"role": "user", "content": "I asked about the risotto ingredients, not programming."},
                {"role": "assistant", "content": (
                    "Absolutely! For the grilled steak, you'll need 6 ribeye steaks, BBQ sauce, "
                    "and charcoal for the grill. I mentioned earlier that this is a meat-based "
                    "menu, so make sure to get the freshest cuts from your butcher."
                )},
            ],
        )

    # ==================== PASS CASES ====================

    @pytest.mark.flaky(reruns=3)
    def test_pass_minor_tangent_recovers(self) -> None:
        """Test case: PASS — Agent mostly tracks context with a minor irrelevant tangent.

        Trace: coherence_score3. PM interview prep conversation where the agent
        inserts a random sourdough bread comment mid-plan but otherwise stays on topic.
        Expected score 3–4 (at or above threshold).
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-minor-tangent-recovers",
            expected=ExpectedResult.PASS,
            messages=[
                {"role": "user", "content": (
                    "I'm preparing for a product manager interview next week. "
                    "What topics should I focus on?"
                )},
                {"role": "assistant", "content": (
                    "Great question! For a PM interview, focus on these core areas:\n"
                    "1. Product sense — identifying user problems and proposing solutions\n"
                    "2. Execution — prioritization frameworks like RICE or MoSCoW\n"
                    "3. Metrics — defining success metrics and KPIs\n"
                    "4. Technical depth — understanding system design at a high level"
                )},
                {"role": "user", "content": "Can you help me create a daily practice plan for the next 5 days?"},
                {"role": "assistant", "content": (
                    "Sure! Here's a 5-day plan:\n"
                    "- Day 1: Product sense — do 2 product critique exercises\n"
                    "- Day 2: Execution — practice prioritization with sample feature lists\n"
                    "- Day 3: You know, sourdough bread takes about 3 days to prepare from "
                    "starter. Anyway, for day 3, work on metrics — define north star metrics "
                    "for 3 sample products\n"
                    "- Day 4: Technical — walk through a system design question\n"
                    "- Day 5: Mock interview — combine all areas in a full practice session"
                )},
                {"role": "user", "content": "For Day 1, what's a good product critique exercise format?"},
                {"role": "assistant", "content": (
                    "For the product critique exercise on Day 1, use this format:\n"
                    "1. Pick an app you use daily (e.g., Spotify, Uber)\n"
                    "2. Identify one key user problem it solves\n"
                    "3. Evaluate: What does it do well? What could improve?\n"
                    "4. Propose one feature improvement with a clear user story\n"
                    "5. Define how you'd measure success\n\n"
                    "Spend 20 minutes per app. Do two apps to build pattern recognition."
                )},
            ],
        )

    def test_pass_perfect_logical_flow(self) -> None:
        """Test case: PASS — Perfect logical flow across turns (expected score 5).

        Trace: coherence_score5. Database migration plan where the agent tracks
        the conversation perfectly, building on each phase when asked for details.
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-perfect-logical-flow",
            expected=ExpectedResult.PASS,
            messages=[
                {"role": "user", "content": (
                    "I need to migrate our production database from MySQL 5.7 to PostgreSQL 15. "
                    "Can you outline the key phases?"
                )},
                {"role": "assistant", "content": (
                    "Here is a 4-phase migration plan:\n\n"
                    "Phase 1: Schema conversion — map MySQL types to PostgreSQL equivalents "
                    "(e.g., TINYINT to SMALLINT, AUTO_INCREMENT to SERIAL).\n"
                    "Phase 2: Data migration rehearsal — export with mysqldump, transform, and "
                    "load into a staging PostgreSQL instance.\n"
                    "Phase 3: Application code changes — update ORM configs, replace "
                    "MySQL-specific SQL syntax.\n"
                    "Phase 4: Cutover — final data sync, DNS switch, rollback plan."
                )},
                {"role": "user", "content": (
                    "Good. For Phase 2, what tools should we use for the data export and "
                    "transform step?"
                )},
                {"role": "assistant", "content": (
                    "For Phase 2, I recommend:\n\n"
                    "1. **pgloader** — handles the MySQL-to-PostgreSQL ETL in one step, "
                    "including type mapping and index recreation.\n"
                    "2. **AWS DMS** (if you're on AWS) — for continuous replication during the "
                    "rehearsal window.\n"
                    "3. For custom transforms, write Python scripts using `psycopg2` to handle "
                    "edge cases like ENUM columns or spatial data.\n\n"
                    "Start with pgloader for a baseline rehearsal, then layer in custom scripts "
                    "for any columns that need special handling."
                )},
                {"role": "user", "content": (
                    "What about Phase 3 — which parts of our Django app would need changes?"
                )},
                {"role": "assistant", "content": (
                    "For Phase 3, here are the Django-specific changes you'll need:\n\n"
                    "1. **settings.py** — switch `ENGINE` from `django.db.backends.mysql` to "
                    "`django.db.backends.postgresql`.\n"
                    "2. **Raw SQL queries** — audit for MySQL-specific syntax like backtick "
                    "quoting, `LIMIT x, y` (use `LIMIT y OFFSET x` instead), and "
                    "`GROUP_CONCAT` (use `STRING_AGG`).\n"
                    "3. **Migrations** — regenerate with `makemigrations` against the new "
                    "backend; check for any MySQL-specific field options.\n"
                    "4. **Full-text search** — if you use MySQL FULLTEXT indexes, switch to "
                    "PostgreSQL's `tsvector`/`tsquery` or "
                    "`django.contrib.postgres.search`.\n\n"
                    "I'd recommend running your test suite against the staging PostgreSQL from "
                    "Phase 2 to catch any remaining incompatibilities before cutover."
                )},
            ],
        )

    # ==================== SKIP CASES ====================

    def test_skip_user_derails_with_topic_jumps(self) -> None:
        """Test case: SKIP — User derails with unrelated topic jumps, no stable thread.

        Trace: coherence_skip. The user asks about a Python import error, then
        jumps to pizza, marathon training, and elephant jokes. The agent
        consistently tries to redirect. Expected: evaluation skipped.
        """
        self.run_quality_test(
            test_label="SKIP-multi-turn-user-derails-topic",
            expected=ExpectedResult.SKIPPED,
            messages=[
                {"role": "user", "content": (
                    "Help me fix this Python import error: "
                    "ModuleNotFoundError: No module named 'pandas'."
                )},
                {"role": "assistant", "content": (
                    "This error means the pandas package isn't installed. You can fix it by "
                    "running:\n```\npip install pandas\n```\nIf you're using a virtual "
                    "environment, make sure it's activated first. Can you share which Python "
                    "version and environment you're using?"
                )},
                {"role": "user", "content": "What's the best pizza place in Chicago?"},
                {"role": "assistant", "content": (
                    "I'm not able to help with restaurant recommendations. Would you like to "
                    "continue troubleshooting the Python import error? If so, please share "
                    "your Python version."
                )},
                {"role": "user", "content": "Actually, how do I train for a half marathon?"},
                {"role": "assistant", "content": (
                    "I'd recommend focusing on one topic at a time. We were working on your "
                    "Python import error. Would you like to continue with that, or start fresh "
                    "with a different question?"
                )},
                {"role": "user", "content": "Tell me a joke about elephants."},
                {"role": "assistant", "content": (
                    "I can help with that later! Let's resolve your original Python issue "
                    "first, or let me know if you'd like to switch topics entirely."
                )},
            ],
        )
