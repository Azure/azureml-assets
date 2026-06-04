# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Tests for None score handling in quality evaluators.

Validates the fix for the math.isnan(None) crash in _do_eval methods.
When _return_not_applicable_result returns score=None (for skipped evaluations),
the _do_eval method must handle it without crashing.

Bug: PR #5042 changed _return_not_applicable_result to return score=None instead
of score=threshold, and prompty templates to return {"status": "skipped", "score": null}.
The subsequent math.isnan(None) call raised TypeError: must be real number, not NoneType.

Fix: Added `_score is not None and math.isnan(_score)` guard before math.isnan() calls.
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Union, Any
import asyncio
import pytest

# Mock azure.ai.evaluation imports before loading any evaluator modules
_mock_modules = [
    'azure', 'azure.ai', 'azure.ai.evaluation',
    'azure.ai.evaluation._evaluators', 'azure.ai.evaluation._evaluators._common',
    'azure.ai.evaluation._constants', 'azure.ai.evaluation._common',
    'azure.ai.evaluation._common.utils', 'azure.ai.evaluation._common._experimental',
    'azure.ai.evaluation._model_configurations',
    'azure.ai.evaluation._user_agent',
]
for mod in _mock_modules:
    sys.modules[mod] = MagicMock()

# Set up constants the evaluators reference
sys.modules['azure.ai.evaluation._constants'].EVALUATION_PASS_FAIL_MAPPING = {True: "pass", False: "fail"}


# ---------------------------------------------------------------------------
# Test configuration for each evaluator
# ---------------------------------------------------------------------------
EVALUATOR_CONFIGS = [
    {
        "name": "groundedness",
        "module_file": "_groundedness.py",
        "evaluator_path": Path(__file__).parent.parent.parent / "groundedness" / "evaluator",
        "result_key": "groundedness",
        "isnan_sites": 2,  # _do_eval has 2 math.isnan checks
    },
    {
        "name": "coherence",
        "module_file": "_coherence.py",
        "evaluator_path": Path(__file__).parent.parent.parent / "coherence" / "evaluator",
        "result_key": "coherence",
        "isnan_sites": 1,
    },
    {
        "name": "fluency",
        "module_file": "_fluency.py",
        "evaluator_path": Path(__file__).parent.parent.parent / "fluency" / "evaluator",
        "result_key": "fluency",
        "isnan_sites": 1,
    },
    {
        "name": "retrieval",
        "module_file": "_retrieval.py",
        "evaluator_path": Path(__file__).parent.parent.parent / "retrieval" / "evaluator",
        "result_key": "retrieval",
        "isnan_sites": 1,
    },
    {
        "name": "similarity",
        "module_file": "_similarity.py",
        "evaluator_path": Path(__file__).parent.parent.parent / "similarity" / "evaluator",
        "result_key": "similarity",
        "isnan_sites": 1,
    },
]


# ---------------------------------------------------------------------------
# Core logic tests — test the exact fix pattern in isolation
# ---------------------------------------------------------------------------

class TestNoneScoreGuard:
    """Test the `_score is not None and math.isnan(_score)` guard pattern."""

    def test_none_score_does_not_crash(self):
        """None score (from _return_not_applicable_result) must not raise TypeError."""
        _score = None
        # This is the FIXED code path
        result = _score is not None and math.isnan(_score)
        assert result is False

    def test_none_score_without_guard_crashes(self):
        """Confirms the original bug: math.isnan(None) raises TypeError."""
        with pytest.raises(TypeError, match="must be real number, not NoneType"):
            math.isnan(None)

    def test_nan_score_detected(self):
        """NaN score (invalid LLM output) should be detected as True."""
        _score = float('nan')
        result = _score is not None and math.isnan(_score)
        assert result is True

    def test_normal_score_passes(self):
        """Normal numeric scores should pass through without issue."""
        for _score in [1.0, 2.5, 3.0, 4.0, 5.0]:
            result = _score is not None and math.isnan(_score)
            assert result is False

    def test_zero_score_passes(self):
        """Zero score should pass through."""
        _score = 0.0
        result = _score is not None and math.isnan(_score)
        assert result is False

    def test_negative_score_passes(self):
        """Negative score should pass through."""
        _score = -1.0
        result = _score is not None and math.isnan(_score)
        assert result is False

    def test_inf_score_passes(self):
        """Infinity score should not be flagged by isnan."""
        _score = float('inf')
        result = _score is not None and math.isnan(_score)
        assert result is False


# ---------------------------------------------------------------------------
# Source code verification — ensure the fix is present in all evaluator files
# ---------------------------------------------------------------------------

class TestFixPresenceInSource:
    """Verify the fix is present in all 5 evaluator source files."""

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_fix_present_in_evaluator(self, config):
        """Verify `_score is not None and math.isnan(_score)` exists in each evaluator."""
        source_file = config["evaluator_path"] / config["module_file"]
        assert source_file.exists(), f"Evaluator file not found: {source_file}"

        content = source_file.read_text(encoding="utf-8")

        fixed_pattern = "_score is not None and math.isnan(_score)"
        fixed_count = content.count(fixed_pattern)

        assert fixed_count == config["isnan_sites"], (
            f"{config['name']}: Expected {config['isnan_sites']} fixed isnan site(s), "
            f"found {fixed_count}"
        )

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_no_unguarded_isnan_calls(self, config):
        """Verify no unguarded math.isnan(_score) calls remain."""
        source_file = config["evaluator_path"] / config["module_file"]
        content = source_file.read_text(encoding="utf-8")

        # Count all math.isnan(_score) occurrences
        total_isnan = content.count("math.isnan(_score)")
        # Count guarded ones
        guarded_isnan = content.count("_score is not None and math.isnan(_score)")

        unguarded = total_isnan - guarded_isnan
        assert unguarded == 0, (
            f"{config['name']}: Found {unguarded} unguarded math.isnan(_score) call(s)"
        )

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_return_not_applicable_returns_none_score(self, config):
        """Verify _return_not_applicable_result sets score to None."""
        source_file = config["evaluator_path"] / config["module_file"]
        content = source_file.read_text(encoding="utf-8")

        # The method should have result_key mapped to None
        result_key = config["result_key"]
        assert f'"{result_key}": None' in content or f"'{result_key}': None" in content or \
               'f"{self._result_key}": None' in content, (
            f"{config['name']}: _return_not_applicable_result should set score to None"
        )


# ---------------------------------------------------------------------------
# Simulated _do_eval behavior — tests the full score-checking logic
# ---------------------------------------------------------------------------

class TestDoEvalScoreChecking:
    """Simulate the _do_eval score-checking logic for each evaluator."""

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_none_score_from_not_applicable_result(self, config):
        """Simulate _return_not_applicable_result returning None score — should not crash."""
        result_key = config["result_key"]
        # This is what _return_not_applicable_result returns
        result = {
            result_key: None,
            f"{result_key}_score": None,
            f"{result_key}_passed": None,
            f"{result_key}_result": "not_applicable",
            f"{result_key}_reason": "Not applicable: intermediate response",
            f"{result_key}_status": "skipped",
        }

        _score = result.get(result_key, 0)
        # This is the fixed code — must not crash
        assert not (_score is not None and math.isnan(_score))

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_nan_score_raises_correctly(self, config):
        """NaN score should trigger the error path."""
        result_key = config["result_key"]
        result = {result_key: float('nan')}

        _score = result.get(result_key, 0)
        # NaN should be detected
        assert _score is not None and math.isnan(_score)

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_valid_score_passes(self, config):
        """Valid numeric score should pass through without triggering error."""
        result_key = config["result_key"]
        result = {result_key: 4.0}

        _score = result.get(result_key, 0)
        assert not (_score is not None and math.isnan(_score))

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_missing_score_key_uses_default(self, config):
        """When result_key is missing, default of 0 is used — should not crash."""
        result = {}  # No score key present
        result_key = config["result_key"]

        _score = result.get(result_key, 0)
        assert _score == 0
        assert not (_score is not None and math.isnan(_score))

    @pytest.mark.parametrize("config", EVALUATOR_CONFIGS, ids=[c["name"] for c in EVALUATOR_CONFIGS])
    def test_explicit_none_in_result_dict(self, config):
        """Explicitly setting score=None in result dict — must not crash."""
        result_key = config["result_key"]
        result = {result_key: None}

        _score = result.get(result_key, 0)
        # None overrides the default of 0 when the key exists
        assert _score is None
        assert not (_score is not None and math.isnan(_score))
