# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Navigation Efficiency Evaluator."""

import pytest
from typing import Any, Dict, List
from azure.ai.evaluation._exceptions import EvaluationException, ErrorCategory
from ...builtin.task_navigation_efficiency.evaluator._task_navigation_efficiency import (
    TaskNavigationEfficiencyEvaluator,
    TaskNavigationEfficiencyMatchingMode,
)


@pytest.mark.unittest
class TestTaskNavigationEfficiencyEvaluatorBehavior:
    """
    Behavioral tests for Task Navigation Efficiency Evaluator.

    Tests different input formats, matching modes, and scenarios.
    """

    evaluator_type = TaskNavigationEfficiencyEvaluator

    # region Test Data
    VALID_ACTIONS: List[Dict[str, Any]] = [
        # Allow extra non-tool-call messages
        {
            "role": "user",
            "content": "query",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "identify_tools_to_call",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "call_tool_A",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_3",
                    "name": "call_tool_B",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_4",
                    "name": "response_synthesis",
                    "arguments": {},
                }
            ],
        },
    ]

    VALID_EXPECTED_ACTIONS: List[str] = [
        "identify_tools_to_call",
        "call_tool_A",
        "call_tool_B",
        "response_synthesis",
    ]

    ACTIONS_WITH_EXTRA_STEP: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "identify_tools_to_call",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "extra_step",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_3",
                    "name": "call_tool_A",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_4",
                    "name": "call_tool_B",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_5",
                    "name": "response_synthesis",
                    "arguments": {},
                }
            ],
        },
    ]

    ACTIONS_OUT_OF_ORDER: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "call_tool_A",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "identify_tools_to_call",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_3",
                    "name": "call_tool_B",
                    "arguments": {},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_4",
                    "name": "response_synthesis",
                    "arguments": {},
                }
            ],
        },
    ]

    ACTIONS_WITH_PARAMS: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "search",
                    "arguments": {"query": "weather", "location": "NYC"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "format_result",
                    "arguments": {"format": "json"},
                }
            ],
        },
    ]

    EXPECTED_ACTIONS_WITH_PARAMS = (
        ["search", "format_result"],
        {
            "search": {"query": "weather", "location": "NYC"},
            "format_result": {"format": "json"},
        },
    )

    ACTIONS_WITH_WRONG_PARAMS: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "search",
                    "arguments": {"query": "weather", "location": "SF"},  # Wrong location
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "format_result",
                    "arguments": {"format": "json"},
                }
            ],
        },
    ]

    EMPTY_ACTIONS: List[Dict[str, Any]] = []
    INVALID_ACTIONS: List[Dict[str, Any]] = [{"role": "assistant", "content": "test"}]
    EMPTY_EXPECTED_ACTIONS: List[str] = []
    INVALID_EXPECTED_ACTIONS = {"not": "a list"}
    STRING_ACTIONS: str = "assistant used tools A and B"
    # endregion

    def _init_evaluator(
        self, matching_mode: TaskNavigationEfficiencyMatchingMode = TaskNavigationEfficiencyMatchingMode.EXACT_MATCH
    ) -> TaskNavigationEfficiencyEvaluator:
        """Create evaluator instance."""
        return TaskNavigationEfficiencyEvaluator(matching_mode=matching_mode)

    def _run_evaluation(
        self, actions: List[Dict[str, Any]], expected_actions, matching_mode: TaskNavigationEfficiencyMatchingMode
    ) -> Dict[str, Any]:
        """Run evaluation and return results."""
        evaluator = self._init_evaluator(matching_mode=matching_mode)

        try:
            results = evaluator(actions=actions, expected_actions=expected_actions)
            return results
        except EvaluationException as e:
            print(f"Error during evaluation: {e}")
            return {
                "task_navigation_efficiency_error_message": str(e),
                "task_navigation_efficiency_error_type": e.category.name,
            }
        except Exception as e:
            print(f"Unexpected error during evaluation: {e}")
            return {
                "task_navigation_efficiency_error_message": str(e),
                "task_navigation_efficiency_error_type": type(e).__name__,
            }

    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Extract result fields and print them."""
        label = results.get("task_navigation_efficiency_label")
        result = results.get("task_navigation_efficiency_result")
        details = results.get("task_navigation_efficiency_details")
        error_message = results.get("task_navigation_efficiency_error_message")
        error_code = results.get("task_navigation_efficiency_error_code")

        print(f"\n[{test_label}] Result: {result}")
        print(f"  Label: {label}")
        print(f"  Details: {details}")
        if error_message or error_code:
            print(f"  Error Message: {error_message}")
            print(f"  Error Code: {error_code}")

        return {
            "label": label,
            "result": result,
            "details": details,
            "error_message": error_message,
            "error_code": error_code,
        }

    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result."""
        assert result_data["result"] == "pass"
        assert result_data["label"] is True
        assert result_data["details"] is not None
        assert "precision_score" in result_data["details"]
        assert "recall_score" in result_data["details"]
        assert "f1_score" in result_data["details"]

    def assert_fail(self, result_data: Dict[str, Any]):
        """Assert a failing result."""
        assert result_data["result"] == "fail"
        assert result_data["label"] is False
        assert result_data["details"] is not None

    def assert_error(self, result_data: Dict[str, Any], error_code: str = None):
        """Assert an error result."""
        assert result_data["label"] is None
        assert result_data["result"] is None
        assert result_data["error_message"] is not None
        if error_code:
            assert result_data["error_code"] == error_code

    # ==================== EXACT MATCH MODE TESTS ====================

    def test_exact_match_perfect_match(self):
        """Test exact match with perfect sequence match."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Exact Match - Perfect Match")
        self.assert_pass(result_data)
        # Should have perfect precision, recall, and F1
        assert result_data["details"]["precision_score"] == 1.0
        assert result_data["details"]["recall_score"] == 1.0
        assert result_data["details"]["f1_score"] == 1.0

    def test_exact_match_with_extra_step(self):
        """Test exact match with extra step (should fail)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_WITH_EXTRA_STEP,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Exact Match - Extra Step")
        self.assert_fail(result_data)

    def test_exact_match_out_of_order(self):
        """Test exact match with out of order steps (should fail)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_OUT_OF_ORDER,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Exact Match - Out of Order")
        self.assert_fail(result_data)

    # ==================== IN ORDER MATCH MODE TESTS ====================

    def test_in_order_match_perfect_match(self):
        """Test in-order match with perfect sequence match."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.IN_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "In-Order Match - Perfect Match")
        self.assert_pass(result_data)

    def test_in_order_match_with_extra_step(self):
        """Test in-order match with extra step (should pass)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_WITH_EXTRA_STEP,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.IN_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "In-Order Match - Extra Step")
        self.assert_pass(result_data)

    def test_in_order_match_out_of_order(self):
        """Test in-order match with out of order steps (should fail)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_OUT_OF_ORDER,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.IN_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "In-Order Match - Out of Order")
        self.assert_fail(result_data)

    # ==================== ANY ORDER MATCH MODE TESTS ====================

    def test_any_order_match_perfect_match(self):
        """Test any-order match with perfect sequence match."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Any-Order Match - Perfect Match")
        self.assert_pass(result_data)

    def test_any_order_match_with_extra_step(self):
        """Test any-order match with extra step (should pass)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_WITH_EXTRA_STEP,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Any-Order Match - Extra Step")
        self.assert_pass(result_data)

    def test_any_order_match_out_of_order(self):
        """Test any-order match with out of order steps (should pass)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_OUT_OF_ORDER,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Any-Order Match - Out of Order")
        self.assert_pass(result_data)

    # ==================== PARAMETER MATCHING TESTS ====================

    def test_exact_match_with_params_matching(self):
        """Test exact match with parameter matching."""
        results = self._run_evaluation(
            actions=self.ACTIONS_WITH_PARAMS,
            expected_actions=self.EXPECTED_ACTIONS_WITH_PARAMS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Exact Match - Params Match")
        self.assert_pass(result_data)

    def test_exact_match_with_params_mismatch(self):
        """Test exact match with parameter mismatch (should fail)."""
        results = self._run_evaluation(
            actions=self.ACTIONS_WITH_WRONG_PARAMS,
            expected_actions=self.EXPECTED_ACTIONS_WITH_PARAMS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Exact Match - Params Mismatch")
        self.assert_fail(result_data)

    def test_any_order_match_with_params_matching(self):
        """Test any-order match with parameter matching."""
        results = self._run_evaluation(
            actions=self.ACTIONS_WITH_PARAMS,
            expected_actions=self.EXPECTED_ACTIONS_WITH_PARAMS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Any-Order Match - Params Match")
        self.assert_pass(result_data)

    # ==================== ERROR HANDLING TESTS ====================

    def test_empty_expected_actions(self):
        """Test with empty expected actions (should error)."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.EMPTY_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Empty Expected Actions")
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    def test_none_expected_actions(self):
        """Test with empty expected actions (should error)."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=None,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "None Expected Actions")
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    def test_invalid_expected_actions_type(self):
        """Test with invalid expected actions type (should error)."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.INVALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Invalid Expected Actions Type")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_string_expected_actions(self):
        """Test with string expected actions (should error)."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.STRING_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "String Expected Actions")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_empty_actions(self):
        """Test with empty actions list."""
        results = self._run_evaluation(
            actions=self.EMPTY_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Empty Actions")
        self.assert_fail(result_data)
        # Empty actions should have zero scores
        assert result_data["details"]["precision_score"] == 0.0
        assert result_data["details"]["recall_score"] == 0.0
        assert result_data["details"]["f1_score"] == 0.0

    def test_string_actions(self):
        """Test with actions as a string (should error)."""
        results = self._run_evaluation(
            actions=self.STRING_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "String Actions")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_none_actions(self):
        """Test with None actions (should error)."""
        results = self._run_evaluation(
            actions=None,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "None Actions")
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    def test_invalid_actions(self):
        """Test with actions that don't contain tool calls."""
        results = self._run_evaluation(
            actions=self.INVALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Invalid Actions")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_invalid_matching_mode_string(self):
        """Test with invalid matching mode string."""
        with pytest.raises(ValueError):
            self._init_evaluator(matching_mode="invalid_mode")

    def test_invalid_matching_mode_type(self):
        """Test with invalid matching mode type."""
        with pytest.raises(EvaluationException):
            self._init_evaluator(matching_mode=123)

    # ==================== PRECISION, RECALL, F1 TESTS ====================

    def test_metrics_partial_match(self):
        """Test precision, recall, F1 with partial match."""
        # Actions has 2 correct and 1 extra, expected has 3
        partial_actions = [
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "tool_call_id": "call_1", "name": "call_tool_A", "arguments": {}}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "tool_call_id": "call_2", "name": "call_tool_B", "arguments": {}}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "tool_call_id": "call_3", "name": "extra_tool", "arguments": {}}],
            },
        ]
        expected = ["call_tool_A", "call_tool_B", "call_tool_C"]

        results = self._run_evaluation(
            actions=partial_actions,
            expected_actions=expected,
            matching_mode=TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Metrics - Partial Match")

        # TP=2, FP=1, FN=1
        # Precision = 2/(2+1) = 0.666...
        # Recall = 2/(2+1) = 0.666...
        # F1 = 2*0.666*0.666/(0.666+0.666) = 0.666...
        assert 0.66 <= result_data["details"]["precision_score"] <= 0.67
        assert 0.66 <= result_data["details"]["recall_score"] <= 0.67
        assert 0.66 <= result_data["details"]["f1_score"] <= 0.67
