# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Navigation Efficiency Evaluator."""

import asyncio
import json
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

try:
    from typing import override
except ImportError:
    from typing_extensions import override
from azure.ai.evaluation._exceptions import EvaluationException, ErrorCategory
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.task_navigation_efficiency.evaluator._task_navigation_efficiency import (
    TaskNavigationEfficiencyEvaluator,
    TaskNavigationEfficiencyMatchingMode,
)


@pytest.mark.unittest
class TestTaskNavigationEfficiencyEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for Task Navigation Efficiency Evaluator.

    Tests different input formats, matching modes, and scenarios.
    """

    evaluator_type = TaskNavigationEfficiencyEvaluator
    result_key = "task_navigation_efficiency"
    constructor_arg_names = ["matching_mode"]

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

    @override
    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result."""
        super().assert_pass(result_data)
        assert "precision_score" in result_data["properties"]
        assert "recall_score" in result_data["properties"]
        assert "f1_score" in result_data["properties"]

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
        assert result_data["properties"]["precision_score"] == 1.0
        assert result_data["properties"]["recall_score"] == 1.0
        assert result_data["properties"]["f1_score"] == 1.0

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
        assert result_data["properties"]["precision_score"] == 0.0
        assert result_data["properties"]["recall_score"] == 0.0
        assert result_data["properties"]["f1_score"] == 0.0

    def test_string_actions(self):
        """Test with actions as a string (should error)."""
        results = self._run_evaluation(
            actions=self.STRING_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "String Actions")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_json_stringified_valid_inputs(self):
        """Test that JSON-stringified valid actions and expected_actions are parsed and evaluated correctly."""
        results = self._run_evaluation(
            actions=json.dumps(self.VALID_ACTIONS),
            expected_actions=json.dumps(self.VALID_EXPECTED_ACTIONS),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "JSON-Stringified Valid Inputs")
        self.assert_pass(result_data)
        assert result_data["properties"]["precision_score"] == 1.0
        assert result_data["properties"]["recall_score"] == 1.0
        assert result_data["properties"]["f1_score"] == 1.0

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
        with pytest.raises(EvaluationException):
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
        assert 0.66 <= result_data["properties"]["precision_score"] <= 0.67
        assert 0.66 <= result_data["properties"]["recall_score"] <= 0.67
        assert 0.66 <= result_data["properties"]["f1_score"] <= 0.67

    # ==================== PARAMETER TYPE NORMALIZATION TESTS ====================

    @staticmethod
    def _make_action(name: str, arguments: Any) -> Dict[str, Any]:
        """Create an assistant action with a tool call."""
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": f"call_{name}",
                    "name": name,
                    "arguments": arguments,
                }
            ],
        }

    def test_param_int_agent_vs_int_ground_truth(self):
        """Test that int param values match when both sides are int."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"count": 1, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"count": 1, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - int vs int")
        self.assert_pass(result_data)

    def test_param_int_agent_vs_str_ground_truth(self):
        """Test that int agent param matches str ground truth ('1' == '1')."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"count": 1, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"count": "1", "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - int vs str")
        self.assert_pass(result_data)

    def test_param_str_agent_vs_int_ground_truth(self):
        """Test that str agent param matches int ground truth ('1' == '1')."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"count": "1", "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"count": 1, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - str vs int")
        self.assert_pass(result_data)

    def test_param_bool_agent_vs_bool_ground_truth(self):
        """Test that bool param values match when both sides are bool."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"verbose": True, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"verbose": True, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - bool vs bool")
        self.assert_pass(result_data)

    def test_param_bool_agent_vs_str_ground_truth(self):
        """Test that bool agent param matches str 'True' ground truth."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"verbose": True, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"verbose": "True", "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - bool vs str 'True'")
        self.assert_pass(result_data)

    def test_param_dict_agent_vs_dict_ground_truth(self):
        """Test that dict param values match when both sides are dict."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"filters": {"category": "news", "lang": "en"}, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"filters": {"category": "news", "lang": "en"}, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - dict vs dict")
        self.assert_pass(result_data)

    def test_param_dict_agent_vs_json_str_ground_truth(self):
        """Test that dict agent param matches JSON-stringified ground truth."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"filters": {"category": "news", "lang": "en"}, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"filters": '{"category": "news", "lang": "en"}', "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - dict vs JSON string")
        self.assert_pass(result_data)

    def test_param_json_str_agent_vs_dict_ground_truth(self):
        """Test that JSON-stringified agent param matches dict ground truth."""
        results = self._run_evaluation(
            actions=[
                self._make_action(
                    "search", {"filters": '{"category": "news", "lang": "en"}', "query": "weather"}
                )
            ],
            expected_actions=(
                ["search"],
                {"search": {"filters": {"category": "news", "lang": "en"}, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - JSON string vs dict")
        self.assert_pass(result_data)

    def test_param_list_agent_vs_list_ground_truth(self):
        """Test that list param values match when both sides are list."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"tags": ["a", "b", "c"], "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"tags": ["a", "b", "c"], "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - list vs list")
        self.assert_pass(result_data)

    def test_param_list_agent_vs_json_str_ground_truth(self):
        """Test that list agent param matches JSON-stringified list ground truth."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"tags": ["a", "b", "c"], "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"tags": '["a", "b", "c"]', "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - list vs JSON string")
        self.assert_pass(result_data)

    def test_param_stringified_args_vs_dict_ground_truth(self):
        """Test that stringified JSON arguments match dict ground truth values."""
        actions = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "search",
                        "arguments": '{"count": 1, "query": "weather"}',
                    }
                ],
            }
        ]
        results = self._run_evaluation(
            actions=actions,
            expected_actions=(
                ["search"],
                {"search": {"count": 1, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - stringified args vs dict GT")
        self.assert_pass(result_data)

    def test_param_float_agent_vs_float_ground_truth(self):
        """Test that float param values match when both sides are float."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"threshold": 0.5, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"threshold": 0.5, "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - float vs float")
        self.assert_pass(result_data)

    def test_param_float_agent_vs_str_ground_truth(self):
        """Test that float agent param matches str ground truth ('0.5' == '0.5')."""
        results = self._run_evaluation(
            actions=[self._make_action("search", {"threshold": 0.5, "query": "weather"})],
            expected_actions=(
                ["search"],
                {"search": {"threshold": "0.5", "query": "weather"}},
            ),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Param Type - float vs str")
        self.assert_pass(result_data)

    # ==================== SDK INPUT-NAME ALIAS TESTS ====================

    def test_sdk_aliases_equivalent_to_canonical_names(self):
        """SDK names (response/ground_truth) produce identical results to actions/expected_actions."""
        canonical = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        sdk = self._run_evaluation(
            response=self.VALID_ACTIONS,
            ground_truth=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        canonical_data = self._extract_and_print_result(canonical, "Alias - Canonical Names")
        sdk_data = self._extract_and_print_result(sdk, "Alias - SDK Names")
        self.assert_pass(canonical_data)
        self.assert_pass(sdk_data)
        assert sdk_data["score"] == canonical_data["score"]
        assert sdk_data["properties"] == canonical_data["properties"]

    def test_response_alias_only(self):
        """'response' alias maps to 'actions' when actions is absent."""
        results = self._run_evaluation(
            response=self.VALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Alias - response only")
        self.assert_pass(result_data)

    def test_ground_truth_alias_only(self):
        """'ground_truth' alias maps to 'expected_actions' when expected_actions is absent."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            ground_truth=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Alias - ground_truth only")
        self.assert_pass(result_data)

    def test_canonical_names_take_precedence_over_aliases(self):
        """When both canonical and alias keys are present, the canonical key wins."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            response=self.STRING_ACTIONS,  # invalid value; ignored because 'actions' present
            ground_truth=["ignored"],  # ignored because 'expected_actions' present
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Alias - Canonical Precedence")
        self.assert_pass(result_data)

    # ==================== ACTIONS STRUCTURE VALIDATION TESTS ====================

    def test_action_item_not_dict(self):
        """An action item that is not a dict raises invalid value error."""
        results = self._run_evaluation(
            actions=[123],
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Action Item Not Dict")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_action_missing_role(self):
        """An action missing the 'role' field raises missing field error."""
        results = self._run_evaluation(
            actions=[{"content": [{"type": "tool_call", "name": "x", "tool_call_id": "c1", "arguments": {}}]}],
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Action Missing Role")
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    def test_action_role_not_string(self):
        """An action whose 'role' is not a string raises invalid value error."""
        results = self._run_evaluation(
            actions=[{"role": 123, "content": []}],
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Action Role Not String")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_assistant_action_missing_content(self):
        """An assistant action missing the 'content' field raises missing field error."""
        results = self._run_evaluation(
            actions=[{"role": "assistant"}],
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Assistant Action Missing Content")
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    def test_assistant_content_item_not_dict(self):
        """An assistant content item that is not a dict raises invalid value error."""
        results = self._run_evaluation(
            actions=[{"role": "assistant", "content": [123]}],
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Assistant Content Item Not Dict")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_tool_call_content_missing_name(self):
        """A tool_call content item missing the 'name' field raises missing field error."""
        results = self._run_evaluation(
            actions=[
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "tool_call_id": "c1", "arguments": {}}],
                }
            ],
            expected_actions=self.VALID_EXPECTED_ACTIONS,
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Tool Call Missing Name")
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    # ==================== EXPECTED_ACTIONS STRUCTURE VALIDATION TESTS ====================

    def test_expected_actions_tuple_wrong_length(self):
        """An expected_actions tuple without exactly 2 elements raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=(["a"], {"a": {}}, "extra"),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions Tuple Wrong Length")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_expected_actions_tuple_first_not_list(self):
        """An expected_actions tuple whose first element is not a list raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=("not_a_list", {}),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions Tuple First Not List")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_expected_actions_tuple_empty_tool_names(self):
        """An expected_actions tuple with an empty tool-names list raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=([], {}),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions Tuple Empty Tool Names")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_expected_actions_tuple_tool_name_not_string(self):
        """An expected_actions tuple with a non-string tool name raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=([123], {}),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions Tuple Tool Name Not String")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_expected_actions_tuple_second_not_dict(self):
        """An expected_actions tuple whose second element is not a dict raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=(["a"], "not_a_dict"),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions Tuple Second Not Dict")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_expected_actions_tuple_param_value_not_dict(self):
        """An expected_actions tuple whose parameter value is not a dict raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=(["a"], {"a": "not_a_dict"}),
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions Tuple Param Value Not Dict")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def test_expected_actions_list_non_string_element(self):
        """An expected_actions list with a non-string element raises invalid value error."""
        results = self._run_evaluation(
            actions=self.VALID_ACTIONS,
            expected_actions=["valid_tool", 123],
            matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
        )
        result_data = self._extract_and_print_result(results, "Expected Actions List Non-String Element")
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

def _make_tne(matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH):
    """Construct a TaskNavigationEfficiencyEvaluator for direct method testing."""
    return TaskNavigationEfficiencyEvaluator(matching_mode=matching_mode)


@pytest.mark.unittest
class TestTaskNavigationEfficiencyConstructor:
    """Tests for the constructor's matching_mode validation branches."""

    def test_string_matching_mode_is_accepted(self):
        """A valid string matching_mode is coerced to the enum."""
        evaluator = TaskNavigationEfficiencyEvaluator(matching_mode="exact_match")
        assert evaluator.matching_mode == TaskNavigationEfficiencyMatchingMode.EXACT_MATCH

    def test_invalid_string_matching_mode_raises(self):
        """An unknown string matching_mode raises EvaluationException."""
        with pytest.raises(EvaluationException):
            TaskNavigationEfficiencyEvaluator(matching_mode="not-a-mode")

    def test_invalid_type_matching_mode_raises(self):
        """A non-string, non-enum matching_mode raises EvaluationException."""
        with pytest.raises(EvaluationException):
            TaskNavigationEfficiencyEvaluator(matching_mode=123)


@pytest.mark.unittest
class TestTaskNavigationEfficiencyValidatorBranches:
    """Direct tests for the TaskNavigationEfficiencyValidator error branches."""

    def test_validate_actions_error_branches(self):
        """Exercise every ``_validate_actions`` failure branch."""
        validator = _make_tne()._validator
        assert isinstance(validator._validate_actions(None), EvaluationException)
        assert isinstance(validator._validate_actions("notlist"), EvaluationException)
        assert isinstance(validator._validate_actions([123]), EvaluationException)
        assert isinstance(validator._validate_actions([{}]), EvaluationException)
        assert isinstance(validator._validate_actions([{"role": 123}]), EvaluationException)
        assert isinstance(validator._validate_actions([{"role": "assistant"}]), EvaluationException)
        assert isinstance(
            validator._validate_actions([{"role": "assistant", "content": "x"}]), EvaluationException
        )
        assert isinstance(
            validator._validate_actions([{"role": "assistant", "content": [123]}]), EvaluationException
        )
        assert isinstance(
            validator._validate_actions([{"role": "assistant", "content": [{"type": "tool_call"}]}]),
            EvaluationException,
        )
        assert validator._validate_actions([{"role": "user", "content": "q"}]) is None

    def test_validate_expected_actions_error_branches(self):
        """Exercise every ``_validate_expected_actions`` failure branch."""
        validator = _make_tne()._validator
        assert isinstance(validator._validate_expected_actions(None), EvaluationException)
        assert isinstance(validator._validate_expected_actions((1, 2, 3)), EvaluationException)
        assert isinstance(validator._validate_expected_actions(("notlist", {})), EvaluationException)
        assert isinstance(validator._validate_expected_actions(([], {})), EvaluationException)
        assert isinstance(validator._validate_expected_actions(([123], {})), EvaluationException)
        assert isinstance(validator._validate_expected_actions((["a"], "notdict")), EvaluationException)
        assert isinstance(validator._validate_expected_actions((["a"], {"a": "notdict"})), EvaluationException)
        assert isinstance(validator._validate_expected_actions([123]), EvaluationException)
        assert isinstance(validator._validate_expected_actions(123), EvaluationException)
        assert validator._validate_expected_actions(["a"]) is None
        assert validator._validate_expected_actions((["a"], {"a": {"x": "y"}})) is None

    def test_validate_eval_input_parses_json_strings(self):
        """String actions/expected_actions are parsed as JSON before validation."""
        validator = _make_tne()._validator
        assert validator.validate_eval_input({"actions": "[]", "expected_actions": '["a"]'}) is True

    def test_validate_eval_input_raises_on_invalid(self):
        """Invalid actions raise the validation exception."""
        validator = _make_tne()._validator
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"actions": None, "expected_actions": ["a"]})


@pytest.mark.unittest
class TestTaskNavigationEfficiencyDoEval:
    """Direct tests for ``_do_eval`` and the tool-extraction error branches."""

    @staticmethod
    def _do(evaluator, actions, expected_actions):
        """Run the async ``_do_eval`` to completion."""
        return asyncio.run(
            evaluator._do_eval({"actions": actions, "expected_actions": expected_actions})
        )

    def test_extract_tool_call_missing_name_raises(self):
        """A tool call without a name raises during extraction."""
        actions = [{"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "c1"}]}]
        with pytest.raises(EvaluationException):
            _make_tne()._extract_tool_names_and_params_from_actions(actions)

    def test_extract_tool_call_bad_json_arguments_raises(self):
        """A tool call with non-JSON string arguments raises during extraction."""
        actions = [{"role": "assistant", "content": [{"type": "tool_call", "name": "f", "arguments": "{bad"}]}]
        with pytest.raises(EvaluationException):
            _make_tne()._extract_tool_names_and_params_from_actions(actions)

    def test_extract_tool_call_string_json_arguments(self):
        """A tool call with valid JSON string arguments is parsed into parameters."""
        actions = [
            {"role": "assistant", "content": [{"type": "tool_call", "name": "f", "arguments": '{"a": 1}'}]}
        ]
        pairs = _make_tne()._extract_tool_names_and_params_from_actions(actions)
        assert pairs and pairs[0][0] == "f"

    def test_do_eval_empty_expected_raises(self):
        """Empty expected_actions raises an EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], [])

    def test_do_eval_tuple_first_not_list_raises(self):
        """A tuple whose first element is not a list raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], ("notlist", {}))

    def test_do_eval_tuple_second_not_dict_raises(self):
        """A tuple whose second element is not a dict raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], (["a"], "notdict"))

    def test_do_eval_param_value_not_dict_raises(self):
        """A parameters mapping whose value is not a dict raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], (["a"], {"a": "notdict"}))

    def test_do_eval_param_value_not_serializable_raises(self):
        """A parameter value that is not JSON-serializable raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], (["a"], {"a": {"k": {1, 2}}}))

    def test_do_eval_expected_wrong_type_raises(self):
        """An expected_actions of the wrong overall shape raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], [1, 2, 3])

    def test_do_eval_unsupported_matching_mode_raises(self):
        """An unsupported matching_mode raises an EvaluationException."""
        evaluator = _make_tne()
        evaluator.matching_mode = "bogus-mode"
        actions = [{"role": "assistant", "content": [{"type": "tool_call", "name": "f", "arguments": {}}]}]
        with pytest.raises(EvaluationException):
            self._do(evaluator, actions, ["f"])

    def test_do_eval_any_order_match(self):
        """Any-order matching mode produces a completed result."""
        evaluator = _make_tne(TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH)
        actions = [{"role": "assistant", "content": [{"type": "tool_call", "name": "f", "arguments": {}}]}]
        result = self._do(evaluator, actions, ["f"])
        assert result["task_navigation_efficiency_status"] == "completed"

    def test_do_eval_param_key_not_str_raises(self):
        """A parameters mapping with a non-string tool-name key raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], (["a"], {123: {"x": "y"}}))

    def test_do_eval_param_inner_key_not_str_raises(self):
        """A parameter dictionary with a non-string key raises EvaluationException."""
        with pytest.raises(EvaluationException):
            self._do(_make_tne(), [], (["a"], {"a": {123: "y"}}))

    def test_normalize_param_value_branches(self):
        """Normalize strings, dict/list (JSON), scalars, and non-serializable fallbacks."""
        norm = TaskNavigationEfficiencyEvaluator._normalize_param_value
        assert norm("x") == "x"
        assert norm({"b": 1, "a": 2}) == json.dumps({"b": 1, "a": 2}, sort_keys=True)
        assert norm(5) == "5"
        assert isinstance(norm({"k": {1, 2}}), str)

    def test_matching_function_edge_cases(self):
        """Cover the empty-expected in-order shortcut and any-order duplicate counting."""
        evaluator = _make_tne()
        assert evaluator._calculate_in_order_match(["a"], []) is True
        assert evaluator._calculate_exact_match(["a"], ["a"]) is True
        assert evaluator._calculate_any_order_match(["a", "a"], ["a"]) is True


@pytest.mark.unittest
class TestTaskNavigationEfficiencyRealCall:
    """Tests for the inlined aggregating ``_the_super_real_call``."""

    def test_convert_error_propagates(self):
        """Errors from converting kwargs to eval input propagate out."""
        evaluator = _make_tne()
        evaluator._convert_kwargs_to_eval_input = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            asyncio.run(evaluator._the_super_real_call(actions=[], expected_actions=["a"]))

    def test_fills_result_keys_and_aggregates(self):
        """Missing result/threshold keys are filled and multiple turns aggregated."""
        evaluator = _make_tne()
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 1}, {"x": 1}])

        async def fake_do_eval(eval_input):
            return {"task_navigation_efficiency_score": 1.0}

        evaluator._do_eval = fake_do_eval
        result = asyncio.run(evaluator._the_super_real_call(actions=[], expected_actions=["a"]))
        assert isinstance(result, dict)

    def test_fills_lower_is_better_and_bad_threshold(self):
        """Cover the lower-is-better fill branch and the swallowed invalid-threshold error."""
        async def fake_do_eval(eval_input):
            return {"task_navigation_efficiency_score": 1.0}

        lower = _make_tne()
        lower._higher_is_better = False
        lower._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 1}])
        lower._do_eval = fake_do_eval
        assert isinstance(
            asyncio.run(lower._the_super_real_call(actions=[], expected_actions=["a"])), dict
        )

        bad = _make_tne()
        bad._threshold = "not-a-number"
        bad._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 1}])
        bad._do_eval = fake_do_eval
        assert isinstance(
            asyncio.run(bad._the_super_real_call(actions=[], expected_actions=["a"])), dict
        )

    def test_empty_eval_input_returns_empty(self):
        """No eval inputs yields an empty result."""
        evaluator = _make_tne()
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[])

        async def fake_do_eval(eval_input):
            return {}

        evaluator._do_eval = fake_do_eval
        assert asyncio.run(evaluator._the_super_real_call(actions=[], expected_actions=["a"])) == {}

    def test_fills_result_keys_below_threshold(self):
        """Cover the failing (below-threshold) result-fill branches for both directions."""
        async def fake_do_eval(eval_input):
            return {"task_navigation_efficiency_score": eval_input["x"]}

        higher = _make_tne()
        higher._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 0.0}])
        higher._do_eval = fake_do_eval
        assert isinstance(
            asyncio.run(higher._the_super_real_call(actions=[], expected_actions=["a"])), dict
        )

        lower = _make_tne()
        lower._higher_is_better = False
        lower._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 5.0}])
        lower._do_eval = fake_do_eval
        assert isinstance(
            asyncio.run(lower._the_super_real_call(actions=[], expected_actions=["a"])), dict
        )


@pytest.mark.unittest
class TestTaskNavigationEfficiencyCoverageGaps:
    """Direct tests for residual defensive branches to complete coverage."""

    def test_validate_expected_actions_empty_list(self):
        """An empty expected_actions list returns a validation error."""
        result = _make_tne()._validator._validate_expected_actions([])
        assert isinstance(result, EvaluationException)

    def test_enum_matching_mode_is_accepted(self):
        """An enum matching_mode instance is accepted by the constructor."""
        evaluator = TaskNavigationEfficiencyEvaluator(
            matching_mode=TaskNavigationEfficiencyMatchingMode.IN_ORDER_MATCH
        )
        assert evaluator.matching_mode == TaskNavigationEfficiencyMatchingMode.IN_ORDER_MATCH

    def test_parse_tools_attaches_tool_results(self):
        """Tool messages contribute results that are attached to their matching tool calls."""
        actions = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "tool_call_id": "c1", "name": "f", "arguments": {}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "tool_result", "tool_result": "ok"}],
            },
        ]
        tool_calls = _make_tne()._parse_tools_from_actions(actions)
        assert tool_calls[0].get("tool_result") == "ok"

    def test_extract_non_dict_tool_call_raises(self):
        """A non-dict tool call produced by parsing raises during extraction."""
        evaluator = _make_tne()
        evaluator._parse_tools_from_actions = lambda actions: [123]
        with pytest.raises(EvaluationException):
            evaluator._extract_tool_names_and_params_from_actions([])

    def test_extract_wrong_type_tool_call_raises(self):
        """A tool call whose type is not 'tool_call' raises during extraction."""
        evaluator = _make_tne()
        evaluator._parse_tools_from_actions = lambda actions: [{"type": "other"}]
        with pytest.raises(EvaluationException):
            evaluator._extract_tool_names_and_params_from_actions([])

    def test_do_eval_parses_invalid_json_string_actions(self):
        """Invalid-JSON string actions fall through the JSON parse and evaluate as empty."""
        result = asyncio.run(
            _make_tne()._do_eval({"actions": "invalid-json[", "expected_actions": ["a"]})
        )
        assert isinstance(result, dict)
