# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Call Success Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from ...builtin.tool_call_success.evaluator._tool_call_success import (
    ToolCallSuccessEvaluator,
)


@pytest.mark.unittest
class TestToolCallSuccessEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Tool Call Success Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolCallSuccessEvaluator

    # Test Configs
    requires_query = False

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.tool_results_without_arguments

    def test_openapi_call_response(self):
        """openapi_call is in UNSUPPORTED_TOOLS and validated before intermediate detection — both return error."""
        # openapi_call-only response -> validation rejects before intermediate check
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.OPENAPI_CALL_ONLY_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "OpenAPI Call Only - Unsupported")
        self.assert_error(result_data, error_code="NOT_APPLICABLE")

        # Full openapi_call/openapi_call_output response -> validation rejects
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.OPENAPI_CALL_FULL_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "OpenAPI Call Full - Unsupported")
        self.assert_error(result_data, error_code="NOT_APPLICABLE")

