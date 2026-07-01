# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Call Accuracy Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import pytest
from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import BaseValidatorUnitTest
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolCallAccuracyEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Call Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_QUERY,
        "tool_calls": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_QUERY,
        "tool_calls": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_QUERY,
        "tool_calls": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_QUERY,
        "tool_calls": data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_QUERY,
        "tool_calls": data.KB_MCP_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_QUERY,
        "tool_calls": data.MCP_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    # endregion

    evaluator_type = ToolCallAccuracyEvaluator

    # Restricted built-in tool types are accepted by the validator as of asset version 12 (formerly
    # rejected with NOT_APPLICABLE). Per-tool expected_flow_inputs for the newly-enabled tool types
    # are tracked in a follow-up PR; until they are captured the flow-mock arg matcher is relaxed
    # for tools with an empty expected_flow_inputs dict.
    check_for_unsupported_tools = False

    is_tool_definition_required = True

    MINIMAL_RESPONSE = BaseToolCallEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()


@pytest.mark.unittest
class TestToolCallAccuracyValidatorUnit(BaseValidatorUnitTest):
    """Low-level unit tests for tool_call_accuracy's repeated validators, utils and methods."""

    evaluator_class = ToolCallAccuracyEvaluator
