# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Tool Input Accuracy Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from typing import List
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import (
    ToolInputAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolInputAccuracyToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for ToolInputAccuracyEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "tool_calls": data.LOCAL_CALLS_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.FILE_SEARCH_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "tool_calls": data.IMAGE_GEN_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MEMORY_SEARCH_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.KB_MCP_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MCP_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    #endregion

    evaluator_type = ToolInputAccuracyEvaluator

    check_for_unsupported_tools = True

    _additional_expected_field_suffixes = ["details"]

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for tool type evaluator tests."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ] + [f"{self._result_prefix}_{suffix}" for suffix in self._additional_expected_field_suffixes]
