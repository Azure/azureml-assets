# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Tool Call Success Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from typing import List
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.tool_call_success.evaluator._tool_call_success import (
    ToolCallSuccessEvaluator,
)


@pytest.mark.unittest
class TestToolCallSuccessToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for ToolCallSuccessEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.

    Note: ToolCallSuccess does NOT pass query to the flow (requires_query=False).
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "response": data.LOCAL_CALLS_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.LOCAL_CALLS_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "response": data.FILE_SEARCH_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.FILE_SEARCH_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "response": data.IMAGE_GEN_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.IMAGE_GEN_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "response": data.MEMORY_SEARCH_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.MEMORY_SEARCH_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "response": data.KB_MCP_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.KB_MCP_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "response": data.MCP_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.MCP_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }
    #endregion

    evaluator_type = ToolCallSuccessEvaluator

    check_for_unsupported_tools = True

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
