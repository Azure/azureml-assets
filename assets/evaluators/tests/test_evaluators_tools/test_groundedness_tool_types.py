# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Groundedness Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.groundedness.evaluator._groundedness import (
    GroundednessEvaluator,
)


@pytest.mark.unittest
class TestGroundednessToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for GroundednessEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.

    Note: Groundedness uses simplify_messages for query/response and extracts
    context from file_search tool results. Only file_search is a supported tool.
    For non-file_search tools not in the unsupported list, the flow is called
    with context="<>" (no context marker).
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.MCP_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }
    #endregion

    evaluator_type = GroundednessEvaluator

    check_for_unsupported_tools = True
