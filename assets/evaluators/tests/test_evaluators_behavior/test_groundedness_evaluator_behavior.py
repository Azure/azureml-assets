# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Groundedness Evaluator."""

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.groundedness.evaluator._groundedness import (
    GroundednessEvaluator,
)


@pytest.mark.unittest
class TestGroundednessEvaluatorBehavior(BaseEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Groundedness Evaluator.

    Tests different input formats and scenarios.
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

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.weather_tool_result_and_assistant_response
