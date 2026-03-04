# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Coherence Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.coherence.evaluator._coherence import (
    CoherenceEvaluator,
)


@pytest.mark.unittest
class TestCoherenceToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for CoherenceEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.

    Note: Coherence uses _preprocess_messages (drops MCP approval, normalizes
    function_call types) on query/response lists. check_for_unsupported_tools=False
    so all tool types PASS. Flow receives preprocessed lists directly (no string
    conversion). Requires query.
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_QUERY,
        "response": data.LOCAL_CALLS_COHERENCE_EXPECTED_FLOW_RESPONSE,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_QUERY,
        "response": data.CODE_INTERPRETER_RESPONSE,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_QUERY,
        "response": data.BING_GROUNDING_RESPONSE,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_QUERY,
        "response": data.BING_CUSTOM_SEARCH_RESPONSE,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_QUERY,
        "response": data.FILE_SEARCH_RESPONSE,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_QUERY,
        "response": data.AZURE_AI_SEARCH_RESPONSE,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_QUERY,
        "response": data.SHAREPOINT_RESPONSE,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_QUERY,
        "response": data.FABRIC_RESPONSE,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_QUERY,
        "response": data.OPENAPI_RESPONSE,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_QUERY,
        "response": data.WEB_SEARCH_RESPONSE,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_QUERY,
        "response": data.BROWSER_AUTOMATION_RESPONSE,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_QUERY,
        "response": data.IMAGE_GEN_RESPONSE,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_QUERY,
        "response": data.MEMORY_SEARCH_RESPONSE,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_QUERY,
        "response": data.KB_MCP_TCS_EXPECTED_FLOW_RESPONSE,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_QUERY,
        "response": data.MCP_TCS_EXPECTED_FLOW_RESPONSE,
    }
    #endregion

    evaluator_type = CoherenceEvaluator
