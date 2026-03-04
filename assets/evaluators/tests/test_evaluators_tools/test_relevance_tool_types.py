# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Relevance Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from typing import List
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.relevance.evaluator._relevance import (
    RelevanceEvaluator,
)


@pytest.mark.unittest
class TestRelevanceToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for RelevanceEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.

    Note: Relevance uses reformat_conversation_history/reformat_agent_response
    to convert query/response to strings. check_for_unsupported_tools=False
    so all tool types PASS. Requires query.
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_IR_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_IR_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_IR_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_IR_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_IR_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_IR_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_IR_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_IR_EXPECTED_FLOW_QUERY,
        "response": data.MCP_IR_EXPECTED_FLOW_RESPONSE,
    }
    #endregion

    evaluator_type = RelevanceEvaluator

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for relevance evaluator."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]
