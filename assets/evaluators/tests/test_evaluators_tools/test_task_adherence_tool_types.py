# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Task Adherence Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from typing import List
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.task_adherence.evaluator._task_adherence import (
    TaskAdherenceEvaluator,
)


@pytest.mark.unittest
class TestTaskAdherenceToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for TaskAdherenceEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "system_message": "",
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_code_interpreter_expected_flow_inputs = {
        "system_message": "",
        "query": data.CODE_INTERPRETER_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_bing_grounding_expected_flow_inputs = {
        "system_message": "",
        "query": data.BING_GROUNDING_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_bing_custom_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_file_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_azure_ai_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "system_message": "",
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "system_message": "",
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_openapi_expected_flow_inputs = {
        "system_message": "",
        "query": data.OPENAPI_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_web_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.WEB_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_browser_automation_expected_flow_inputs = {
        "system_message": "",
        "query": data.BROWSER_AUTOMATION_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_image_generation_expected_flow_inputs = {
        "system_message": "",
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_memory_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_kb_mcp_expected_flow_inputs = {
        "system_message": "",
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_mcp_expected_flow_inputs = {
        "system_message": "",
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "response": data.MCP_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }
    #endregion

    evaluator_type = TaskAdherenceEvaluator

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
