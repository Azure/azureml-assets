# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Adherence Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.task_adherence.evaluator._task_adherence import (
    TaskAdherenceEvaluator,
)


@pytest.mark.unittest
class TestTaskAdherenceEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Task Adherence Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
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
    # endregion

    evaluator_type = TaskAdherenceEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    _additional_expected_field_suffixes = ["details"]
