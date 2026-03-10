# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Completion Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.task_completion.evaluator._task_completion import (
    TaskCompletionEvaluator,
)


@pytest.mark.unittest
class TestTaskCompletionEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Task Completion Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.CODE_INTERPRETER_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BING_GROUNDING_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.AZURE_AI_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.SHAREPOINT_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FABRIC_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.OPENAPI_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.WEB_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BROWSER_AUTOMATION_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.KB_MCP_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "response": data.MCP_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MCP_TC_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }
    # endregion

    evaluator_type = TaskCompletionEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    _additional_expected_field_suffixes = ["details"]
