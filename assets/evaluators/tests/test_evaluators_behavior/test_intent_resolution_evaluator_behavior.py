# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Intent Resolution Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.intent_resolution.evaluator._intent_resolution import (
    IntentResolutionEvaluator,
)


@pytest.mark.unittest
class TestIntentResolutionEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Intent Resolution Evaluator.

    Tests different input formats and scenarios.
    """

    #region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_IR_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.CODE_INTERPRETER_TOOL_DEFINITIONS,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_IR_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BING_GROUNDING_TOOL_DEFINITIONS,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BING_CUSTOM_SEARCH_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.AZURE_AI_SEARCH_TOOL_DEFINITIONS,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_IR_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.SHAREPOINT_TOOL_DEFINITIONS,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_IR_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FABRIC_TOOL_DEFINITIONS,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_IR_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.OPENAPI_TOOL_DEFINITIONS,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.WEB_SEARCH_TOOL_DEFINITIONS,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BROWSER_AUTOMATION_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_IR_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_IR_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_IR_EXPECTED_FLOW_QUERY,
        "response": data.MCP_IR_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    #endregion

    evaluator_type = IntentResolutionEvaluator
