# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Fluency Evaluator."""

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.fluency.evaluator._fluency import FluencyEvaluator


@pytest.mark.unittest
class TestFluencyEvaluatorBehavior(BaseEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Fluency Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    # Fluency calls reformat_agent_response() which extracts text-only content
    # from assistant messages, so expected inputs are the reformatted strings.
    test_function_tool_local_calls_expected_flow_inputs = {
        "response": data.LOCAL_CALLS_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_code_interpreter_expected_flow_inputs = {
        "response": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_bing_grounding_expected_flow_inputs = {
        "response": data.BING_GROUNDING_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "response": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_file_search_expected_flow_inputs = {
        "response": data.FILE_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "response": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "response": data.SHAREPOINT_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "response": data.FABRIC_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_openapi_expected_flow_inputs = {
        "response": data.OPENAPI_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_web_search_expected_flow_inputs = {
        "response": data.WEB_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_browser_automation_expected_flow_inputs = {
        "response": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_image_generation_expected_flow_inputs = {
        "response": data.IMAGE_GEN_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_memory_search_expected_flow_inputs = {
        "response": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_kb_mcp_expected_flow_inputs = {
        "response": data.KB_MCP_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_mcp_expected_flow_inputs = {
        "response": data.MCP_IR_EXPECTED_FLOW_RESPONSE,
    }
    # endregion

    evaluator_type = FluencyEvaluator

    # Test Configs
    requires_query = False
