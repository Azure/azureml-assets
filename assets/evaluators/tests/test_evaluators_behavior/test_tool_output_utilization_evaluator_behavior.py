# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Output Utilization Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import (
    ToolOutputUtilizationEvaluator,
)


@pytest.mark.unittest
class TestToolOutputUtilizationEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Output Utilization Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    # TOU defines its own _get_agent_response that JSON-encodes dict/list tool_result
    # payloads via _stringify_tool_result (vs the SDK helper used by TA/TC which uses
    # Python repr). The three fixtures below therefore use the TOU-flavored response
    # constants.
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_TOU_EXPECTED_FLOW_RESPONSE,
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
        "tool_definitions": data.MCP_TOU_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    # Phase 2: azure_ai_search, sharepoint_grounding, and azure_fabric are now
    # accepted by the TOU validator. The base class branches to NOT_APPLICABLE for
    # these tests whenever check_for_unsupported_tools is True, so we override the
    # three tests below to assert PASS instead.
    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.AZURE_AI_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.SHAREPOINT_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FABRIC_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }
    # endregion

    evaluator_type = ToolOutputUtilizationEvaluator

    check_for_unsupported_tools = True

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.VALID_RESPONSE
    requires_tool_definitions = True

    # --- Phase 2 overrides: these three tools used to be unsupported but TOU now
    # accepts them, so we override the base-class tests (which still branch to
    # NOT_APPLICABLE when check_for_unsupported_tools is True) to assert PASS.

    def test_azure_ai_search(self):
        """Azure AI Search tool with azure_ai_search type - now supported in Phase 2."""
        self._run_tool_type_test(
            test_label="Azure AI Search",
            evaluation_inputs={
                "query": data.AZURE_AI_SEARCH_QUERY,
                "response": data.AZURE_AI_SEARCH_RESPONSE,
                "tool_definitions": data.AZURE_AI_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_azure_ai_search_expected_flow_inputs,
        )

    def test_sharepoint_grounding(self):
        """Test SharePoint grounding tool with sharepoint_grounding type - now supported in Phase 2."""
        self._run_tool_type_test(
            test_label="SharePoint Grounding",
            evaluation_inputs={
                "query": data.SHAREPOINT_QUERY,
                "response": data.SHAREPOINT_RESPONSE,
                "tool_definitions": data.SHAREPOINT_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_sharepoint_grounding_expected_flow_inputs,
        )

    def test_fabric_data_agent(self):
        """Fabric data agent tool with azure_fabric type - now supported in Phase 2."""
        self._run_tool_type_test(
            test_label="Fabric Data Agent",
            evaluation_inputs={
                "query": data.FABRIC_QUERY,
                "response": data.FABRIC_RESPONSE,
                "tool_definitions": data.FABRIC_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_fabric_data_agent_expected_flow_inputs,
        )
