# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Intent Resolution Evaluator."""

import asyncio
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_evaluator_behavior_test import _TurnLevelUtilE2ETests
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ToolDefinitionsValidatorUnitTests,
)
from ..common.evaluator_mock_config import (
    create_mocked_evaluator,
    run_none_score_not_applicable,
    run_intermediate_response_not_applicable,
)
from ...builtin.intent_resolution.evaluator._intent_resolution import (
    IntentResolutionEvaluator,
)


@pytest.mark.unittest
class TestIntentResolutionEvaluatorBehavior(
    BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest, _TurnLevelUtilE2ETests
):
    """
    Behavioral tests for Intent Resolution Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
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
    # endregion

    evaluator_type = IntentResolutionEvaluator


# region Not-applicable handling tests (skipped score + intermediate response)

@pytest.mark.unittest
class TestIntentResolutionNotApplicableHandling:
    """Regression tests for skipped-score and intermediate-response handling in _do_eval."""

    def test_turn_level_none_score_returns_not_applicable(self):
        """A skipped (None) score from the flow yields a standardized not-applicable result."""
        run_none_score_not_applicable(
            IntentResolutionEvaluator,
            "intent_resolution",
            query="What is the capital of France?",
            response="Paris is the capital of France.",
        )

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        run_intermediate_response_not_applicable(
            IntentResolutionEvaluator,
            "intent_resolution",
            query="What is the capital of France?",
        )


# endregion


@pytest.mark.unittest
class TestIntentResolutionValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
):
    """Low-level unit tests for intent_resolution's repeated validators, utils and methods."""

    evaluator_class = IntentResolutionEvaluator


# region _do_eval override raise branches

@pytest.mark.unittest
class TestIntentResolutionDoEvalBranches:
    """Cover intent_resolution's override ``_do_eval`` missing-field and invalid-output raises."""

    def test_missing_query_and_response_raises(self):
        """Missing both query and response raises a MISSING_FIELD error."""
        evaluator = create_mocked_evaluator(IntentResolutionEvaluator, "intent_resolution")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({}))

    def test_non_dict_output_raises(self):
        """A non-dict flow output raises an invalid-output error."""
        evaluator = create_mocked_evaluator(IntentResolutionEvaluator, "intent_resolution")

        async def str_flow(timeout=None, **kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = MagicMock(side_effect=str_flow)
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": "q", "response": "r"}))
