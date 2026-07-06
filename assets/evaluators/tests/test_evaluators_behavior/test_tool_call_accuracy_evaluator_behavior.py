# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Call Accuracy Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import asyncio

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ToolDefinitionExtractionUnitTests,
    ToolDefinitionsValidatorUnitTests,
)
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
    _get_built_in_definition,
)
from ..common.evaluator_mock_config import create_mocked_evaluator


@pytest.mark.unittest
class TestToolCallAccuracyEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Call Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_QUERY,
        "tool_calls": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_QUERY,
        "tool_calls": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_QUERY,
        "tool_calls": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_QUERY,
        "tool_calls": data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_QUERY,
        "tool_calls": data.KB_MCP_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_QUERY,
        "tool_calls": data.MCP_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    # endregion

    evaluator_type = ToolCallAccuracyEvaluator

    # Restricted built-in tool types are accepted by the validator as of asset version 12 (formerly
    # rejected with NOT_APPLICABLE). Per-tool expected_flow_inputs for the newly-enabled tool types
    # are tracked in a follow-up PR; until they are captured the flow-mock arg matcher is relaxed
    # for tools with an empty expected_flow_inputs dict.
    check_for_unsupported_tools = False

    is_tool_definition_required = True

    MINIMAL_RESPONSE = BaseToolCallEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()


@pytest.mark.unittest
class TestToolCallAccuracyValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
    ToolDefinitionExtractionUnitTests,
):
    """Low-level unit tests for tool_call_accuracy's repeated validators, utils and methods."""

    evaluator_class = ToolCallAccuracyEvaluator


@pytest.mark.unittest
class TestToolCallAccuracyInternalBranches:
    """Cover tool_call_accuracy input-conversion and helper branches not hit elsewhere."""

    _VALID_TOOL_CALL = {
        "type": "tool_call",
        "tool_call_id": "c1",
        "name": "search",
        "arguments": {"q": "x"},
    }
    _SEARCH_DEFINITION = {
        "name": "search",
        "type": "function",
        "description": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }

    def test_get_built_in_definition_unknown_returns_none(self):
        """Return None when the tool name is not a known built-in."""
        assert _get_built_in_definition("not_a_builtin_tool") is None

    def test_validate_tool_calls_rejects_non_dict_item(self):
        """Flag a tool-call list whose items are not dictionaries."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = evaluator._validator._validate_tool_calls([123])
        assert isinstance(result, EvaluationException)

    def test_convert_kwargs_without_tool_calls_returns_error(self):
        """Return the no-tool-calls error message when none are supplied."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(query="q")
        assert result["error_message"] == evaluator._NO_TOOL_CALLS_MESSAGE

    def test_convert_kwargs_wraps_non_list_inputs(self):
        """Wrap dict-shaped tool calls and definitions into lists before extraction."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            tool_calls=dict(self._VALID_TOOL_CALL),
            tool_definitions=dict(self._SEARCH_DEFINITION),
        )
        assert isinstance(result, dict)

    def test_convert_kwargs_missing_definitions_when_none_provided(self):
        """Return the no-definitions error when extraction fails with an empty list."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", tool_calls=[dict(self._VALID_TOOL_CALL)], tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_convert_kwargs_missing_definitions_when_unmatched(self):
        """Return the missing-definitions error when a used tool has no definition."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            tool_calls=[dict(self._VALID_TOOL_CALL)],
            tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
        )
        assert result["error_message"] == evaluator._TOOL_DEFINITIONS_MISSING_MESSAGE

    def test_convert_kwargs_string_tool_calls_empty_definitions(self):
        """Return the no-definitions error when string tool calls resolve to no definitions."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", tool_calls="just a string", tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_do_eval_invalid_output_raises(self):
        """Raise when the judge returns a non-dict ``llm_output`` payload."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")

        async def _bad_flow(**kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = _bad_flow
        eval_input = {
            "query": "q",
            "tool_calls": [dict(self._VALID_TOOL_CALL)],
            "tool_definitions": [dict(self._SEARCH_DEFINITION)],
        }
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval(eval_input))

    def test_do_eval_missing_query_raises(self):
        """Raise when ``_do_eval`` is invoked without a query."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"tool_calls": [dict(self._VALID_TOOL_CALL)]}))

    def test_real_call_error_message_returns_not_applicable(self):
        """Return a not-applicable result when input conversion yields an error message."""
        evaluator = create_mocked_evaluator(ToolCallAccuracyEvaluator, "tool_call_accuracy")
        result = asyncio.run(
            evaluator._real_call(
                query="q",
                tool_calls=[dict(self._VALID_TOOL_CALL)],
                tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
            )
        )
        assert isinstance(result, dict)
