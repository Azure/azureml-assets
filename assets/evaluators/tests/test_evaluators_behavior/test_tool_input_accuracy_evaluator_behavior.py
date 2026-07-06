# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Input Accuracy Evaluator."""

import asyncio
from unittest.mock import MagicMock

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    AgentResponseReformatUnitTests,
    ConversationHistoryReformatUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    LogSafeSummaryUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ToolDefinitionExtractionUnitTests,
    ToolDefinitionsValidatorUnitTests,
)
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import (
    ToolInputAccuracyEvaluator,
    _log_safe_summary,
    reformat_agent_response,
    reformat_conversation_history,
)
from ..common.evaluator_mock_config import create_mocked_evaluator


@pytest.mark.unittest
class TestToolInputAccuracyEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Input Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "tool_calls": data.LOCAL_CALLS_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.FILE_SEARCH_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "tool_calls": data.IMAGE_GEN_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MEMORY_SEARCH_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.KB_MCP_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MCP_TIA_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    # endregion

    evaluator_type = ToolInputAccuracyEvaluator

    # Restricted built-in tool types are accepted by the validator as of asset version 13 (formerly
    # rejected with NOT_APPLICABLE). Per-tool expected_flow_inputs for the newly-enabled tool types
    # are tracked in a follow-up PR; until they are captured the flow-mock arg matcher is relaxed
    # for tools with an empty expected_flow_inputs dict.
    check_for_unsupported_tools = False

    # Test Configs
    requires_tool_definitions = True

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.tool_calls_with_arguments

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()

    def test_zero_parameters_extraction_accuracy_is_100_percent(self):
        """Zero total parameters returns 100% accuracy without dividing by zero."""
        evaluator = self._init_evaluator()
        accuracy = evaluator._calculate_parameter_extraction_accuracy(
            {"total_parameters_passed": 0, "correct_parameters_passed": 0}
        )
        assert accuracy == 100.0

    def test_partial_parameters_extraction_accuracy(self):
        """Partial correct parameters yields the expected percentage."""
        evaluator = self._init_evaluator()
        accuracy = evaluator._calculate_parameter_extraction_accuracy(
            {"total_parameters_passed": 4, "correct_parameters_passed": 1}
        )
        assert accuracy == 25.0


@pytest.mark.unittest
class TestToolInputAccuracyValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
    ToolDefinitionExtractionUnitTests,
    AgentResponseReformatUnitTests,
    LogSafeSummaryUnitTests,
    ConversationHistoryReformatUnitTests,
):
    """Low-level unit tests for tool_input_accuracy's repeated validators, utils and methods."""

    evaluator_class = ToolInputAccuracyEvaluator


@pytest.mark.unittest
class TestToolInputAccuracyInternalBranches:
    """Cover tool_input_accuracy reformat, conversion and eval branches not hit elsewhere."""

    _TOOL_CALL_RESPONSE = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "c1",
                    "name": "search",
                    "arguments": {"q": "x"},
                }
            ],
        }
    ]
    _TEXT_RESPONSE = [
        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}
    ]

    def test_reformat_agent_response_empty_extraction_logs_and_falls_back(self):
        """Fall back to the original response (and log) when no agent text is extracted."""
        response = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = reformat_agent_response(response, logger=MagicMock())
        assert result == response

    def test_reformat_agent_response_parse_error_logs_and_falls_back(self):
        """Fall back to the original response (and log) when parsing raises."""
        response = [123]
        result = reformat_agent_response(response, logger=MagicMock())
        assert result == response

    def test_reformat_conversation_history_malformed_logs_and_falls_back(self):
        """Skip role-less messages and fall back to the raw query when history is malformed."""
        query = [{"content": "no role"}]
        result = reformat_conversation_history(query, logger=MagicMock(), include_tool_calls=True)
        assert result == query

    def test_reformat_conversation_history_formats_multi_turn(self):
        """Format a balanced multi-turn conversation into a readable string."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        result = reformat_conversation_history(query)
        assert "answer" in result

    def test_convert_kwargs_requires_response(self):
        """Return the response-required error when no response is supplied."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(query="q")
        assert "error_message" in result

    def test_convert_kwargs_no_tool_calls_in_response(self):
        """Return the no-tool-calls error when the response has no tool calls."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(query="q", response=self._TEXT_RESPONSE)
        assert result["error_message"] == evaluator._NO_TOOL_CALLS_MESSAGE

    def test_convert_kwargs_wraps_dict_definitions(self):
        """Wrap dict-shaped tool definitions when the response is a string."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            response="just a string response",
            tool_definitions={"name": "x", "type": "function", "parameters": {}},
        )
        assert isinstance(result, dict)

    def test_convert_kwargs_string_response_empty_definitions(self):
        """Return the no-definitions error when a string response has no definitions."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", response="just a string response", tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_convert_kwargs_extraction_fails_no_definitions(self):
        """Return the no-definitions error when extraction fails with an empty list."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", response=[dict(m) for m in self._TOOL_CALL_RESPONSE], tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_convert_kwargs_extraction_fails_unmatched_definitions(self):
        """Return the missing-definitions error when a used tool has no definition."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            response=[dict(m) for m in self._TOOL_CALL_RESPONSE],
            tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
        )
        assert result["error_message"] == evaluator._TOOL_DEFINITIONS_MISSING_MESSAGE

    def test_do_eval_missing_query_raises(self):
        """Raise when ``_do_eval`` is invoked without a query."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({}))

    def test_do_eval_invalid_output_raises(self):
        """Raise when the judge returns a non-dict ``llm_output`` payload."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")

        async def _bad_flow(**kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = _bad_flow
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": "q"}))

    def test_real_call_error_message_returns_not_applicable(self):
        """Return a not-applicable result when input conversion yields an error message."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = asyncio.run(
            evaluator._real_call(
                query="q",
                response=[dict(m) for m in self._TEXT_RESPONSE],
                tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
            )
        )
        assert result["tool_input_accuracy"] is None

    def test_log_safe_summary_handles_raising_object(self):
        """Return a safe placeholder when summarizing an object whose length raises."""
        class _Bad:
            def __len__(self):
                raise RuntimeError("boom")

        assert "summary unavailable" in _log_safe_summary(_Bad())
