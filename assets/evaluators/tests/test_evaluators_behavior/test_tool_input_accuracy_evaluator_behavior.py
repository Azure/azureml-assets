# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Input Accuracy Evaluator."""

import pytest

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
    ToolResponseEvalUnitTests,
)
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import (
    ToolInputAccuracyEvaluator,
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
    ToolResponseEvalUnitTests,
):
    """Low-level unit tests for tool_input_accuracy's repeated validators, utils and methods."""

    evaluator_class = ToolInputAccuracyEvaluator


@pytest.mark.unittest
class TestToolInputAccuracyInternalBranches:
    """Cover the tool_input_accuracy-specific convert-kwargs branch not shared via mixins."""

    def test_convert_kwargs_requires_response(self):
        """Return the response-required error when no response is supplied."""
        evaluator = create_mocked_evaluator(ToolInputAccuracyEvaluator, "tool_input_accuracy")
        result = evaluator._convert_kwargs_to_eval_input(query="q")
        assert "error_message" in result
