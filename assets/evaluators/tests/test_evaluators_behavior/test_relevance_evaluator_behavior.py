# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Relevance Evaluator."""

import asyncio

import pytest
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from ..common.evaluator_mock_config import (
    create_mocked_evaluator,
    run_none_score_not_applicable,
    run_intermediate_response_not_applicable,
)
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator


@pytest.mark.unittest
class TestRelevanceErrorCoverage:
    """Cover deterministic relevance error responses."""

    def test_do_eval_rejects_non_text_conversation_input(self):
        """Missing query and response fields produce a classified user error."""
        evaluator = create_mocked_evaluator(RelevanceEvaluator, "relevance")

        with pytest.raises(EvaluationException) as exc_info:
            asyncio.run(evaluator._do_eval({"messages": []}))

        assert exc_info.value.blame == ErrorBlame.USER_ERROR
        assert exc_info.value.category == ErrorCategory.INVALID_VALUE
        assert exc_info.value.target == ErrorTarget.CONVERSATION

    def test_do_eval_rejects_invalid_flow_output(self):
        """A non-dictionary flow result produces a classified system error."""
        evaluator = create_mocked_evaluator(RelevanceEvaluator, "relevance")

        async def invalid_flow(timeout, **kwargs):
            return {"llm_output": "invalid"}

        evaluator._flow = invalid_flow

        with pytest.raises(EvaluationException) as exc_info:
            asyncio.run(evaluator._do_eval({"query": "question", "response": "answer"}))

        assert exc_info.value.blame == ErrorBlame.SYSTEM_ERROR
        assert exc_info.value.category == ErrorCategory.FAILED_EXECUTION
        assert exc_info.value.target == ErrorTarget.RELEVANCE_EVALUATOR


@pytest.mark.unittest
class TestRelevanceEvaluatorBehavior(BaseEvaluatorBehaviorTest, BaseToolEvaluationTest, _TurnLevelUtilE2ETests):
    """
    Behavioral tests for Relevance Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_IR_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_IR_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_IR_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_IR_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_IR_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_IR_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_IR_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_IR_EXPECTED_FLOW_RESPONSE,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_IR_EXPECTED_FLOW_QUERY,
        "response": data.MCP_IR_EXPECTED_FLOW_RESPONSE,
    }
    # endregion

    evaluator_type = RelevanceEvaluator

    def test_messages_split_at_latest_user_turn(self):
        """Messages input sends only messages after the latest user turn as the response."""
        evaluator = create_mocked_evaluator(RelevanceEvaluator, "relevance")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Earlier question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "Latest question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Latest answer"}]},
        ]

        evaluator(messages=messages)

        flow_input = evaluator._flow.call_args.kwargs
        assert "Latest question" in str(flow_input["query"])
        assert "Latest answer" not in str(flow_input["query"])
        assert "Latest answer" in str(flow_input["response"])


# region Not-applicable handling tests (skipped score + intermediate response)

@pytest.mark.unittest
class TestRelevanceNotApplicableHandling:
    """Regression tests for skipped-score and intermediate-response handling in _do_eval."""

    def test_turn_level_none_score_returns_not_applicable(self):
        """A skipped (None) score from the flow yields a standardized not-applicable result."""
        run_none_score_not_applicable(
            RelevanceEvaluator,
            "relevance",
            query="What is the capital of France?",
            response="Paris is the capital of France.",
        )

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        run_intermediate_response_not_applicable(
            RelevanceEvaluator,
            "relevance",
            query="What is the capital of France?",
        )


# endregion


@pytest.mark.unittest
class TestRelevanceValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
):
    """Low-level unit tests for relevance's repeated validators, utils and methods."""

    evaluator_class = RelevanceEvaluator
