# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Intent Resolution Evaluator."""

import asyncio
import builtins
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_decision_routing_test import DecisionRoutingTests, construct_evaluator, enable_fake_decision_sdk
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

    def test_messages_are_split_into_query_and_response(self):
        """Top-level messages are routed through the existing turn-level path."""
        self.run_messages_input_test()

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


# endregion


# region Native Decision routing coverage on the released SDK

DECISION_STRATEGY_MODULE = "azure.ai.evaluation._evaluators._common._decision_strategy"


@pytest.mark.unittest
class TestIntentResolutionDecisionRouting(DecisionRoutingTests):
    """Cover the native Decision branches of intent_resolution without a Decision-capable SDK."""

    result_key = "intent_resolution"
    class_name = "IntentResolutionEvaluator"
    inputs = {"query": "What is the capital of France?", "response": "Paris is the capital of France."}
    strategy_attribute = "_decision"

    def test_broken_decision_import_is_not_hidden(self, monkeypatch):
        """A broken SDK dependency propagates instead of being read as a missing Decision capability."""
        original_import = builtins.__import__
        error = ModuleNotFoundError(name="broken_dependency")

        def broken_import(name, *args, **kwargs):
            if name == DECISION_STRATEGY_MODULE:
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        with pytest.raises(ModuleNotFoundError) as caught:
            self.load()
        assert caught.value is error

    def test_native_state_is_the_rendered_query_and_response(self, monkeypatch):
        """The native state carries exactly the prepared query and response; the prompt never renders tools."""
        module = self.load()
        enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key)
        evaluator, _ = construct_evaluator(module, self.class_name)
        result = evaluator(
            query=deepcopy(data.LOCAL_CALLS_QUERY),
            response=deepcopy(data.LOCAL_CALLS_RESPONSE),
            tool_definitions=deepcopy(data.LOCAL_CALLS_TOOL_DEFINITIONS),
        )
        assert result == {"intent_resolution": 1}
        state = evaluator._decision.evaluate.await_args.args[0]
        assert state == {
            "query": data.LOCAL_CALLS_IR_EXPECTED_FLOW_QUERY,
            "response": data.LOCAL_CALLS_IR_EXPECTED_FLOW_RESPONSE,
        }

    def test_messages_reach_the_native_state_as_the_split_turn(self, monkeypatch):
        """Top-level messages are split at the latest user turn before the native state is built."""
        module = self.load()
        enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key)
        evaluator, _ = construct_evaluator(module, self.class_name)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "First answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "Latest question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Latest answer"}]},
        ]
        assert evaluator(messages=deepcopy(messages)) == {"intent_resolution": 1}
        state = evaluator._decision.evaluate.await_args.args[0]
        assert set(state) == {"query", "response"}
        assert "Latest question" in state["query"] and "First answer" in state["query"]
        assert state["response"] == "Latest answer"

    def test_chat_skip_with_string_model_output_keeps_usage(self):
        """A skipped chat answer delivered as a JSON string keeps its properties and measured usage."""
        evaluator = create_mocked_evaluator(IntentResolutionEvaluator, "intent_resolution")
        usage = {"input_token_count": 4, "output_token_count": 2, "total_token_count": 6, "model_id": "chat"}
        parsed = evaluator._return_not_applicable_result(
            "No evidence", 3, {"llm_output": '{"properties": {"diagnostic": "empty"}}', **usage}
        )
        assert parsed["intent_resolution"] is None
        assert parsed["intent_resolution_status"] == "skipped"
        assert parsed["intent_resolution_properties"]["diagnostic"] == "empty"
        assert parsed["intent_resolution_properties"]["total_tokens"] == 6
        assert parsed["intent_resolution_total_tokens"] == 6
        malformed = evaluator._return_not_applicable_result("No evidence", 3, {"llm_output": "not json", **usage})
        assert "diagnostic" not in malformed["intent_resolution_properties"]
        assert malformed["intent_resolution_properties"]["total_tokens"] == 6


# endregion
