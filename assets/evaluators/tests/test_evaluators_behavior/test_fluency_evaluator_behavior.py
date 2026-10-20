# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Fluency Evaluator."""

import builtins
import json
from unittest.mock import AsyncMock

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests
from .base_validator_unit_test import (
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from . import base_decision_routing_test as routing
from ...builtin.fluency.evaluator._fluency import FluencyEvaluator
from ..common.evaluator_mock_config import create_mocked_evaluator, run_none_score_not_applicable


@pytest.mark.unittest
class TestFluencyEvaluatorBehavior(BaseEvaluatorBehaviorTest, BaseToolEvaluationTest, _TurnLevelUtilE2ETests):
    """
    Behavioral tests for Fluency Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    # Fluency calls _extract_final_text_response() which extracts text-only content
    # from assistant messages, so expected inputs are the flattened text strings.
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

    def test_messages_split_at_latest_user_turn(self):
        """Messages input evaluates only the response after the latest user turn."""
        evaluator = create_mocked_evaluator(FluencyEvaluator, "fluency")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Earlier question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Earlier answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "Latest question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Latest answer"}]},
        ]

        evaluator(messages=messages)

        flow_input = evaluator._flow.call_args.kwargs
        assert flow_input["response"] == "Latest answer"

    # ==================== JSON RESPONSE SUPPORT TESTS ====================

    def test_plain_string_response_still_works(self):
        """A plain string response (not JSON) should keep working as before."""
        results = self._run_evaluation(response="It is sunny today.")
        result_data = self._extract_and_print_result(results, "Plain String Response")
        self.assert_pass_or_fail(result_data)

    def test_json_encoded_message_list_response_flattens_to_final_text(self):
        """A JSON-encoded list of chat messages is parsed and only the final assistant text reaches the flow.

        Fluency flattens the preprocessed conversation to plain assistant text before calling the
        flow, so tool calls/results must never leak into what the flow receives.
        """
        import json

        final_answer = "The current weather in Seattle is sunny with a high of 72 degrees."
        json_response = json.dumps(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "fetch_weather",
                            "arguments": {"location": "Seattle"},
                        }
                    ],
                },
                {
                    "tool_call_id": "call_1",
                    "role": "tool",
                    "content": [{"type": "tool_result", "tool_result": {"weather": "rainy, 40 degrees"}}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": final_answer}],
                },
            ]
        )
        _, captured = self._run_and_capture_flow_input(response=json_response)
        flow_input_json = json.dumps(captured, default=str)
        assert final_answer in flow_input_json, "final assistant text did not reach the flow"
        assert "rainy, 40 degrees" not in flow_input_json, "tool result content leaked into the flow"
        assert "fetch_weather" not in flow_input_json, "tool call content leaked into the flow"


# region None score handling tests

@pytest.mark.unittest
class TestFluencyNoneScoreHandling:
    """Tests for None score handling in _do_eval (math.isnan fix).

    When _return_not_applicable_result returns score=None, _do_eval must not
    crash on math.isnan(None).
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow should not raise TypeError."""
        run_none_score_not_applicable(FluencyEvaluator, "fluency", response="It is sunny today.")


# endregion


@pytest.mark.unittest
class TestFluencyValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
):
    """Low-level unit tests for fluency's repeated validators, utils and methods."""

    evaluator_class = FluencyEvaluator


# region Decision routing (released SDK)


@pytest.mark.unittest
class TestFluencyDecisionRouting(routing.DecisionRoutingTests):
    """Cover the native Decision branches on the released SDK the asset job installs."""

    result_key = "fluency"
    class_name = "FluencyEvaluator"
    inputs = {"response": "It is sunny today."}
    strategy_attribute = "_decision"

    def _native(self, monkeypatch):
        """Construct an evaluator that routes natively against mocked strategies and return it."""
        module = self.load()
        routing.enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key)
        evaluator, _ = routing.construct_evaluator(module, self.class_name)
        return evaluator

    def test_public_call_routes_through_the_shared_base_loop(self, monkeypatch):
        """A public native call reaches the SDK loop and sends exactly the response fluency.prompty renders."""
        evaluator = self._native(monkeypatch)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello there."}]},
        ]
        assert evaluator(messages=messages) == {"fluency": 1}
        evaluator._decision.evaluate.assert_awaited_once_with({"response": "Hello there."})

    def test_messages_without_a_user_turn_are_reported_as_user_error(self, monkeypatch):
        """A message split failure surfaces as an evaluation error on the chat path."""
        module = self.load()

        def fail(messages):
            raise ValueError("messages must contain at least one message with role 'user'.")

        monkeypatch.setitem(routing.module_namespace(module, self.class_name), "_split_messages_at_latest_user", fail)
        evaluator, _ = routing.construct_evaluator(module, self.class_name, config=routing.CHAT_CONFIG)
        with pytest.raises(EvaluationException, match="at least one message with role 'user'"):
            evaluator(messages=[{"role": "assistant", "content": "Hello"}])

    def test_chat_skip_delivered_as_text_keeps_usage_and_properties(self):
        """A skipped chat judgment delivered as JSON text keeps its properties and measured usage."""
        evaluator = routing.construct_evaluator(self.load(), self.class_name, config=routing.CHAT_CONFIG)[0]
        usage = {"input_token_count": 4, "output_token_count": 2, "total_token_count": 6, "model_id": "judge"}
        skipped = json.dumps({"status": "skipped", "reason": "No evidence", "properties": {"custom": True}})
        evaluator._flow = AsyncMock(return_value={"llm_output": skipped, **usage})
        result = evaluator(**self.inputs)
        assert result["fluency"] is None
        assert result["fluency_result"] == "not_applicable"
        assert result["fluency_total_tokens"] == 6
        assert result["fluency_properties"]["custom"] is True
        assert result["fluency_properties"]["model"] == "judge"
        fallback = evaluator._return_not_applicable_result("Empty", 3, {"llm_output": "not json", **usage})
        assert fallback["fluency_properties"]["total_tokens"] == 6
        assert "custom" not in fallback["fluency_properties"]

    def test_broken_decision_import_is_not_hidden(self, monkeypatch):
        """A dependency failure inside the strategy module propagates instead of reading as an older SDK."""
        original_import = builtins.__import__
        error = ModuleNotFoundError(name="broken_dependency")

        def broken_import(name, *args, **kwargs):
            if name == "azure.ai.evaluation._evaluators._common._decision_strategy":
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        with pytest.raises(ModuleNotFoundError) as caught:
            self.load()
        assert caught.value is error


# endregion
