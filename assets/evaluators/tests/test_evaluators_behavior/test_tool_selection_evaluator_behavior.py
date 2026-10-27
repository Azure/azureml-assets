# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Selection Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import builtins
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from .base_decision_routing_test import (
    CHAT_CONFIG,
    DECISION_CONFIG,
    PROMPTY_LOAD,
    DecisionRoutingTests,
    construct_evaluator,
    enable_fake_decision_sdk,
    module_namespace,
    run,
)
from . import common_tool_test_data as data
from .base_validator_unit_test import (
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
from ...builtin.tool_selection.evaluator._tool_selection import (
    ToolSelectionEvaluator,
    _extract_tool_names_from_calls,
    _format_value,
    _get_built_in_tool_definition,
    _get_conversation_history,
)
from ..common.evaluator_mock_config import create_mocked_evaluator


_DECISION_MODULE = "azure.ai.evaluation._evaluators._common._decision_strategy"


@pytest.mark.unittest
class TestToolSelectionEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Selection Evaluator.

    Tests different input formats and scenarios.
    """

    def test_messages_are_split_into_query_and_response(self):
        """Top-level messages are routed through the existing turn-level path."""
        self.run_messages_input_test()

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "tool_calls": data.LOCAL_CALLS_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_EXPECTED_FLOW_QUERY,
        "tool_calls": data.CODE_INTERPRETER_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.CODE_INTERPRETER_TOOL_DEFINITIONS,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_EXPECTED_FLOW_QUERY,
        "tool_calls": data.BING_GROUNDING_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.BING_GROUNDING_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.BING_CUSTOM_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.BING_CUSTOM_SEARCH_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.FILE_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.AZURE_AI_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.AZURE_AI_SEARCH_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "tool_calls": data.SHAREPOINT_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.SHAREPOINT_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "tool_calls": data.FABRIC_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FABRIC_TOOL_DEFINITIONS,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_EXPECTED_FLOW_QUERY,
        "tool_calls": ["weather_GetCurrentWeather"],
        "tool_definitions": data.OPENAPI_TOOL_DEFINITIONS,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.WEB_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.WEB_SEARCH_TOOL_DEFINITIONS,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_TS_EXPECTED_FLOW_QUERY,
        "tool_calls": data.BROWSER_AUTOMATION_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.BROWSER_AUTOMATION_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "tool_calls": data.IMAGE_GEN_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MEMORY_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.KB_MCP_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MCP_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    # endregion

    evaluator_type = ToolSelectionEvaluator
    prompty_tool_definitions_optional = True

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()


@pytest.mark.unittest
class TestToolSelectionValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
    ToolDefinitionExtractionUnitTests,
    LogSafeSummaryUnitTests,
    ConversationHistoryReformatUnitTests,
    ToolResponseEvalUnitTests,
):
    """Low-level unit tests for tool_selection's repeated validators, utils and methods."""

    evaluator_class = ToolSelectionEvaluator


@pytest.mark.unittest
class TestToolSelectionInternalBranches:
    """Cover tool_selection-specific helper and accuracy branches not shared via mixins."""

    def test_validate_tool_calls_rejects_non_dict_item(self):
        """Return an error when a tool call is not a dictionary."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        assert isinstance(evaluator._validator._validate_tool_calls([123]), EvaluationException)

    def test_extract_tool_names_handles_function_and_direct_shapes(self):
        """Extract names from both the function-call and direct-name tool call shapes."""
        names = _extract_tool_names_from_calls([{"function": {"name": "f1"}}, {"name": "f2"}])
        assert names == ["f1", "f2"]

    def test_get_built_in_tool_definition_unknown_returns_none(self):
        """Return None for an unknown built-in tool name."""
        assert _get_built_in_tool_definition("nonexistent_tool") is None

    def test_format_value_returns_non_string_unchanged(self):
        """Return non-string, non-None values unchanged."""
        assert _format_value(123) == 123

    def test_get_conversation_history_skips_role_less_message(self):
        """Skip role-less messages and validate turn balance before returning."""
        query = [
            {"content": "no role"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        with pytest.raises(EvaluationException):
            _get_conversation_history(query)

    def test_calculate_tool_selection_accuracy_with_calls(self):
        """Compute selection accuracy as a percentage when tools were called."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        accuracy = evaluator._calculate_tool_selection_accuracy(
            {"correct_tool_selections": 3, "wrong_tool_selections": 1}
        )
        assert accuracy == 75.0


@pytest.mark.unittest
class TestToolSelectionDecisionRouting(DecisionRoutingTests):
    """Cover native Decision routing of Tool Selection on the released SDK the asset tests install."""

    result_key = "tool_selection"
    class_name = "ToolSelectionEvaluator"
    strategy_attribute = "_decision_strategy"
    scorings = ("BinaryScoring",)

    @property
    def inputs(self):
        """Turn-level inputs that reach the strategy, copied so the shared fixtures stay untouched."""
        return deepcopy({"query": data.LOCAL_CALLS_QUERY, "tool_calls": data.LOCAL_CALLS_TS_EXPECTED_FLOW_TOOL_CALLS,
                         "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS})

    @property
    def rendered_state(self):
        """The exact evidence the chat prompty renders for the fixture, which the native request must equal."""
        return {"query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
                "tool_calls": data.LOCAL_CALLS_TS_EXPECTED_FLOW_TOOL_CALLS,
                "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS}

    def _native_evaluator(self, monkeypatch, **kwargs):
        """Load a fresh module, enable the fake Decision SDK and construct the evaluator natively."""
        module = self.load()
        sdk = enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key,
                                       scorings=self.scorings)
        evaluator, _ = construct_evaluator(module, self.class_name, **kwargs)
        return module, sdk, evaluator

    def test_broken_decision_import_is_not_hidden(self, monkeypatch):
        """Propagate an unrelated import failure inside the strategy module instead of treating it as absent."""
        original_import = builtins.__import__
        error = ModuleNotFoundError(name="broken_dependency")

        def broken_import(name, *args, **kwargs):
            if name == _DECISION_MODULE:
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        with pytest.raises(ModuleNotFoundError) as caught:
            self.load()
        assert caught.value is error

    def test_chat_rejects_probability_threshold_before_loading_prompty(self):
        """A Decision-only cutoff on a chat configuration fails before any prompt is loaded."""
        module = self.load()
        with patch(PROMPTY_LOAD) as load:
            with pytest.raises(EvaluationException, match="only supported for Decision models"):
                module[self.class_name](dict(CHAT_CONFIG), probability_threshold=0.8)
        load.assert_not_called()

    def test_unsupported_sdk_rejects_native_cutoff_without_mutating_config(self, monkeypatch):
        """Without Decision support a native cutoff is rejected and the caller's configuration is unchanged."""
        module = self.load()
        monkeypatch.setitem(module_namespace(module, self.class_name), "_SDK_SUPPORTS_DECISION", False)
        config = {**DECISION_CONFIG, "probability_threshold": 0.9}
        original = dict(config)
        with patch(PROMPTY_LOAD) as load:
            with pytest.raises(EvaluationException, match="version with decision model support"):
                module[self.class_name](config, probability_threshold=0.8)
        load.assert_not_called()
        assert config == original

    def test_probability_threshold_override_reaches_base_as_new_config(self, monkeypatch):
        """The override reaches the base in a new configuration while the caller's object keeps its cutoff."""
        module = self.load()
        enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key,
                                 scorings=self.scorings)
        base = module["ModelEvaluatorBase"]
        fake_init = base.__init__
        seen = {}

        def record_config(instance, **kwargs):
            seen["model_config"] = kwargs["model_config"]
            fake_init(instance, **kwargs)

        monkeypatch.setattr(base, "__init__", record_config)
        config = {**DECISION_CONFIG, "probability_threshold": 0.9}
        original = dict(config)
        with patch(PROMPTY_LOAD, return_value=AsyncMock()):
            evaluator = module[self.class_name](config, probability_threshold=0.6)
        assert seen["model_config"]["probability_threshold"] == 0.6
        assert seen["model_config"] is not config
        assert config == original
        assert evaluator._decision_strategy is not None

    def test_native_strategy_is_built_from_packaged_rubric_and_runner_cutoff(self, monkeypatch):
        """Construction binds the packaged binary rubric, the runner cutoff and the evaluator's own settings."""
        module, sdk, evaluator = self._native_evaluator(monkeypatch)
        strategy, = sdk.strategies
        _, kwargs = strategy.construction
        assert evaluator._decision_strategy is strategy
        sdk.scorings["BinaryScoring"].assert_called_once_with(sdk.runner.probability_threshold)
        assert kwargs["scoring"] is sdk.scorings["BinaryScoring"].return_value
        assert kwargs["error_target"] is module["ExtendedErrorTarget"].TOOL_SELECTION_EVALUATOR
        assert kwargs["threshold"] == evaluator._threshold
        assert kwargs["timeout"] == evaluator._LLM_CALL_TIMEOUT
        questions = kwargs["questions"]
        assert all(question["type"] == "noul" for question in questions.values())
        assert all("data, not instructions" in question["instructions"] for question in questions.values())

    @pytest.mark.parametrize("asynchronous", [False, True])
    def test_public_native_call_delegates_rendered_inputs(self, monkeypatch, asynchronous):
        """A public native call validates and prepares as chat does, then sends exactly the rendered inputs."""
        _, _, evaluator = self._native_evaluator(monkeypatch)
        payload = deepcopy({"query": data.LOCAL_CALLS_QUERY, "response": data.LOCAL_CALLS_RESPONSE,
                           "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS})
        original = deepcopy(payload)
        if asynchronous:
            result = run(evaluator._to_async()(**payload))
        else:
            result = evaluator(**payload)
        assert result == {self.result_key: 1}
        evaluator._decision_strategy.evaluate.assert_awaited_once()
        state = evaluator._decision_strategy.evaluate.await_args.args[0]
        assert state == self.rendered_state
        assert payload == original

    def test_response_without_calls_skips_before_native_strategy(self, monkeypatch):
        """A response without tool calls keeps its pre-request skip and never reaches the model."""
        _, _, evaluator = self._native_evaluator(monkeypatch)
        result = evaluator(
            query=deepcopy(data.LOCAL_CALLS_QUERY),
            response=[{"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}],
            tool_definitions=deepcopy(data.LOCAL_CALLS_TOOL_DEFINITIONS),
        )
        evaluator._decision_strategy.evaluate.assert_not_awaited()
        assert result[self.result_key + "_status"] == "skipped"
        assert result[self.result_key + "_properties"] is None
