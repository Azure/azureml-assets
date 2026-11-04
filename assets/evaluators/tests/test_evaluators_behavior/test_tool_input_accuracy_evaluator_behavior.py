# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Input Accuracy Evaluator."""

import builtins
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
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


_DECISION_MODULE = "azure.ai.evaluation._evaluators._common._decision_strategy"


@pytest.mark.unittest
class TestToolInputAccuracyEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Input Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    def test_messages_are_split_into_query_and_response(self):
        """Top-level messages are routed through the existing turn-level path."""
        self.run_messages_input_test()

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
    prompty_tool_definitions_optional = True

    # Restricted built-in tool types are accepted by the validator as of asset version 13 (formerly
    # rejected with NOT_APPLICABLE). Per-tool expected_flow_inputs for the newly-enabled tool types
    # are tracked in a follow-up PR; until they are captured the flow-mock arg matcher is relaxed
    # for tools with an empty expected_flow_inputs dict.
    check_for_unsupported_tools = False

    # Test Configs
    absent_tool_definitions_assert_type = BaseToolsEvaluatorBehaviorTest.AssertType.SKIPPED

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


@pytest.mark.unittest
class TestToolInputAccuracyDecisionRouting(DecisionRoutingTests):
    """Cover native Decision routing of Tool Input Accuracy on the released SDK the asset tests install."""

    result_key = "tool_input_accuracy"
    class_name = "ToolInputAccuracyEvaluator"
    strategy_attribute = "_decision_strategy"
    scorings = ("BinaryScoring",)

    @property
    def inputs(self):
        """Turn-level inputs that reach the strategy, copied so the shared fixtures stay untouched."""
        return deepcopy({"query": data.LOCAL_CALLS_QUERY, "tool_calls": data.LOCAL_CALLS_TIA_EXPECTED_FLOW_TOOL_CALLS,
                         "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS})

    @property
    def rendered_state(self):
        """The exact evidence the chat prompty renders for the fixture, which the native request must equal."""
        return {"query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
                "tool_calls": data.LOCAL_CALLS_TIA_EXPECTED_FLOW_TOOL_CALLS,
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
        assert kwargs["error_target"] is module["ExtendedErrorTarget"].TOOL_INPUT_ACCURACY_EVALUATOR
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
