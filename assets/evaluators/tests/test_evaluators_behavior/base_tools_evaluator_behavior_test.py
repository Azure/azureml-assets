# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of for tools evaluators.

Tests various input scenarios: query, response, and tool_definitions.
"""

import asyncio
import inspect
import json
from pathlib import Path

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from azure.ai.evaluation._legacy.prompty._prompty import AsyncPrompty

from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ..common.evaluator_mock_config import (
    create_mocked_evaluator,
    create_none_score_flow_side_effect,
    assert_none_score_result,
)


class BaseToolsEvaluatorBehaviorTest(BaseEvaluatorBehaviorTest):
    """
    Base class for tools evaluator behavioral tests with tool_definitions.

    Extends BaseEvaluatorBehaviorTest with tool definition support.
    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "ToolOutputUtilization")
    Subclasses may override:
    - absent_tool_definitions_assert_type: AssertType - expected outcome when tool definitions are absent, None
      or empty (PASS when the evaluator scores without them, SKIPPED when it skips the row as not applicable)
    - prompty_tool_definitions_optional: bool - True when the evaluator's Prompty renders without tool definitions,
      which enables the Prompty rendering tests
    - requires_query: bool - whether query is required
    - MINIMAL_RESPONSE: list - minimal valid response format for the evaluator
    - expected_result_fields: list - expected fields in the evaluation result
      tools evaluators
    """

    # Test Configs
    absent_tool_definitions_assert_type = BaseEvaluatorBehaviorTest.AssertType.PASS
    prompty_tool_definitions_optional = False

    # region Test Data
    # Tool definition test data
    VALID_TOOL_DEFINITIONS = [
        {
            "name": "fetch_weather",
            "description": "Fetches the weather information for the specified location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
        {
            "name": "send_email",
            "description": "Sends an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
        },
    ]

    INVALID_TOOL_DEFINITIONS = [
        {
            "fetch_weather": {
                "description": "Fetches the weather information for the specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        },
        {
            "send_email": {
                "description": "Sends an email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            },
        },
    ]

    INVALID_TOOL_DEFINITIONS_AS_STRING: str = json.dumps(INVALID_TOOL_DEFINITIONS)
    # endregion

    def run_messages_input_test(self):
        """Assert top-level messages reach the existing query/response path."""
        evaluator = create_mocked_evaluator(self.evaluator_type, self.result_key)
        captured_kwargs = {}
        evaluator._validator.validate_eval_input = lambda kwargs: None

        if hasattr(evaluator, "_the_super_real_call"):
            async def capture_real_call(**kwargs):
                captured_kwargs.update(kwargs)
                return {}

            evaluator._the_super_real_call = capture_real_call
        else:
            evaluator._return_not_applicable_result = lambda *args: {}

            def capture_conversion(**kwargs):
                captured_kwargs.update(kwargs)
                return {"error_message": "captured"}

            evaluator._convert_kwargs_to_eval_input = capture_conversion

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        asyncio.run(
            evaluator._real_call(
                messages=messages,
                tool_definitions=self.VALID_TOOL_DEFINITIONS,
            )
        )

        assert captured_kwargs["query"] == messages[:2]
        assert captured_kwargs["response"] == messages[2:]

        with pytest.raises(EvaluationException):
            asyncio.run(
                evaluator._real_call(
                    messages=[{"role": "assistant", "content": "answer"}],
                    tool_definitions=self.VALID_TOOL_DEFINITIONS,
                )
            )

    def run_skipped_llm_status_not_applicable_test(self):
        """Run a skipped-status flow output and assert a not-applicable result.

        Regression: when the LLM flow returns ``status='skipped'`` (score=None), the
        evaluator must return a standardized not-applicable result without crashing.
        """
        self._run_none_score_not_applicable_test(self.VALID_RESPONSE)

    def run_intermediate_response_not_applicable_test(self):
        """Run an intermediate (function_call) response and assert a not-applicable result.

        Regression: a response whose final assistant turn is an unresolved function_call
        must be treated as not-applicable rather than evaluated.
        """
        self._run_none_score_not_applicable_test(self.FUNCTION_CALL_ONLY_RESPONSE)

    def _run_none_score_not_applicable_test(self, response):
        """Mock the flow to a None/skipped score, run with ``response``, assert not-applicable.

        Shared by the skipped-status and intermediate-response regressions, which differ
        only in the ``response`` payload passed to the evaluator.
        """
        result = self._run_evaluation_with_flow_side_effect(
            create_none_score_flow_side_effect(),
            query=self.VALID_QUERY,
            response=response,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        assert_none_score_result(result, self.result_key)

    def test_util_tool_definitions_reach_flow_e2e(self):
        """End-to-end: needed tool definitions are extracted (built-in + provided) and reach the flow."""
        _, captured = self._run_and_capture_flow_input(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        flow_input_json = json.dumps(captured, default=str)
        assert (
            "fetch_weather" in flow_input_json or "send_email" in flow_input_json
        ), "tool definitions did not reach the flow"

    # ==================== TOOL DEFINITIONS TESTS ====================

    def run_tool_definitions_test(
        self, input_tool_definitions, description: str, assert_type: BaseEvaluatorBehaviorTest.AssertType
    ):
        """Test various tool definitions inputs."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=input_tool_definitions,
        )
        result_data = self._extract_and_print_result(results, description)

        expected_behavior = assert_type
        if assert_type == self.AssertType.MISSING_FIELD:
            expected_behavior = self.absent_tool_definitions_assert_type

        self.assert_expected_behavior(expected_behavior, result_data)

    def test_tool_definitions_not_present(self):
        """Tool definitions not present - scored or skipped as not applicable, never a missing field error."""
        self.run_tool_definitions_test(
            input_tool_definitions=None,
            description="Tool Definitions Not Present",
            assert_type=self.AssertType.MISSING_FIELD,
        )

    def test_tool_definitions_as_string(self):
        """Tool definitions as string - should pass."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.INVALID_TOOL_DEFINITIONS_AS_STRING,
            description="Tool Definitions String",
            assert_type=self.AssertType.PASS,
        )

    def test_tool_definitions_invalid_format_as_string(self):
        """Tool definitions as string - should pass."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.INVALID_TOOL_DEFINITIONS_AS_STRING,
            description="Tool Definitions Invalid Format as String",
            assert_type=self.AssertType.PASS,
        )

    def test_tool_definitions_invalid_format(self):
        """Tool definitions in invalid format - should raise invalid value error."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.INVALID_TOOL_DEFINITIONS,
            description="Tool Definitions Invalid Format",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_wrong_type(self):
        """Tool definitions in wrong type - should raise invalid value error."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.WRONG_TYPE,
            description="Tool Definitions Wrong Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_empty_list(self):
        """Tool definitions as empty list - scored or skipped as not applicable, never a missing field error."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.EMPTY_LIST,
            description="Tool Definitions Empty List",
            assert_type=self.AssertType.MISSING_FIELD,
        )

    # ==================== PROMPTY RENDERING TESTS ====================
    # The tests above mock the judge flow, so they never load the Prompty. A legacy Prompty input without a
    # ``default`` raises ``MissingRequiredInputError`` when it is absent, so these render the real Prompty to
    # guarantee that omitted tool definitions reach the judge instead of failing before the model call.
    _PROMPTY_MODEL = {
        "configuration": {
            "azure_endpoint": "https://example.openai.azure.com",
            "azure_deployment": "d",
            "api_key": "k",
        }
    }
    _PROMPTY_SENTINEL = "SENTINEL_TOOL_DEFINITION_PAYLOAD"

    def _render_prompty(self, **inputs) -> str:
        if not self.prompty_tool_definitions_optional:
            pytest.skip("Prompty tool definitions are not optional for this evaluator")
        prompty_path = Path(inspect.getfile(self.evaluator_type)).parent / self.evaluator_type._PROMPTY_FILE
        prompty = AsyncPrompty.load(source=prompty_path, model=self._PROMPTY_MODEL)
        messages = prompty.render(query="q", response="r", tool_calls=[{"name": "f"}], **inputs)
        return "\n".join(str(message.get("content")) for message in messages).replace("\r\n", "\n")

    def test_prompty_renders_provided_tool_definitions(self):
        """Provided tool definitions reach the rendered prompt."""
        assert self._PROMPTY_SENTINEL in self._render_prompty(tool_definitions=self._PROMPTY_SENTINEL)

    @pytest.mark.parametrize(
        "absent", [{}, {"tool_definitions": None}, {"tool_definitions": ""}, {"tool_definitions": []}]
    )
    def test_prompty_renders_without_tool_definitions(self, absent):
        """Absent, None, empty string and empty list definitions render a prompt instead of raising."""
        rendered = self._render_prompty(**absent)
        assert self._PROMPTY_SENTINEL not in rendered
        assert rendered.strip()

    def test_prompty_drops_tool_definitions_block_when_absent(self):
        """The tool definitions section is omitted from the prompt, not rendered empty."""
        with_definitions = self._render_prompty(tool_definitions=self._PROMPTY_SENTINEL)
        without_definitions = self._render_prompty()
        assert len(without_definitions) < len(with_definitions.replace(self._PROMPTY_SENTINEL, ""))

    # ==================== TOOL DEFINITIONS PARAMETER TESTS ====================
    def test_tool_definitions_missing_name_parameter(self):
        """Tool definitions missing 'name' parameter - should raise invalid value error."""
        modified_tool_definitions = self.remove_parameter_from_input(
            input_data=self.VALID_TOOL_DEFINITIONS, parameter_name="name"
        )
        self.run_tool_definitions_test(
            input_tool_definitions=modified_tool_definitions,
            description="Tool Definitions Missing 'name'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_missing_parameters_parameter(self):
        """Tool definitions missing 'parameters' parameter - should raise invalid value error."""
        modified_tool_definitions = self.remove_parameter_from_input(
            input_data=self.VALID_TOOL_DEFINITIONS, parameter_name="parameters"
        )
        self.run_tool_definitions_test(
            input_tool_definitions=modified_tool_definitions,
            description="Tool Definitions Missing 'parameters'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_invalid_parameters_type(self):
        """Tool definitions with invalid 'parameters' type - should raise invalid value error."""
        modified_tool_definitions = self.update_parameter_in_input(
            input_data=self.VALID_TOOL_DEFINITIONS, parameter_name="parameters", parameter_value=self.WRONG_TYPE
        )
        self.run_tool_definitions_test(
            input_tool_definitions=modified_tool_definitions,
            description="Tool Definitions Invalid 'parameters' Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )
