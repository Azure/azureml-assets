# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for testing evaluators with specific tool types (function, built-in, etc.).

Extends BasePromptyEvaluatorRunner with:
- Enhanced mocking that captures exact input sent to _flow
- Assertion helpers for verifying tool definitions, tool calls, and query format
- Convenience method for running tool-type-specific tests
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from azure.ai.evaluation._exceptions import EvaluationException

from .base_prompty_evaluator_runner import BasePromptyEvaluatorRunner
from .evaluator_mock_config import get_flow_side_effect_for_evaluator


class BaseToolTypeEvaluatorTest(BasePromptyEvaluatorRunner):
    """
    Base class for tool-type-specific evaluator tests with flow input capture.

    Extends BasePromptyEvaluatorRunner with enhanced mocking that records
    the exact kwargs sent to the evaluator's _flow method. This allows tests
    to assert both that the evaluator succeeds AND that it forwards the
    correct tool definitions, tool calls, and query to the LLM.

    Subclasses should set:
    - evaluator_type: The evaluator class to test
    """

    # We handle mocking ourselves to capture flow input
    use_mocking = False

    # Captured kwargs from the last _flow call
    _captured_flow_kwargs: List[Dict[str, Any]] = []

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for tool type evaluator tests."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
            f"{self._result_prefix}_details",
        ]

    def _run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run evaluation with enhanced mocking that captures _flow input.

        Overrides the base implementation to set up a capturing mock that
        records all kwargs sent to the evaluator's _flow method.

        Args:
            **kwargs: Keyword arguments. Args in constructor_arg_names go to
                      the constructor, remaining go to the evaluator call.

        Returns:
            Dictionary containing evaluation results.
        """
        # Split kwargs into constructor args vs call args
        constructor_kwargs = {}
        call_kwargs = {}
        for key, value in kwargs.items():
            if key in self.constructor_arg_names:
                constructor_kwargs[key] = value
            else:
                call_kwargs[key] = value

        evaluator = self._init_evaluator(**constructor_kwargs)

        # Set up capturing mock
        self._captured_flow_kwargs = []
        side_effect = get_flow_side_effect_for_evaluator(self.result_key)

        async def capturing_side_effect(timeout, **flow_kwargs):
            self._captured_flow_kwargs.append(flow_kwargs)
            return await side_effect(timeout, **flow_kwargs)

        evaluator._flow = MagicMock(side_effect=capturing_side_effect)

        if hasattr(evaluator, "_ensure_query_prompty_loaded"):
            evaluator._ensure_query_prompty_loaded = MagicMock()

        try:
            results = evaluator(**call_kwargs)
            return results
        except EvaluationException as e:
            print(f"Error during evaluation: {e}")
            return {
                f"{self.result_key}_error_message": e.message,
                f"{self.result_key}_error_code": e.category.name,
            }
        except Exception as e:
            print(f"Unexpected error during evaluation: {e}")
            return {
                f"{self.result_key}_error_message": str(e),
            }

    # ==================== FLOW INPUT ASSERTION HELPERS ====================

    def assert_flow_was_called(self):
        """Assert that _flow was called at least once."""
        assert len(self._captured_flow_kwargs) > 0, \
            "Expected _flow to be called, but it was never called."

    def assert_flow_call_count(self, expected_count: int):
        """Assert the number of times _flow was called."""
        assert len(self._captured_flow_kwargs) == expected_count, \
            f"Expected _flow to be called {expected_count} time(s), " \
            f"but it was called {len(self._captured_flow_kwargs)} time(s)."

    def get_flow_kwargs(self, call_index: int = 0) -> Dict[str, Any]:
        """Get the kwargs from a specific _flow call.

        Args:
            call_index: Index of the call (default 0 for first call).

        Returns:
            Dictionary of kwargs sent to _flow.
        """
        assert call_index < len(self._captured_flow_kwargs), \
            f"Call index {call_index} out of range (only {len(self._captured_flow_kwargs)} calls captured)."
        return self._captured_flow_kwargs[call_index]

    def assert_flow_received_query(self, expected_query, call_index: int = 0):
        """Assert that _flow received the expected query.

        Args:
            expected_query: Expected query value.
            call_index: Index of the _flow call to check.
        """
        flow_kwargs = self.get_flow_kwargs(call_index)
        actual_query = flow_kwargs.get("query")
        assert actual_query == expected_query, \
            f"Expected query:\n{expected_query}\n\nActual query:\n{actual_query}"

    def assert_flow_received_tool_calls(self, expected_tool_calls, call_index: int = 0):
        """Assert that _flow received the expected tool calls.

        Args:
            expected_tool_calls: Expected tool calls list.
            call_index: Index of the _flow call to check.
        """
        flow_kwargs = self.get_flow_kwargs(call_index)
        actual_tool_calls = flow_kwargs.get("tool_calls")
        assert actual_tool_calls == expected_tool_calls, \
            f"Expected tool_calls:\n{expected_tool_calls}\n\nActual tool_calls:\n{actual_tool_calls}"

    def assert_flow_received_tool_definitions(self, expected_tool_definitions, call_index: int = 0):
        """Assert that _flow received the expected tool definitions.

        Args:
            expected_tool_definitions: Expected tool definitions list.
            call_index: Index of the _flow call to check.
        """
        flow_kwargs = self.get_flow_kwargs(call_index)
        actual_tool_definitions = flow_kwargs.get("tool_definitions")
        assert actual_tool_definitions == expected_tool_definitions, (
            f"Expected tool_definitions:\n{expected_tool_definitions}"
            f"\n\nActual tool_definitions:\n{actual_tool_definitions}"
        )

    def assert_flow_tool_definitions_contain(
        self, expected_definition: Dict[str, Any], call_index: int = 0
    ):
        """Assert that _flow's tool_definitions contain a specific definition.

        Args:
            expected_definition: Tool definition dict that should be present.
            call_index: Index of the _flow call to check.
        """
        flow_kwargs = self.get_flow_kwargs(call_index)
        actual_tool_definitions = flow_kwargs.get("tool_definitions", [])
        assert expected_definition in actual_tool_definitions, \
            f"Expected tool definition not found in _flow input:\n" \
            f"Expected: {expected_definition}\n" \
            f"Actual definitions: {actual_tool_definitions}"

    def assert_flow_tool_definitions_count(self, expected_count: int, call_index: int = 0):
        """Assert the number of tool definitions sent to _flow.

        Args:
            expected_count: Expected number of tool definitions.
            call_index: Index of the _flow call to check.
        """
        flow_kwargs = self.get_flow_kwargs(call_index)
        actual_tool_definitions = flow_kwargs.get("tool_definitions", [])
        assert len(actual_tool_definitions) == expected_count, \
            f"Expected {expected_count} tool definition(s), " \
            f"but got {len(actual_tool_definitions)}: {actual_tool_definitions}"

    # ==================== CONVENIENCE TEST RUNNER ====================

    def run_tool_type_test(
        self,
        *,
        test_label: str,
        query: Any,
        response: Optional[Any] = None,
        tool_definitions: Optional[Any] = None,
        tool_calls: Optional[Any] = None,
        expected_flow_query: Optional[Any] = None,
        expected_flow_tool_calls: Optional[Any] = None,
        expected_flow_tool_definitions: Optional[Any] = None,
        expected_flow_tool_definitions_count: Optional[int] = None,
        expected_flow_tool_definition_contains: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run a tool-type-specific test with flow input assertions.

        Runs the evaluator with the given inputs, asserts success, and
        optionally verifies the exact input sent to the mocked _flow.

        Args:
            test_label: Descriptive label for the test.
            query: Query input for the evaluator.
            response: Optional response input.
            tool_definitions: Optional tool definitions.
            tool_calls: Optional tool calls.
            expected_flow_query: If set, assert _flow received this query.
            expected_flow_tool_calls: If set, assert _flow received these tool calls.
            expected_flow_tool_definitions: If set, assert _flow received these exact definitions.
            expected_flow_tool_definitions_count: If set, assert count of definitions sent to _flow.
            expected_flow_tool_definition_contains: If set, assert each definition is present in _flow input.

        Returns:
            Dictionary containing the extracted result data.
        """
        results = self._run_evaluation(
            query=query,
            response=response,
            tool_definitions=tool_definitions,
            tool_calls=tool_calls,
        )
        result_data = self._extract_and_print_result(results, test_label)

        # Assert the evaluator succeeded
        self.assert_pass(result_data)

        # Assert _flow was called
        self.assert_flow_was_called()

        # Assert flow input if expectations provided
        if expected_flow_query is not None:
            self.assert_flow_received_query(expected_flow_query)

        if expected_flow_tool_calls is not None:
            self.assert_flow_received_tool_calls(expected_flow_tool_calls)

        if expected_flow_tool_definitions is not None:
            self.assert_flow_received_tool_definitions(expected_flow_tool_definitions)

        if expected_flow_tool_definitions_count is not None:
            self.assert_flow_tool_definitions_count(expected_flow_tool_definitions_count)

        if expected_flow_tool_definition_contains is not None:
            for definition in expected_flow_tool_definition_contains:
                self.assert_flow_tool_definitions_contain(definition)

        return result_data
