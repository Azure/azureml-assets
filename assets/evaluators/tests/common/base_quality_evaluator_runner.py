# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for quality tests of evaluators with real flow execution (no mocking)."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .base_prompty_evaluator_runner import BasePromptyEvaluatorRunner


# Type aliases for clarity
Message = Dict[str, Any]
ConversationFormat = List[Message]
QueryInput = Union[str, ConversationFormat]
ResponseInput = Union[str, ConversationFormat]


class ExpectedResult(Enum):
    """Expected evaluation result for quality tests."""

    PASS = "pass"
    FAIL = "fail"
    PASS_OR_FAIL = "pass_or_fail"
    PASS_WITH_SCORE_3 = "pass_with_score_3"


class BaseQualityEvaluatorRunner(BasePromptyEvaluatorRunner):
    """
    Base class for quality tests that use real LLM flow execution.

    This is a thin wrapper around BasePromptyEvaluatorRunner that disables mocking
    and provides convenience methods for quality testing.

    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator
    """

    use_mocking = False  # Quality tests always use real flow execution

    def run_quality_test(
        self,
        *,
        test_label: str,
        expected: ExpectedResult,
        query: Optional[QueryInput] = None,
        response: Optional[ResponseInput] = None,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run a quality test with the given inputs and assert the expected result.

        This is a convenience method that combines _run_evaluation(),
        _extract_and_print_result(), and the appropriate assertion in one call.

        Args:
            test_label: Descriptive label for the test (printed in output).
            expected: Expected result (PASS, FAIL, PASS_OR_FAIL, or PASS_WITH_SCORE_3).
            query: User query - either a simple string or conversation format
                   (list of message dicts).
            response: Assistant response - either a simple string or conversation
                      format (list of message dicts).
            tool_definitions: Optional list of tool definition dicts.
            tool_calls: Optional list of tool call dicts.
            context: Optional context string.

        Returns:
            Dictionary containing the extracted result data (label, score, reason, etc.)

        Raises:
            AssertionError: If the result doesn't match the expected outcome.

        Example:
            # Simple string format
            self.run_quality_test(
                test_label="PASS-coherent-response",
                expected=ExpectedResult.PASS,
                query="What is the capital of France?",
                response="Paris is the capital of France."
            )

            # Complex conversation format with tools
            self.run_quality_test(
                test_label="FAIL-wrong-tool",
                expected=ExpectedResult.FAIL,
                query=[{"role": "user", "content": [{"type": "text", "text": "Send email"}]}],
                response=[{"role": "assistant", "content": [{"type": "tool_call", ...}]}],
                tool_definitions=ToolDefinitionSets.EMAIL_AND_FILE
            )
        """
        # Run the evaluation
        results = self._run_evaluation(
            query=query,
            response=response,
            tool_definitions=tool_definitions,
            tool_calls=tool_calls,
            context=context,
            **kwargs,
        )

        # Extract and print results
        result_data = self._extract_and_print_result(results, test_label)

        # Assert expected outcome
        if expected == ExpectedResult.PASS:
            self.assert_pass(result_data)
        elif expected == ExpectedResult.FAIL:
            self.assert_fail(result_data)
        elif expected == ExpectedResult.PASS_OR_FAIL:
            self.assert_pass_or_fail(result_data)
        elif expected == ExpectedResult.PASS_WITH_SCORE_3:
            self.assert_score_in_range(result_data, min_score=3, max_score=3)
        return result_data
