# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of evaluators using AIProjectClient.

Runs evaluations for testing.
"""

import os
from enum import Enum
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import EvaluationException, ErrorCategory
from azure.identity import DefaultAzureCredential

from .evaluator_mock_config import get_flow_side_effect_for_evaluator


class BaseEvaluatorRunner:
    """
    Base class for running evaluators for testing.

    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "Relevance")
    """

    # Subclasses must implement
    evaluator_type: type[PromptyEvaluatorBase] = None

    def _init_evaluator(self) -> PromptyEvaluatorBase:
        """Create evaluator instance."""
        if self.evaluator_type is None:
            raise ValueError("Evaluator type not set. Subclass must define evaluator_type.")

        # Dummy model config and credential for testing - not used since flow is mocked
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
        )
        credential = DefaultAzureCredential()

        evaluator = self.evaluator_type(model_config=model_config, credential=credential)
        return evaluator

    def _run_evaluation(
        self,
        query: List[Dict[str, Any]],
        response: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        tool_definitions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run evaluation and return results."""
        evaluator_name = self.evaluator_type._RESULT_KEY
        evaluator = self._init_evaluator()

        # Mock the flow with appropriate side effect based on evaluator type
        evaluator._flow = MagicMock(side_effect=get_flow_side_effect_for_evaluator(evaluator_name))

        # Special handling for groundedness evaluator to disable flow reloading
        if hasattr(evaluator, "_ensure_query_prompty_loaded"):
            evaluator._ensure_query_prompty_loaded = MagicMock()

        try:
            results = evaluator(
                query=query,
                response=response,
                tool_calls=tool_calls,
                tool_definitions=tool_definitions,
            )
            return results
        except EvaluationException as e:
            print(f"Error during evaluation: {e}")
            return {
                f"{evaluator_name}_error_message": e.message,
                f"{evaluator_name}_error_code": e.category.name,
            }
        except Exception as e:
            print(f"Unexpected error during evaluation: {e}")
            return {
                f"{evaluator_name}_error_message": str(e),
            }

    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Extract result fields and print them."""
        evaluator_name = self.evaluator_type._RESULT_KEY
        label = results.get(f"{evaluator_name}_result")
        reason = results.get(f"{evaluator_name}_reason")
        score = results.get(f"{evaluator_name}")
        error_message = results.get(f"{evaluator_name}_error_message")
        error_code = results.get(f"{evaluator_name}_error_code")

        print(f"\n[{test_label}] Result: {label}")
        print(f"  Score: {score}")
        print(f"  Reason: {reason}")
        if error_message or error_code:
            print(f"  Error Message: {error_message}")
            print(f"  Error Code: {error_code}")

        return {
            "label": label,
            "reason": reason,
            "score": score,
            "error_message": error_message,
            "error_code": error_code,
        }

    # Helper Methods and Enums
    class AssertType(Enum):
        """Enumeration of assertion types for evaluator testing.

        Attributes:
            MISSING_FIELD: Indicates test should assert a missing field error.
            INVALID_VALUE: Indicates test should assert an invalid value error.
            PASS: Indicates test should assert a passing result.
        """

        MISSING_FIELD = "MISSING_FIELD"
        INVALID_VALUE = "INVALID_VALUE"
        PASS = "PASS"

    def assert_expected_behavior(self, assert_type: AssertType, result_data: Dict[str, Any]):
        """Assert the expected behavior based on the assert type.

        Args:
            assert_type: Type of assertion to perform (MISSING_FIELD, INVALID_VALUE, or PASS).
            result_data: Dictionary containing evaluation result data to validate.

        Raises:
            ValueError: If an unknown assert type is provided.
        """
        if assert_type == self.AssertType.MISSING_FIELD:
            self.assert_missing_field_error(result_data)
        elif assert_type == self.AssertType.INVALID_VALUE:
            self.assert_invalid_value_error(result_data)
        elif assert_type == self.AssertType.PASS:
            self.assert_pass(result_data)
        else:
            raise ValueError(f"Unknown assert type: {assert_type}")

    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result.

        Validates that the result has a 'pass' label and a score >= 1.0.

        Args:
            result_data: Dictionary containing evaluation result data with 'label' and 'score' keys.

        Raises:
            AssertionError: If the result does not meet passing criteria.
        """
        assert result_data["label"] == "pass"
        score_type = type(result_data["score"])
        assert score_type is float or score_type is int
        assert result_data["score"] >= 1.0

    def assert_fail(self, result_data: Dict[str, Any]):
        """Assert a failing result.

        Validates that the result has a 'fail' label and a score of 0.0.

        Args:
            result_data: Dictionary containing evaluation result data with 'label' and 'score' keys.

        Raises:
            AssertionError: If the result does not meet failing criteria.
        """
        assert result_data["label"] == "fail"
        score_type = type(result_data["score"])
        assert score_type is float or score_type is int
        assert result_data["score"] == 0.0

    def assert_pass_or_fail(self, result_data):
        """Assert a pass or fail result.

        Validates that the result has either a 'pass' or 'fail' label and a score >= 0.0.

        Args:
            result_data: Dictionary containing evaluation result data with 'label' and 'score' keys.

        Raises:
            AssertionError: If the result does not have a valid pass/fail label or score.
        """
        assert result_data["label"] in ["pass", "fail"]
        score_type = type(result_data["score"])
        assert score_type is float or score_type is int
        assert result_data["score"] >= 0.0

    def assert_error(self, result_data: Dict[str, Any], error_code: str):
        """Assert an error result.

        Validates that the result contains an error with the expected error code and no label or score.

        Args:
            result_data: Dictionary containing evaluation result data.
            error_code: Expected error code to validate against.

        Raises:
            AssertionError: If the result does not match the expected error state.
        """
        assert result_data["label"] is None
        assert result_data["score"] is None
        assert result_data["error_code"] == error_code

    def assert_missing_field_error(self, result_data: Dict[str, Any]):
        """Assert a missing field error result.

        Validates that the result contains a MISSING_FIELD error.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result does not contain the expected missing field error.
        """
        self.assert_error(result_data, ErrorCategory.MISSING_FIELD.name)

    def assert_invalid_value_error(self, result_data: Dict[str, Any]):
        """Assert an invalid value error result.

        Validates that the result contains an INVALID_VALUE error.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result does not contain the expected invalid value error.
        """
        self.assert_error(result_data, ErrorCategory.INVALID_VALUE.name)

    def assert_not_applicable(self, result_data: Dict[str, Any]):
        """Assert a not applicable result.

        Validates that the result has a 'pass' label and 'not applicable' score.
        Note: Currently validates as 'pass' label, but may need to be updated to a dedicated 'not applicable' label.

        Args:
            result_data: Dictionary containing evaluation result data with 'label' and 'score' keys.

        Raises:
            AssertionError: If the result does not match the not applicable criteria.
        """
        assert result_data["label"] == "pass"  # TODO: this should be not applicable?
        assert result_data["score"] == "not applicable"
