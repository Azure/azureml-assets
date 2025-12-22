# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of evaluators using AIProjectClient.

Runs evaluations for testing.
"""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import EvaluationException
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

    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result."""
        assert result_data["label"] == "pass"
        score_type = type(result_data["score"])
        assert score_type is float or score_type is int
        assert result_data["score"] >= 1.0

    def assert_fail(self, result_data: Dict[str, Any]):
        """Assert a failing result."""
        assert result_data["label"] == "fail"
        score_type = type(result_data["score"])
        assert score_type is float or score_type is int
        assert result_data["score"] == 0.0

    def assert_pass_or_fail(self, result_data):
        """Assert a pass or fail result."""
        assert result_data["label"] in ["pass", "fail"]
        score_type = type(result_data["score"])
        assert score_type is float or score_type is int
        assert result_data["score"] >= 0.0

    def assert_error(self, result_data: Dict[str, Any], error_code: str = "FAILED_EXECUTION"):
        """Assert an error result."""
        assert result_data["label"] is None
        assert result_data["score"] is None
        assert result_data["error_code"] == error_code

    def assert_not_applicable(self, result_data: Dict[str, Any]):
        """Assert a not applicable result."""
        assert result_data["label"] == "pass"  # TODO: this should be not applicable?
        assert result_data["score"] == "not applicable"
