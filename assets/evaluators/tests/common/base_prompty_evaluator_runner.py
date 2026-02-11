# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for evaluator tests.

Supports both mocked flow (for behavioral tests) and real flow execution (for quality tests).
"""

import os
from enum import Enum
from typing import Any, Dict, List, Type, override

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import ErrorCategory
from azure.identity import DefaultAzureCredential

from .base_evaluator_runner import BaseEvaluatorRunner


class BasePromptyEvaluatorRunner(BaseEvaluatorRunner):
    """
    Base class for running prompty-based evaluators for testing.

    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "Relevance")

    Subclasses may override:
    - use_mocking: bool - whether to mock the flow (default: True for behavioral tests)
    """

    # Subclasses must implement
    evaluator_type: Type[PromptyEvaluatorBase] = None

    # Subclasses may override
    use_mocking: bool = True  # Set to False for quality tests with real flow execution

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for prompty evaluators."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_threshold"
        ]

    @property
    def result_key(self) -> str:
        """Get the result key from the evaluator type."""
        return self.evaluator_type._RESULT_KEY

    @override
    def _init_evaluator(self, **kwargs) -> PromptyEvaluatorBase:
        """Create evaluator instance with model config.

        Args:
            **kwargs: Keyword arguments passed to the evaluator constructor.

        Returns:
            Configured evaluator instance.

        Raises:
            ValueError: If evaluator_type, or result_key is not set.
        """
        if self.use_mocking:
            # Dummy model config for behavioral tests - not used since flow is mocked
            model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
            )
        else:
            # Real model config for quality tests - makes actual LLM calls
            model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            )

        credential = DefaultAzureCredential()
        is_reasoning_model = os.getenv("AZURE_OPENAI_IS_REASONING_MODEL", "false").lower() == "true"

        return super()._init_evaluator(
            model_config=model_config, credential=credential, is_reasoning_model=is_reasoning_model, **kwargs
        )

    @override
    def _run_evaluation(
        self,
        query: List[Dict[str, Any]] = None,
        response: List[Dict[str, Any]] = None,
        tool_calls: List[Dict[str, Any]] = None,
        tool_definitions: List[Dict[str, Any]] = None,
        context: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run evaluation and return results.

        Args:
            query: Query conversation messages (optional)
            response: Response conversation messages (optional)
            tool_calls: Tool calls data (optional)
            tool_definitions: Tool definitions (optional)
            context: Additional context (optional)

        Returns:
            Dictionary containing evaluation results
        """
        return super()._run_evaluation(
                query=query,
                response=response,
                tool_calls=tool_calls,
                tool_definitions=tool_definitions,
                context=context,
                **kwargs,
            )

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

    @override
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
