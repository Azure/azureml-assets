# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import reformat_agent_response
from azure.ai.evaluation._common._experimental import experimental
from enum import Enum

from abc import ABC, abstractmethod
from typing import Any


# region Validators


class ValidatorInterface(ABC):
    """Abstract base class defining the interface that all validators must implement."""

    @abstractmethod
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate the evaluation input dictionary."""
        pass


class DeflectionRateValidator(ValidatorInterface):
    """Validate deflection rate evaluator inputs (response only)."""

    error_target: ErrorTarget

    def __init__(self, error_target: ErrorTarget):
        """Initialize DeflectionRateValidator."""
        self.error_target = error_target

    def _validate_response(self, response) -> Optional[EvaluationException]:
        if not response:
            return EvaluationException(
                message="Response input is required but not provided.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=self.error_target,
            )
        return None

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate evaluation input."""
        response = eval_input.get("response")
        response_validation_exception = self._validate_response(response)
        if response_validation_exception:
            raise response_validation_exception
        return True


# endregion Validators

logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes DEFLECTION_RATE_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["DEFLECTION_RATE_EVALUATOR"] = "DeflectionRateEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


@experimental
class DeflectionRateEvaluator(PromptyEvaluatorBase[Union[str, int]]):
    """The Deflection Rate evaluator determines whether an AI assistant deflected a user query.

    A deflection occurs when the AI indicates the topic is out of scope, suggests seeking help
    elsewhere, or fails to provide a direct answer. This evaluator is useful for measuring:
        - Chatbot effectiveness in resolving queries without human intervention
        - Customer support automation rates
        - Self-service success rates

    Deflection types:
        - plain_denial: Explicitly states it cannot answer
        - send_elsewhere: Suggests seeking help from another source
        - reframe: Reframes the question to fall within its scope
        - plain_answer: Provides a direct answer (no deflection)

    Scoring is binary:
    - 0: No deflection - the system provided a direct answer
    - 1: Deflection - the system indicated the topic is out of scope

    Note: Lower scores are better for this evaluator (desirable_direction: decrease).

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START deflection_rate_evaluator]
            :end-before: [END deflection_rate_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a DeflectionRateEvaluator with a response.

    """

    _PROMPTY_FILE = "deflection_rate.prompty"
    _RESULT_KEY = "deflection_rate"
    _OPTIONAL_PARAMS = []

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/deflection_rate"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, threshold=0, **kwargs):
        """Initialize the DeflectionRateEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword threshold: The threshold for the evaluator. Default is 0 (no deflection expected).
        :type threshold: int
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = False  # Lower deflection is better

        # Initialize input validator
        self._validator = DeflectionRateValidator(error_target=ExtendedErrorTarget.DEFLECTION_RATE_EVALUATOR)

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        response: Union[str, List[dict]],
    ) -> Dict[str, Union[str, int]]:
        """Evaluate deflection rate for a given response.

        The response can be either a string or a list of messages.

        Example with string input:
            evaluator = DeflectionRateEvaluator(model_config)
            response = "The capital of France is Paris."

            result = evaluator(response=response)

        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :return: A dictionary with the deflection rate evaluation results.
        :rtype: Dict[str, Union[str, int]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Invoke the instance using the overloaded __call__ signature.

        For detailed parameter types and return value documentation, see the overloaded __call__ definition.
        """
        return super().__call__(*args, **kwargs)

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: threshold,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_deflection_type": "",
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
        }

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
        self._validator.validate_eval_input(kwargs)

        return await super()._real_call(**kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:  # type: ignore[override]
        """Do Deflection Rate evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "response" not in eval_input:
            raise EvaluationException(
                message="Response must be provided as input to the Deflection Rate evaluator.",
                internal_message="Response must be provided as input to the Deflection Rate evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.DEFLECTION_RATE_EVALUATOR,
            )

        # Reformat response if it's a list of messages
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            score_value = llm_output.get("score", 0)
            if isinstance(score_value, str):
                score = int(score_value) if score_value.isdigit() else 0
            else:
                score = int(score_value) if score_value else 0
            
            # For deflection, lower is better, so pass when score <= threshold
            success_result = "pass" if score <= self._threshold else "fail"
            reason = llm_output.get("explanation", "")
            deflection_type = llm_output.get("deflection_type", "")
            
            return {
                self._result_key: score,
                f"{self._result_key}_result": success_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_deflection_type": deflection_type,
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ExtendedErrorTarget.DEFLECTION_RATE_EVALUATOR,
        )
