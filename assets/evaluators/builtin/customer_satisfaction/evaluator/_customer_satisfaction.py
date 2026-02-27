# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import os
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._common.utils import reformat_conversation_history, reformat_agent_response
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


class CustomerSatisfactionValidator(ValidatorInterface):
    """Validate customer satisfaction evaluator inputs (query and response)."""

    error_target: ErrorTarget

    def __init__(self, error_target: ErrorTarget):
        """Initialize CustomerSatisfactionValidator."""
        self.error_target = error_target

    def _validate_query(self, query) -> Optional[EvaluationException]:
        if not query:
            return EvaluationException(
                message="Query input is required but not provided.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=self.error_target,
            )
        return None

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
        query = eval_input.get("query")
        response = eval_input.get("response")
        query_validation_exception = self._validate_query(query)
        if query_validation_exception:
            raise query_validation_exception
        response_validation_exception = self._validate_response(response)
        if response_validation_exception:
            raise response_validation_exception
        return True


# endregion Validators

logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes CUSTOMER_SATISFACTION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["CUSTOMER_SATISFACTION_EVALUATOR"] = "CustomerSatisfactionEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


def _is_intermediate_response(response):
    """Check if response is intermediate (last content item is function_call or mcp_approval_request)."""
    if isinstance(response, list) and len(response) > 0:
        last_msg = response[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            content = last_msg.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                last_content = content[-1]
                if (isinstance(last_content, dict) and
                        last_content.get("type") in ("function_call", "mcp_approval_request")):
                    return True
    return False


@experimental
class CustomerSatisfactionEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Customer Satisfaction evaluator predicts customer satisfaction on a 1-5 Likert scale.

    This evaluator assesses whether an AI agent's response would likely result in a satisfied
    customer based on:
        - Helpfulness: Did the agent address what the user actually needed?
        - Completeness: Was the response thorough and complete?
        - Clarity: Was the response clear and easy to understand?
        - Tone: Was the tone appropriate, professional, and empathetic?
        - Resolution: Did the interaction resolve the user's issue?

    Scoring is on a 1-5 Likert scale:
    - 5: Very Satisfied - Issue fully resolved, excellent interaction
    - 4: Satisfied - Mostly addressed needs with minor gaps
    - 3: Neutral - Partially addressed needs, adequate response
    - 2: Dissatisfied - Failed to adequately address needs
    - 1: Very Dissatisfied - Completely failed to help or made things worse

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the evaluator. Default is 3.
    :type threshold: int

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START customer_satisfaction_evaluator]
            :end-before: [END customer_satisfaction_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a CustomerSatisfactionEvaluator with a query and response.

    """

    _PROMPTY_FILE = "customer_satisfaction.prompty"
    _RESULT_KEY = "customer_satisfaction"
    _OPTIONAL_PARAMS = []

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/customer_satisfaction"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, threshold=3, **kwargs):
        """Initialize the CustomerSatisfactionEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword threshold: The threshold for the evaluator. Default is 3.
        :type threshold: int
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True

        # Initialize input validator
        self._validator = CustomerSatisfactionValidator(
            error_target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR
        )

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
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate customer satisfaction for a given query and response.

        The query and response can be either a string or a list of messages.

        Example with string inputs:
            evaluator = CustomerSatisfactionEvaluator(model_config)
            query = "I need to cancel my order #12345"
            response = "I've cancelled your order. Refund will be processed in 3-5 days."

            result = evaluator(query=query, response=response)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :return: A dictionary with the customer satisfaction evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate customer satisfaction for a conversation.

        :keyword conversation: The conversation to evaluate.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The customer satisfaction score
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
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
            f"{self._result_key}_dimensions": {},
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
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do Customer Satisfaction evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Customer Satisfaction evaluator.",
                internal_message="Both query and response required for Customer Satisfaction evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
            )

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )

        # Reformat inputs if they are lists of messages
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = reformat_conversation_history(
                eval_input["query"], logger, include_system_messages=True
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = reformat_agent_response(
                eval_input["response"], logger, include_tool_messages=True
            )

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            score_value = llm_output.get("score", 3)
            if isinstance(score_value, str):
                score = float(score_value) if score_value.replace(".", "").isdigit() else 3.0
            else:
                score = float(score_value) if score_value else 3.0

            # Clamp score to 1-5 range
            score = max(1.0, min(5.0, score))

            success_result = "pass" if score >= self._threshold else "fail"
            reason = llm_output.get("explanation", "")
            dimensions = llm_output.get("dimensions", {})

            return {
                self._result_key: score,
                f"{self._result_key}_result": success_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_dimensions": dimensions,
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }

        # Check if base returned nan (invalid output case)
        if isinstance(llm_output, float) and math.isnan(llm_output):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
            )

        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
        )
