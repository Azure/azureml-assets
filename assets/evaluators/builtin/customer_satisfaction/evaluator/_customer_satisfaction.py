# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Any, Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._common.constants import EvaluationLevel
from azure.ai.evaluation._common.utils import reformat_conversation_history, reformat_agent_response
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    validate_model_config,
    _resolve_evaluation_level,
    _is_intermediate_response,
    _preprocess_messages,
    _wrap_string_messages,
    _merge_query_response_messages,
    _split_messages_at_latest_user,
    serialize_messages,
)
from azure.ai.evaluation._common._experimental import experimental

from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    MessageRole,
    MessagesOrQueryResponseInputValidator,
)

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty


logger = logging.getLogger(__name__)


# Use the SDK's ErrorTarget member when the installed version defines it; otherwise fall back to EVALUATE.
_ERROR_TARGET = getattr(ErrorTarget, "CUSTOMER_SATISFACTION_EVALUATOR", ErrorTarget.EVALUATE)


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
    _MULTI_TURN_PROMPTY_FILE = "customer_satisfaction_multi_turn.prompty"
    _RESULT_KEY = "customer_satisfaction"
    _OPTIONAL_PARAMS = ["messages"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/customer_satisfaction"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, threshold=3, evaluation_level=None, **kwargs):
        """Initialize the CustomerSatisfactionEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword threshold: The threshold for the evaluator. Default is 3.
        :type threshold: int
        :keyword evaluation_level: Force a specific evaluation level for this invocation. When ``None``
            (default), the level is auto-detected from input shape (``messages`` -> conversation,
            ``query``/``response`` -> turn). Set to ``EvaluationLevel.CONVERSATION`` or
            ``EvaluationLevel.TURN`` to override auto-detection.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True

        # Validate and store evaluation level
        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, _ERROR_TARGET
        )

        # Initialize input validator
        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=_ERROR_TARGET,
            requires_query=True,
            enforce_tool_definitions=False,
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

        # Load the multi-turn prompty flow for multi-turn evaluation
        multi_turn_prompty_path = os.path.join(current_dir, self._MULTI_TURN_PROMPTY_FILE)
        prompty_model_config = construct_prompty_model_config(
            validate_model_config(model_config),
            self._DEFAULT_OPEN_API_VERSION,
            f"azure-ai-evaluation (type=evaluator subtype={self.__class__.__name__})",
        )
        self._multi_turn_flow = AsyncPrompty.load(
            source=multi_turn_prompty_path,
            model=prompty_model_config,
            token_credential=credential,
            is_reasoning_model=self._is_reasoning_model,
        )

    @overload
    def __call__(
        self,
        *,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate customer satisfaction for last agent response given a query, response.

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
        messages: List[dict],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate customer satisfaction for a full multi-turn conversation session.

        Example with messages:
            evaluator = CustomerSatisfactionEvaluator(model_config)
            messages = [
                {'role': 'user', 'content': [{'type': 'text', 'text': 'I need to cancel my order'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'I have cancelled it.'}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'What about the refund?'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Refund processed in 3-5 days.'}]},
            ]
            result = evaluator(messages=messages)

        :keyword messages: The full multi-turn conversation as a list of message dicts.
        :paramtype messages: List[dict]
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
        """Return a result indicating that the evaluation is not applicable (skipped).

        Not-applicable results have no score since the evaluator cannot make a judgment
        (e.g., intermediate responses that are not final agent responses).
        """
        return self._build_result(
            score=None,
            result="not_applicable",
            reason=f"Not applicable: {error_message}",
            status="skipped",
            properties={},
        )

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether to use conversation-level evaluation.

        When ``_evaluation_level`` is set, it takes precedence. Otherwise, auto-detect
        based on whether ``messages`` is present in the input.

        :param eval_input: The evaluation input.
        :type eval_input: Dict
        :return: True if conversation-level evaluation should be used.
        :rtype: bool
        """
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
        # Auto-detect (_evaluation_level is None)
        return eval_input.get("messages") is not None

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Reshape inputs based on evaluation level before validation
        if self._evaluation_level == EvaluationLevel.CONVERSATION and not kwargs.get("messages"):
            query = kwargs.get("query")
            response = kwargs.get("response")
            if isinstance(query, str) and isinstance(response, str) and query and response:
                query, response = _wrap_string_messages(query, response)
            if isinstance(query, list) and isinstance(response, list):
                kwargs["messages"] = _merge_query_response_messages(query, response)
        elif self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            if any(m.get("role") == MessageRole.USER for m in kwargs["messages"]):
                query_messages, response_messages = _split_messages_at_latest_user(kwargs["messages"])
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages
                kwargs.pop("messages", None)

        self._validator.validate_eval_input(kwargs)

        return await self._the_super_real_call(**kwargs)

    async def _the_super_real_call(self, **kwargs):
        """Perform the asynchronous call where real end-to-end evaluation logic runs.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Convert inputs into list of evaluable inputs.
        try:
            eval_input_list = self._convert_kwargs_to_eval_input(**kwargs)
        except Exception as e:
            logger.error(f"Error converting kwargs to eval_input_list: {e}")
            raise e
        per_turn_results = []
        # Evaluate all inputs.
        for eval_input in eval_input_list:
            result = await self._do_eval(eval_input)
            # logic to determine threshold pass/fail
            # if it wasn't computed in _do_eval
            try:
                keys = list(result.keys())
                contains_result_key = any(key.endswith("_result") for key in keys)
                contains_threshold_key = any(key.endswith("_threshold") for key in keys)
                if not contains_result_key or not contains_threshold_key:
                    for key in keys:
                        if key.endswith("_score"):
                            score_value = result[key]
                            base_key = key[:-6]  # Remove "_score" suffix
                            result_key = f"{base_key}_result"
                            threshold_key = f"{base_key}_threshold"
                            threshold_value = (
                                self._threshold.get(base_key) if isinstance(self._threshold, dict) else self._threshold
                            )
                            if not isinstance(threshold_value, (int, float)):
                                raise EvaluationException(
                                    "Threshold value must be a number.",
                                    internal_message=str(threshold_value),
                                    target=ErrorTarget.EVALUATE,
                                    category=ErrorCategory.INVALID_VALUE,
                                    blame=ErrorBlame.USER_ERROR,
                                )
                            if not contains_threshold_key:
                                result[threshold_key] = threshold_value
                            if not contains_result_key:
                                if self._higher_is_better:
                                    if float(score_value) >= threshold_value:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[True]
                                    else:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[False]
                                else:
                                    if float(score_value) <= threshold_value:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[True]
                                    else:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[False]
            except Exception as e:
                logger.warning(f"Error calculating binary result: {e}")
            per_turn_results.append(result)
        # Return results as-is if only one result was produced.
        if len(per_turn_results) == 1:
            return per_turn_results[0]
        if len(per_turn_results) == 0:
            return {}  # TODO raise something?
        # Otherwise, aggregate results.
        return self._aggregate_results(per_turn_results=per_turn_results)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do Customer Satisfaction evaluation.

        Routes to conversation-level or turn-level evaluation based on
        ``_evaluation_level`` (if set)
        or auto-detects from input shape (default).

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # Multi-turn path (messages)
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_multi_turn(eval_input)

        # Single-turn path (query/response)
        if eval_input.get("query") is None or eval_input.get("response") is None:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Customer Satisfaction evaluator.",
                internal_message="Both query and response required for Customer Satisfaction evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=_ERROR_TARGET,
            )

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )

        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

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
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_multi_turn(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Evaluate customer satisfaction for a full multi-turn conversation session.

        :param eval_input: The input containing ``messages``.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        messages = eval_input["messages"]

        messages = _preprocess_messages(messages)
        conversation_text = serialize_messages(messages)

        prompty_kwargs: Dict[str, Any] = {"conversation": conversation_text}

        prompty_output_dict = await self._multi_turn_flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_kwargs)
        return self._parse_prompty_output(prompty_output_dict)

    def _build_result(
        self,
        score: Optional[int],
        result: str,
        reason: str,
        status: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build a standardized result dictionary.

        :param score: The evaluation score (1, 0, or None).
        :param result: The result label ("pass", "fail", "not_applicable", or "error").
        :param reason: The reasoning or explanation string.
        :param status: The evaluation status ("completed", "skipped", or "error").
        :param properties: The properties dictionary.
        :param prompty_output_dict: Optional raw prompty output for extracting token metadata.
        :return: The standardized result dictionary.
        """
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        metadata = {
            "prompt_tokens": p.get("input_token_count", 0),
            "completion_tokens": p.get("output_token_count", 0),
            "total_tokens": p.get("total_token_count", 0),
            "finish_reason": p.get("finish_reason", ""),
            "model": p.get("model_id", ""),
            "sample_input": p.get("sample_input", ""),
            "sample_output": p.get("sample_output", ""),
        }
        result_payload = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_passed": result == "pass" if result in ["pass", "fail"] else None,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_properties": {**properties, **metadata},
        }
        # Add top-level token metadata fields for backward compatibility.
        result_payload.update({f"{self._result_key}_{key}": value for key, value in metadata.items()})
        return result_payload

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Any]:
        """Parse the prompty output into a standardized result dictionary.

        Shared between single-turn and multi-turn evaluation paths.

        :param prompty_output_dict: Raw output from the prompty flow.
        :type prompty_output_dict: Dict
        :return: The parsed evaluation result.
        :rtype: Dict
        """
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            score = None
            result = "error"
            reason = "Evaluator returned invalid output."
            status = "error"
            properties = {}
        else:
            status = llm_output.get("status", "completed")
            reason = llm_output.get("reason", "")
            properties = llm_output.get("properties") or {}

            if status == "skipped":
                score = None
                result = "skipped"
            else:
                score = llm_output.get("score", self._threshold)
                result = "pass" if score >= self._threshold else "fail"

        return self._build_result(
            score=score,
            result=result,
            reason=reason,
            status=status,
            properties=properties,
            prompty_output_dict=prompty_output_dict,
        )
