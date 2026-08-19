# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from enum import Enum
from typing import Any, Dict, Optional, Union, List

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    reformat_agent_response,
    validate_model_config,
)
from azure.ai.evaluation._common._experimental import experimental

from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ConversationValidator,
)

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.constants import EvaluationLevel
    from azure.ai.evaluation._common.utils import (
        _is_intermediate_response,
        _preprocess_messages,
        _resolve_evaluation_level,
        _split_messages_at_latest_user,
        serialize_messages,
    )
    from azure.ai.evaluation._evaluators._common._validators import MessagesOrQueryResponseInputValidator
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import _is_intermediate_response
    from ...task_completion.evaluator._task_completion import (
        EvaluationLevel,
        MessagesOrQueryResponseInputValidator,
        _preprocess_messages,
        _resolve_evaluation_level,
        _split_messages_at_latest_user,
        serialize_messages,
    )

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._evaluators._common._validators import MessageRole
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    class MessageRole(str, Enum):
        """Valid message roles."""

        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        TOOL = "tool"
        DEVELOPER = "developer"


logger = logging.getLogger(__name__)


# Use the SDK's ErrorTarget member when the installed version defines it; otherwise fall back to EVALUATE.
_ERROR_TARGET = getattr(ErrorTarget, "DEFLECTION_RATE_EVALUATOR", ErrorTarget.EVALUATE)


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
    _MULTI_TURN_PROMPTY_FILE = "deflection_rate_multi_turn.prompty"
    _RESULT_KEY = "deflection_rate"
    _OPTIONAL_PARAMS = ["messages"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/deflection_rate"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, threshold=0, evaluation_level=None, **kwargs):
        """Initialize the DeflectionRateEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword threshold: The threshold for the evaluator. Default is 0 (no deflection expected).
        :type threshold: int
        :keyword evaluation_level: Force turn or conversation evaluation. When omitted, ``messages``
            selects conversation evaluation and ``response`` selects turn evaluation.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = False  # Lower deflection is better
        self._evaluation_level = _resolve_evaluation_level(evaluation_level, _ERROR_TARGET)

        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=_ERROR_TARGET,
            requires_query=False,
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

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
    ) -> Dict[str, Union[str, int]]:
        """Evaluate deflection rate across a complete conversation."""

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

    def _return_not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a standardized result for an evaluation that was skipped."""
        return self._build_result(
            score=None,
            result="not_applicable",
            reason=f"Not applicable: {error_message}",
            status="skipped",
            properties={},
        )

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether to use conversation-level evaluation."""
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
        return eval_input.get("messages") is not None

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        if self._evaluation_level == EvaluationLevel.CONVERSATION and kwargs.get("messages") is None:
            raise EvaluationException(
                message="Messages must be provided for conversation-level Deflection Rate evaluation.",
                internal_message="Messages must be provided for conversation-level Deflection Rate evaluation.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=_ERROR_TARGET,
            )

        if self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            messages = kwargs["messages"]
            if isinstance(messages, list) and any(
                isinstance(message, dict) and message.get("role") == "user" for message in messages
            ):
                query_messages, response_messages = _split_messages_at_latest_user(messages)
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages
                kwargs.pop("messages", None)

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
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        if eval_input.get("response") is None:
            raise EvaluationException(
                message="Response must be provided as input to the Deflection Rate evaluator.",
                internal_message="Response must be provided as input to the Deflection Rate evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=_ERROR_TARGET,
            )

        # Check for intermediate response (function_call or mcp_approval_request)
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )

        # Reformat response if it's a list of messages
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
            eval_input["response"] = reformat_agent_response(
                eval_input["response"], logger, include_tool_messages=True
            )

        # The single-turn prompt evaluates only the response; query/messages are routing inputs.
        eval_input.pop("query", None)
        eval_input.pop("messages", None)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[int, str]]:
        """Evaluate deflection rate across a full serialized conversation."""
        messages = _preprocess_messages(eval_input["messages"])
        conversation_text = serialize_messages(messages)
        prompty_output_dict = await self._multi_turn_flow(
            timeout=self._LLM_CALL_TIMEOUT,
            messages=conversation_text,
        )
        return self._parse_prompty_output(prompty_output_dict)

    def _build_result(
        self,
        score: Optional[int],
        result: str,
        reason: str,
        status: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
        deflection_type: str = "",
    ) -> Dict[str, Any]:
        """Build a standardized result dictionary with legacy metadata fields."""
        metadata = self._get_token_metadata(prompty_output_dict or {})
        result_payload = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_passed": result == "pass" if result in ["pass", "fail"] else None,
            f"{self._result_key}_result": result,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": {**properties, **metadata},
            f"{self._result_key}_deflection_type": deflection_type,
        }
        result_payload.update({f"{self._result_key}_{key}": value for key, value in metadata.items()})
        return result_payload

    @staticmethod
    def _get_token_metadata(prompty_output: Dict) -> Dict[str, Any]:
        """Extract token usage and model metadata from prompty output."""
        return {
            "prompt_tokens": prompty_output.get("input_token_count", 0),
            "completion_tokens": prompty_output.get("output_token_count", 0),
            "total_tokens": prompty_output.get("total_token_count", 0),
            "finish_reason": prompty_output.get("finish_reason", ""),
            "model": prompty_output.get("model_id", ""),
            "sample_input": prompty_output.get("sample_input", ""),
            "sample_output": prompty_output.get("sample_output", ""),
        }

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Any]:
        """Parse either prompt flow's output into the standardized result schema."""
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=_ERROR_TARGET,
            )

        llm_status = str(llm_output.get("status") or "completed").strip().lower()
        if llm_status == "skipped":
            reason = llm_output.get("reason", llm_output.get("explanation", ""))
            return self._return_not_applicable_result(reason, self._threshold)

        score_value = llm_output.get("score", 0)
        if isinstance(score_value, str):
            score = int(score_value) if score_value.isdigit() else 0
        else:
            score = int(score_value) if score_value else 0

        success_result = "pass" if score <= self._threshold else "fail"
        return self._build_result(
            score=score,
            result=success_result,
            reason=llm_output.get("explanation", llm_output.get("reason", "")),
            status="completed",
            properties=llm_output.get("properties") or {},
            prompty_output_dict=prompty_output_dict,
            deflection_type=llm_output.get("deflection_type", ""),
        )
