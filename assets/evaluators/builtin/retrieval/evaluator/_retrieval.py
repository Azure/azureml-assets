# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Union
from typing_extensions import overload, override

from azure.ai.evaluation._evaluators._common._base_prompty_eval import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

from azure.ai.evaluation._common.constants import PROMPT_BASED_REASON_EVALUATORS
from azure.ai.evaluation._common.utils import parse_quality_evaluator_reason_score

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import _is_intermediate_response, _preprocess_messages
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (
        _is_intermediate_response,
        _preprocess_messages,
    )

# Re-exported so the module keeps exposing the message-preprocessing helpers used
# by the test suite; they are invoked indirectly through _preprocess_messages.
try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )

logger = logging.getLogger(__name__)


class RetrievalEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """
    Evaluate retrieval score for a given query and context or a multi-turn conversation, including reasoning.

    The retrieval measure assesses the AI system's performance in retrieving information
    for additional context (e.g. a RAG scenario).

    Retrieval scores range from 1 to 5, with 1 being the worst and 5 being the best.

    High retrieval scores indicate that the AI system has successfully extracted and ranked
    the most relevant information at the top, without introducing bias from external knowledge
    and ignoring factual correctness. Conversely, low retrieval scores suggest that the AI system
    has failed to surface the most relevant context chunks at the top of the list
    and/or introduced bias and ignored factual correctness.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the evaluation. Default is 3.
    :type threshold: float
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, the evaluator will use reasoning model configuration (o1/o3 models).
        This will adjust parameters like max_completion_tokens and remove unsupported parameters. Default is False.
    :paramtype is_reasoning_model: bool
    :return: A function that evaluates and generates metrics for "chat" scenario.
    :rtype: Callable

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START retrieval_evaluator]
            :end-before: [END retrieval_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a RetrievalEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START retrieval_evaluator]
            :end-before: [END retrieval_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call RetrievalEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_retrieval_evaluator]
            :end-before: [END threshold_retrieval_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a RetrievalEvaluator.

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE = "retrieval.prompty"
    _RESULT_KEY = "retrieval"

    id = "azureai://built-in/evaluators/retrieval"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold: float = 3, credential=None, **kwargs):
        """Initialize the Retrieval evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation.
        :type threshold: float
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold,
            credential=credential,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        query: str,
        context: str,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate retrieval for a given query and context.

        :keyword query: The query to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype query: Optional[str]
        :keyword context: The context to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype context: Optional[str]
        :return: The scores for Chat scenario.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate retrieval for a multi-turn evaluation.

        If the conversation has more than one turn,
        the evaluator will aggregate the results of each turn.

        :keyword conversation: The conversation to be evaluated.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The scores for Chat scenario.
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
        """

    @override
    def __call__(self, *args, **kwargs):  # pylint: disable=docstring-missing-param
        """Evaluate retrieval score chat scenario.

        Accepts either a query and context for a single evaluation,
        or a conversation for a multi-turn evaluation. If the conversation has more than one turn,
        the evaluator will aggregate the results of each turn.

        :keyword query: The query to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype query: Optional[str]
        :keyword context: The context to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype context: Optional[str]
        :keyword conversation: The conversation to be evaluated.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The scores for Chat scenario.
        :rtype: :rtype: Dict[str, Union[float, Dict[str, List[str, float]]]]
        """
        return super().__call__(*args, **kwargs)

    @override
    def _convert_kwargs_to_eval_input(self, **kwargs) -> List[Dict]:
        """Convert conversation messages into query/context retrieval turns."""
        messages = kwargs.pop("messages", None)
        conversation = kwargs.get("conversation")
        if messages is None and conversation is not None:
            if isinstance(conversation, dict):
                messages = conversation.get("messages")
            else:
                messages = getattr(conversation, "messages", None)
            if messages is not None:
                kwargs.pop("conversation")
        if messages is None:
            return super()._convert_kwargs_to_eval_input(**kwargs)
        if not isinstance(messages, list) or not messages:
            raise EvaluationException(
                message="RetrievalEvaluator: 'messages' must be a non-empty list.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.RETRIEVAL_EVALUATOR,
            )

        explicit_context = kwargs.pop("context", None)
        explicit_query = kwargs.pop("query", None)
        if explicit_context:
            query = explicit_query or self._get_latest_user_query(messages)
            return super()._convert_kwargs_to_eval_input(query=query, context=explicit_context, **kwargs)

        eval_inputs = self._extract_retrieval_turns(messages)
        if not eval_inputs:
            raise EvaluationException(
                message=(
                    "RetrievalEvaluator: No valid context was provided or could be "
                    "extracted from tool outputs in 'messages'."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.NOT_APPLICABLE,
                target=ErrorTarget.RETRIEVAL_EVALUATOR,
            )
        if explicit_query and len(eval_inputs) == 1:
            eval_inputs[0]["query"] = explicit_query
        return eval_inputs

    @classmethod
    def _extract_retrieval_turns(cls, messages: List[Dict[str, Any]]) -> List[Dict]:
        """Build one retrieval input for each user turn containing tool output."""
        turns: List[Dict] = []
        query = ""
        context_parts: List[str] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "user":
                cls._append_retrieval_turn(turns, query, context_parts)
                query = cls._extract_message_text(message.get("content"))
                context_parts = []
            elif role == "tool" and query:
                context_parts.extend(cls._extract_tool_message_context(message.get("content")))

        cls._append_retrieval_turn(turns, query, context_parts)
        return turns

    @staticmethod
    def _append_retrieval_turn(turns: List[Dict], query: str, context_parts: List[str]) -> None:
        """Append a retrieval turn when both query and tool context are present."""
        context = "\n\n".join(part for part in context_parts if part)
        if query and context:
            turns.append({"query": query, "context": context})

    @classmethod
    def _get_latest_user_query(cls, messages: List[Dict[str, Any]]) -> str:
        """Return the latest user text in a conversation."""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                query = cls._extract_message_text(message.get("content"))
                if query:
                    return query
        return ""

    @staticmethod
    def _extract_message_text(content: Any) -> str:
        """Extract plain text from a normalized message content value."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"text", "input_text", "output_text"}:
                continue
            text = block.get("text") or block.get("content")
            if isinstance(text, str) and text:
                text_parts.append(text)
        return "\n".join(text_parts)

    @classmethod
    def _extract_tool_message_context(cls, content: Any) -> List[str]:
        """Serialize arbitrary tool output payloads without tool-name filtering."""
        if isinstance(content, str):
            return [content] if content else []
        if not isinstance(content, list):
            return []

        context_parts = []
        for block in content:
            if isinstance(block, str):
                context_parts.append(block)
                continue
            if not isinstance(block, dict) or block.get("type") == "tool_call":
                continue
            payload = block
            for key in ("tool_result", "output", "response", "result", "content", "text"):
                if key in block:
                    payload = block[key]
                    break
            serialized = cls._stringify_tool_output(payload)
            if serialized:
                context_parts.append(serialized)
        return context_parts

    @staticmethod
    def _stringify_tool_output(value: Any) -> str:
        """Preserve plain text and deterministically serialize structured output."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if value is None:
            return ""
        return str(value)

    def _return_not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a result indicating that the tool call is not applicable for evaluation.

        :param error_message: The error message indicating why the evaluation is not applicable.
        :type error_message: str
        :param threshold: The threshold value for the evaluation.
        :type threshold: Union[int, float]
        :return: A dictionary containing the result of the evaluation.
        :rtype: Dict[str, Union[str, float, None]]
        """
        token_metadata = self._get_token_metadata({})
        result = {
            f"{self._result_key}": None,
            f"{self._result_key}_score": None,
            f"{self._result_key}_passed": None,
            f"{self._result_key}_result": "not_applicable",
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_status": "skipped",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_properties": None,
        }
        # Add top-level token metadata fields for backward compatibility.
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result

    async def _the_super_do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Do a relevance evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Only text conversation inputs are supported.",
                internal_message="Only text conversation inputs are supported.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.CONVERSATION,
            )
        # Check for intermediate response
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        # Preprocess messages if they are lists
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        # Call the prompty flow to get the evaluation result.
        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        score = math.nan
        reason = ""
        llm_properties = {}
        if prompty_output_dict:
            llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)
            parsed_output = None
            if isinstance(llm_output, dict):
                parsed_output = llm_output
            elif isinstance(llm_output, str):
                try:
                    parsed_output = json.loads(llm_output)
                except (json.JSONDecodeError, TypeError):
                    parsed_output = None
            if parsed_output and isinstance(parsed_output, dict):
                llm_status = parsed_output.get("status", "completed")
                if llm_status == "skipped":
                    skip_reason = parsed_output.get("reason", "")
                    return self._return_not_applicable_result(skip_reason, self._threshold)
                score = parsed_output.get("score", math.nan)
                reason = parsed_output.get("reason", "")
                llm_properties = parsed_output.get("properties", {}) or {}
            else:
                if isinstance(llm_output, str) and self._result_key in PROMPT_BASED_REASON_EVALUATORS:
                    score, reason = parse_quality_evaluator_reason_score(llm_output)
                elif isinstance(llm_output, str):
                    match = re.search(r"\d", llm_output)
                    if match:
                        score = float(match.group())
            score = float(score) if score is not None else math.nan
            score_result = self._get_binary_result(score)
            token_metadata = self._get_token_metadata(prompty_output_dict)
            llm_properties.update(token_metadata)
            result = {
                self._result_key: score,
                f"{self._result_key}_score": score,
                f"{self._result_key}_passed": score_result == "pass",
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_status": "completed",
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_properties": llm_properties,
            }
            # Add top-level token metadata fields for backward compatibility.
            result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
            return result
        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ErrorTarget.EVALUATE,
        )

    @staticmethod
    def _get_token_metadata(prompty_output: Dict) -> Dict:
        """Extract token usage and model metadata from the prompty output dict."""
        return {
            "prompt_tokens": prompty_output.get("input_token_count", 0),
            "completion_tokens": prompty_output.get("output_token_count", 0),
            "total_tokens": prompty_output.get("total_token_count", 0),
            "finish_reason": prompty_output.get("finish_reason", ""),
            "model": prompty_output.get("model_id", ""),
            "sample_input": prompty_output.get("sample_input", ""),
            "sample_output": prompty_output.get("sample_output", ""),
        }

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do a retrieval evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if eval_input.get("response") is not None and isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        result = await self._the_super_do_eval(eval_input)
        # Check if base returned nan (invalid output case); None means not-applicable/skipped
        _score = result.get(self._result_key, 0)
        if _score is not None and math.isnan(_score):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.RETRIEVAL_EVALUATOR,
            )
        return result

    @override
    async def _real_call(self, **kwargs):
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
