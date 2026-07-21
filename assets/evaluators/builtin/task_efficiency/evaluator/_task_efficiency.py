# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import reformat_tool_definitions
from azure.ai.evaluation._common.utils import (
    _extract_text_from_content,
    _get_agent_response,
    _pretty_format_conversation_history,
)
from azure.ai.evaluation._common._experimental import experimental


# region Validators


class ValidatorInterface(ABC):
    """Abstract base class defining the interface that all validators must implement."""

    @abstractmethod
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate the evaluation input dictionary."""


class MessageRole(str, Enum):
    """Valid message roles in conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    DEVELOPER = "developer"


class ContentType(str, Enum):
    """Valid content types in messages."""

    TEXT = "text"
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    MCP_APPROVAL_REQUEST = "mcp_approval_request"
    MCP_APPROVAL_RESPONSE = "mcp_approval_response"
    OPENAPI_CALL = "openapi_call"
    OPENAPI_CALL_OUTPUT = "openapi_call_output"


class MessagesValidator(ValidatorInterface):
    """Validate that the input contains a non-empty list of message dicts."""

    def __init__(self, error_target: ErrorTarget):
        self.error_target = error_target

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        messages = eval_input.get("messages")
        if messages is None:
            raise EvaluationException(
                message="'messages' is a required input for TaskEfficiencyEvaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=self.error_target,
            )
        if not isinstance(messages, list) or len(messages) == 0:
            raise EvaluationException(
                message="'messages' must be a non-empty list of message dicts.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise EvaluationException(
                    message=f"Message at index {idx} must be a dictionary.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            if "role" not in msg:
                raise EvaluationException(
                    message=f"Message at index {idx} must contain a 'role' field.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.MISSING_FIELD,
                    target=self.error_target,
                )
        return True


# endregion Validators


logger = logging.getLogger(__name__)


def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TASK_EFFICIENCY_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TASK_EFFICIENCY_EVALUATOR"] = "TaskEfficiencyEvaluator"
    return Enum("ExtendedErrorTarget", existing_members)


ExtendedErrorTarget = _create_extended_error_target()


def _drop_mcp_approval_messages(messages):
    """Remove MCP approval request/response messages."""
    if not isinstance(messages, list):
        return messages
    return [
        msg for msg in messages
        if not (
            isinstance(msg, dict)
            and isinstance(msg.get("content"), list)
            and (
                (msg.get("role") == "assistant" and any(
                    isinstance(c, dict) and c.get("type") == "mcp_approval_request" for c in msg["content"]))
                or (msg.get("role") == "tool" and any(
                    isinstance(c, dict) and c.get("type") == "mcp_approval_response" for c in msg["content"]))
            )
        )
    ]


def _normalize_function_call_types(messages):
    """Normalize function_call / openapi_call types to tool_call / tool_result."""
    if not isinstance(messages, list):
        return messages
    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            continue
        for item in msg["content"]:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "function_call":
                item["type"] = "tool_call"
            elif t == "function_call_output":
                item["type"] = "tool_result"
                if "function_call_output" in item:
                    item["tool_result"] = item.pop("function_call_output")
            elif t == "openapi_call":
                item["type"] = "tool_call"
            elif t == "openapi_call_output":
                item["type"] = "tool_result"
                if "openapi_call_output" in item:
                    item["tool_result"] = item.pop("openapi_call_output")
    return messages


def _preprocess_messages(messages):
    """Drop MCP approval messages and normalize function call types."""
    messages = _drop_mcp_approval_messages(messages)
    messages = _normalize_function_call_types(messages)
    return messages


def _count_assistant_tool_calls(messages: List[dict]) -> int:
    """Count assistant tool-call content items across all messages."""
    if not isinstance(messages, list):
        return 0
    tool_call_types = {
        ContentType.TOOL_CALL.value,
        ContentType.FUNCTION_CALL.value,
        ContentType.OPENAPI_CALL.value,
    }
    count = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != MessageRole.ASSISTANT.value:
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in tool_call_types:
                count += 1
    return count


def serialize_messages(messages: List[dict]) -> str:
    """Serialize chat messages into a labeled text transcript for the multi-turn prompty."""
    if not messages:
        return ""

    all_user_queries: List = []
    all_agent_responses: List = []
    cur_user_query: List = []
    cur_agent_response: List = []
    system_message = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role:
            continue

        normalized = msg
        if role == MessageRole.ASSISTANT and isinstance(msg.get("content"), str):
            normalized = {**msg, "content": [{"type": "text", "text": msg["content"]}]}

        if role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
            content = msg.get("content", "")
            if isinstance(content, list):
                system_message = "\n".join(_extract_text_from_content(content))
            else:
                system_message = content

        elif role == MessageRole.USER and "content" in msg:
            if cur_agent_response:
                formatted = _get_agent_response(cur_agent_response, include_tool_messages=True)
                all_agent_responses.append([formatted])
                cur_agent_response = []
            content = msg["content"]
            if isinstance(content, str):
                text_in_msg = [content]
            else:
                text_in_msg = _extract_text_from_content(content)
            if text_in_msg:
                cur_user_query.append(text_in_msg)

        elif role in (MessageRole.ASSISTANT, MessageRole.TOOL):
            if cur_user_query:
                all_user_queries.append(cur_user_query)
                cur_user_query = []
            cur_agent_response.append(normalized)

    if cur_user_query:
        all_user_queries.append(cur_user_query)
    if cur_agent_response:
        formatted = _get_agent_response(cur_agent_response, include_tool_messages=True)
        all_agent_responses.append([formatted])

    conversation_history: Dict = {
        "user_queries": all_user_queries,
        "agent_responses": all_agent_responses[:len(all_user_queries) - 1]
        if len(all_user_queries) > 0
        else [],
    }
    if system_message:
        conversation_history["system_message"] = system_message

    result = _pretty_format_conversation_history(conversation_history)

    start = max(len(all_user_queries) - 1, 0)
    for i, agent_response in enumerate(all_agent_responses[start:], start=start):
        result += f"Agent turn {i + 1}:\n"
        for msg_text in agent_response:
            if isinstance(msg_text, list):
                for submsg in msg_text:
                    result += "  " + "\n  ".join(submsg.split("\n")) + "\n"
            else:
                result += "  " + "\n  ".join(msg_text.split("\n")) + "\n"
        result += "\n"

    return result.rstrip("\n")


@experimental
class TaskEfficiencyEvaluator(PromptyEvaluatorBase[Union[str, int]]):
    """The Task Efficiency evaluator scores how efficient an agent's multi-turn trajectory was.

    It judges the agent's tool-use process, not the correctness of the final answer or whether the
    task was completed. It penalizes redundant tool calls, repeated reads of the same file, identical
    failed-command retries, oscillation between targets, and dead-end exploration after the answer
    was already in context.

    Scoring is on an integer 1-5 rubric:

    - 5: Highly efficient - only the strictly needed steps; nothing redundant.
    - 4: Mostly efficient - one or two minor redundancies; no loops.
    - 3: Moderately efficient - several redundant calls; ~15-35% of steps removable.
    - 2: Inefficient - substantial waste; ~35-60% of steps removable.
    - 1: Highly inefficient - severe looping or thrashing; >60% removable.

    A conversation with zero assistant tool calls is reported with status ``skipped`` and score ``None``.

    :param model_config: Configuration for the Azure OpenAI model used as the judge.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :keyword threshold: Minimum score (1-5) to consider the trajectory a pass. Default is 3.
    :paramtype threshold: int
    """

    _MULTI_TURN_PROMPTY_FILE = "task_efficiency_multi_turn.prompty"
    _RESULT_KEY = "task_efficiency"
    _OPTIONAL_PARAMS = ["tool_definitions"]
    _MIN_SCORE = 1
    _MAX_SCORE = 5

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/task_efficiency"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._MULTI_TURN_PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", 3)

        self._validator = MessagesValidator(error_target=ExtendedErrorTarget.TASK_EFFICIENCY_EVALUATOR)

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold_value,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int]]:
        """Evaluate task efficiency for a full multi-turn conversation.

        :keyword messages: The full multi-turn conversation as a list of message dicts. Must contain
            assistant tool calls; conversations without any tool-use trajectory return status ``skipped``.
        :paramtype messages: List[dict]
        :keyword tool_definitions: Optional list of tool definitions the agent had available.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: Dictionary with the task efficiency evaluation results.
        :rtype: Dict[str, Union[str, int]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Invoke the instance using the overloaded __call__ signature."""
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _get_token_metadata(prompty_output: Dict) -> Dict:
        return {
            "prompt_tokens": prompty_output.get("input_token_count", 0),
            "completion_tokens": prompty_output.get("output_token_count", 0),
            "total_tokens": prompty_output.get("total_token_count", 0),
            "finish_reason": prompty_output.get("finish_reason", ""),
            "model": prompty_output.get("model_id", ""),
            "sample_input": prompty_output.get("sample_input", ""),
            "sample_output": prompty_output.get("sample_output", ""),
        }

    def _return_not_applicable_result(self, error_message: str):
        token_metadata = self._get_token_metadata({})
        result = {
            f"{self._result_key}": None,
            f"{self._result_key}_score": None,
            f"{self._result_key}_passed": None,
            f"{self._result_key}_result": "not_applicable",
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_status": "skipped",
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": None,
        }
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result

    @override
    async def _real_call(self, **kwargs):
        self._validator.validate_eval_input(kwargs)
        # Strip single-turn keys not used by this evaluator.
        kwargs.pop("query", None)
        kwargs.pop("response", None)
        return await self._the_super_real_call(**kwargs)

    async def _the_super_real_call(self, **kwargs):
        try:
            eval_input_list = self._convert_kwargs_to_eval_input(**kwargs)
        except Exception as e:
            logger.error(f"Error converting kwargs to eval_input_list: {e}")
            raise
        per_turn_results = []
        for eval_input in eval_input_list:
            per_turn_results.append(await self._do_eval(eval_input))
        if len(per_turn_results) == 1:
            return per_turn_results[0]
        if len(per_turn_results) == 0:
            return {}
        return self._aggregate_results(per_turn_results=per_turn_results)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:  # type: ignore[override]
        messages = eval_input.get("messages")
        if not messages:
            raise EvaluationException(
                message="'messages' is required for TaskEfficiencyEvaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.TASK_EFFICIENCY_EVALUATOR,
            )

        messages = _preprocess_messages(messages)

        # Short-circuit: efficiency is undefined when there is no tool-use trajectory.
        if _count_assistant_tool_calls(messages) == 0:
            return self._return_not_applicable_result(
                "No assistant tool calls in trajectory; nothing to evaluate for efficiency."
            )

        conversation_text = serialize_messages(messages)
        prompty_kwargs: Dict[str, Any] = {"messages": conversation_text}
        tool_definitions = eval_input.get("tool_definitions")
        if tool_definitions:
            prompty_kwargs["tool_definitions"] = reformat_tool_definitions(tool_definitions, logger)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_kwargs)
        return self._parse_prompty_output(prompty_output_dict)

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Union[int, str]]:
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TASK_EFFICIENCY_EVALUATOR,
            )

        if llm_output.get("status", "completed") == "skipped":
            return self._return_not_applicable_result(llm_output.get("reason", ""))

        score_value = llm_output.get("score")
        if score_value is None:
            raise EvaluationException(
                message="Evaluator returned invalid output: missing 'score'.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TASK_EFFICIENCY_EVALUATOR,
            )
        try:
            score = int(round(float(score_value)))
        except (TypeError, ValueError):
            raise EvaluationException(
                message=f"Evaluator returned invalid output: invalid 'score' value: {score_value}",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TASK_EFFICIENCY_EVALUATOR,
            )
        # Clamp judge output to the declared rubric range.
        score = max(self._MIN_SCORE, min(self._MAX_SCORE, score))

        threshold = self._threshold if isinstance(self._threshold, (int, float)) else 3
        success_result = "pass" if score >= threshold else "fail"
        reason = llm_output.get("reason", "")
        llm_properties = llm_output.get("properties", {}) or {}
        token_metadata = self._get_token_metadata(prompty_output_dict)
        llm_properties.update(token_metadata)
        result = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_passed": success_result == "pass",
            f"{self._result_key}_result": success_result,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": "completed",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_properties": llm_properties,
        }
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result
