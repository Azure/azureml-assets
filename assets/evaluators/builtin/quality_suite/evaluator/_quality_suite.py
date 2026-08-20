# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality Evaluation Suite composite evaluator."""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from typing_extensions import overload, override

from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    reformat_tool_definitions,
    validate_model_config,
)
from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ConversationValidator,
    ToolDefinitionsValidator,
)

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

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

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._evaluators._common._validators import MessageRole
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    # azure-ai-evaluation 1.18.1 MessageRole; the 1.17.x SDK enum omits DEVELOPER,
    # which serialize_messages below relies on.
    class MessageRole(str, Enum):
        """Valid message roles in conversations."""

        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        TOOL = "tool"
        DEVELOPER = "developer"

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.constants import EvaluationLevel
    from azure.ai.evaluation._common.utils import (
        _resolve_evaluation_level,
        _wrap_string_messages,
        _merge_query_response_messages,
        _split_messages_at_latest_user,
        serialize_messages,
    )
    from azure.ai.evaluation._evaluators._common._validators import MessagesOrQueryResponseInputValidator
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    # Bodies below are copied from azure-ai-evaluation 1.18.1 (the earliest release
    # that ships these symbols). The only change is that serialize_messages uses the
    # module-level MessageRole above so the DEVELOPER role stays available on 1.17.x.
    from azure.ai.evaluation._common.utils import (
        _extract_text_from_content,
        _get_agent_response,
        _pretty_format_conversation_history,
    )

    class EvaluationLevel(str, Enum):
        """Supported evaluation levels for multi-turn evaluators."""

        CONVERSATION = "conversation"
        TURN = "turn"

    def _merge_query_response_messages(query: List[dict], response: List[dict]) -> List[dict]:
        """Merge query and response message lists into a single conversation."""
        return [*query, *response]

    def _split_messages_at_latest_user(messages: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Split messages into query/response slices at the latest user turn."""
        latest_user_index = max(
            (i for i, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        if latest_user_index == -1:
            raise ValueError("messages must contain at least one message with role 'user'.")
        return messages[:latest_user_index + 1], messages[latest_user_index + 1:]

    def _wrap_string_messages(query: str, response: str) -> Tuple[List[dict], List[dict]]:
        """Wrap string query/response into separate message lists."""
        return (
            [{"role": "user", "content": [{"type": "text", "text": query}]}],
            [{"role": "assistant", "content": [{"type": "text", "text": response}]}],
        )

    def _resolve_evaluation_level(
        evaluation_level: Optional[Union[EvaluationLevel, str]],
        error_target: ErrorTarget,
    ) -> Optional[EvaluationLevel]:
        """Validate and normalize the evaluation level."""
        valid = [level.value for level in EvaluationLevel]
        if evaluation_level is None or evaluation_level == "":
            return None
        if isinstance(evaluation_level, EvaluationLevel):
            return evaluation_level
        if isinstance(evaluation_level, str):
            try:
                return EvaluationLevel(evaluation_level)
            except ValueError as exc:
                raise EvaluationException(
                    message=f"Invalid evaluation_level '{evaluation_level}'. Must be one of: {valid}.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=error_target,
                ) from exc
        raise EvaluationException(
            message=f"Invalid evaluation_level '{evaluation_level}'. Must be one of: {valid}.",
            blame=ErrorBlame.USER_ERROR,
            category=ErrorCategory.INVALID_VALUE,
            target=error_target,
        )

    def serialize_messages(messages):
        """Serialize chat messages into the labeled transcript used by the multi-turn prompt."""
        if not messages:
            return ""

        all_user_queries = []
        all_agent_responses = []
        cur_user_query = []
        cur_agent_response = []
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
                text_in_msg = [content] if isinstance(content, str) else _extract_text_from_content(content)
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
            "agent_responses": all_agent_responses[: len(all_user_queries) - 1] if all_user_queries else [],
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

    class MessagesOrQueryResponseInputValidator(ToolDefinitionsValidator):
        """Validate query/response or messages inputs, optionally checking tool definitions."""

        enforce_tool_definitions: bool = False

        def __init__(
            self,
            error_target: ErrorTarget,
            requires_query: bool = True,
            optional_tool_definitions: bool = True,
            check_for_unsupported_tools: bool = False,
            *,
            enforce_tool_definitions: bool = False,
        ):
            """Initialize the messages or query/response input validator."""
            super().__init__(error_target, requires_query, optional_tool_definitions, check_for_unsupported_tools)
            self.enforce_tool_definitions = enforce_tool_definitions

        @override
        def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
            """Validate evaluation input, supporting messages as an alternative to query/response."""
            messages = eval_input.get("messages")
            if messages is not None:
                if not isinstance(messages, list):
                    raise EvaluationException(
                        message="messages must be provided as a list of message dictionaries.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                if len(messages) == 0:
                    raise EvaluationException(
                        message="messages list must not be empty.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )

                valid_roles = {role.value for role in MessageRole}
                roles_present: set = set()
                for index, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise EvaluationException(
                            message=(
                                "Each item in 'messages' must be a dictionary, "
                                f"but item at index {index} is {type(message).__name__}."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    role = message.get("role")
                    if role is None:
                        raise EvaluationException(
                            message=(
                                "Each message must contain a 'role' key, "
                                f"but message at index {index} is missing it."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    if role not in valid_roles:
                        raise EvaluationException(
                            message=(
                                f"Invalid role '{role}' at message index {index}. "
                                f"Must be one of: {sorted(valid_roles)}."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    roles_present.add(role)

                if self.requires_query and MessageRole.USER.value not in roles_present:
                    raise EvaluationException(
                        message="messages must contain at least one message with role 'user'.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                if MessageRole.ASSISTANT.value not in roles_present:
                    raise EvaluationException(
                        message="messages must contain at least one message with role 'assistant'.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )

                if self.enforce_tool_definitions:
                    tool_definitions_validation_exception = self._validate_tool_definitions(
                        eval_input.get("tool_definitions")
                    )
                    if tool_definitions_validation_exception:
                        raise tool_definitions_validation_exception
                return True

            if self.enforce_tool_definitions:
                return super().validate_eval_input(eval_input)
            return ConversationValidator.validate_eval_input(self, eval_input)


if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty  # type: ignore[reportMissingImports]
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty


logger = logging.getLogger(__name__)


def _create_extended_error_target():
    """Create an extended ErrorTarget enum for QualityEvaluationSuite."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["QUALITY_EVALUATION_SUITE"] = "QualityEvaluationSuite"
    return Enum("ExtendedErrorTarget", existing_members)


ExtendedErrorTarget = _create_extended_error_target()

# Ordered description of the six member evaluators batched into one LLM call.
_EVALUATORS: Tuple[Dict[str, Union[str, int]], ...] = (
    {"name": "fluency", "min": 1, "max": 5, "default_threshold": 3},
    {"name": "coherence", "min": 1, "max": 5, "default_threshold": 3},
    {"name": "intent_resolution", "min": 1, "max": 5, "default_threshold": 3},
    {"name": "task_adherence", "min": 0, "max": 1, "default_threshold": 1},
    {"name": "groundedness", "min": 1, "max": 5, "default_threshold": 3},
    {"name": "task_completion", "min": 0, "max": 1, "default_threshold": 1},
)


@experimental
class QualityEvaluationSuite(PromptyEvaluatorBase[Union[str, int]]):
    """Batch six quality evaluators into one LLM call.

    The suite preserves the member evaluators' LLM results and derived threshold/pass
    fields exclusively under ``quality_suite_evaluators``. The primary
    ``quality_suite`` result is an any-fail aggregate: it passes only when
    every evaluated member meets its configured threshold.
    """

    _PROMPTY_FILE = "quality_suite.prompty"
    _MULTI_TURN_PROMPTY_FILE = "quality_suite_multi_turn.prompty"
    _RESULT_KEY = "quality_suite"
    _OPTIONAL_PARAMS = ["messages", "tool_definitions"]
    _EVALUATORS = _EVALUATORS

    _validator: ValidatorInterface
    id = "azureai://built-in/evaluators/quality_suite"

    @override
    def __init__(self, model_config, *, credential=None, evaluation_level=None, threshold=None, **kwargs):
        """Initialize the Quality Evaluation Suite."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = {
            evaluator["name"]: evaluator["default_threshold"] for evaluator in self._EVALUATORS
        }
        if isinstance(threshold, dict):
            threshold_value.update({name: value for name, value in threshold.items() if name in threshold_value})

        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ExtendedErrorTarget.QUALITY_EVALUATION_SUITE
        )
        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=ExtendedErrorTarget.QUALITY_EVALUATION_SUITE,
            requires_query=False,
            enforce_tool_definitions=False,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold_value,
            _higher_is_better=True,
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
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Evaluate a single query and response."""

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Evaluate a full multi-turn conversation."""

    @override
    def __call__(self, *args, **kwargs):
        """Invoke the evaluator."""
        return super().__call__(*args, **kwargs)

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether the conversation-level prompt should be used."""
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
        return eval_input.get("messages") is not None

    def _threshold_for(self, name: str) -> Union[int, float]:
        """Return the effective threshold for an evaluator."""
        return self._threshold.get(name) if isinstance(self._threshold, dict) else self._threshold

    def _build_aggregate_result(
        self, evaluators: Dict[str, Dict], token_metadata: Dict
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build the any-fail aggregate and preserve raw evaluator results."""
        evaluated_scores: Dict[str, Union[int, float]] = {}
        skipped_evaluators: List[str] = []
        normalized_evaluators: Dict[str, Dict] = {}
        for evaluator in self._EVALUATORS:
            name = evaluator["name"]
            evaluator_output = dict(evaluators[name])
            threshold = self._threshold_for(name)
            status = evaluator_output.get("status", "completed")
            score = evaluator_output.get("score")
            evaluator_output["threshold"] = threshold
            evaluator_output["passed"] = None
            if status == "skipped" or score is None:
                skipped_evaluators.append(name)
                normalized_evaluators[name] = evaluator_output
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise EvaluationException(
                    message=f"Invalid score value for {name}: {score}.",
                    blame=ErrorBlame.SYSTEM_ERROR,
                    category=ErrorCategory.FAILED_EXECUTION,
                    target=ExtendedErrorTarget.QUALITY_EVALUATION_SUITE,
                )
            if score < evaluator["min"] or score > evaluator["max"]:
                raise EvaluationException(
                    message=(
                        f"Invalid score value for {name}: {score}. "
                        f"Expected a value between {evaluator['min']} and {evaluator['max']}."
                    ),
                    blame=ErrorBlame.SYSTEM_ERROR,
                    category=ErrorCategory.FAILED_EXECUTION,
                    target=ExtendedErrorTarget.QUALITY_EVALUATION_SUITE,
                )
            evaluator_output["passed"] = score >= threshold
            normalized_evaluators[name] = evaluator_output
            evaluated_scores[name] = score

        failed_evaluators = [
            name for name, score in evaluated_scores.items() if score < self._threshold_for(name)
        ]
        if not evaluated_scores:
            aggregate_score = None
            aggregate_result = "not_applicable"
            aggregate_passed = None
            aggregate_status = "skipped"
            aggregate_reason = "Not applicable: all conversation-quality evaluators were skipped."
        else:
            aggregate_score = 0 if failed_evaluators else 1
            aggregate_result = EVALUATION_PASS_FAIL_MAPPING[aggregate_score == 1]
            aggregate_passed = aggregate_score == 1
            aggregate_status = "completed"
            if failed_evaluators:
                aggregate_reason = "Failed conversation-quality evaluators: " + ", ".join(failed_evaluators) + "."
            elif skipped_evaluators:
                aggregate_reason = (
                    "All evaluated conversation-quality evaluators passed; some evaluators were skipped."
                )
            else:
                aggregate_reason = "All conversation-quality evaluators passed."

        aggregate_properties = dict(token_metadata)
        aggregate_properties["failed_evaluators"] = failed_evaluators
        aggregate_properties["skipped_evaluators"] = skipped_evaluators
        result: Dict[str, Union[str, int, float, Dict, None]] = {
            self._RESULT_KEY: aggregate_score,
            f"{self._RESULT_KEY}_score": aggregate_score,
            f"{self._RESULT_KEY}_passed": aggregate_passed,
            f"{self._RESULT_KEY}_result": aggregate_result,
            f"{self._RESULT_KEY}_reason": aggregate_reason,
            f"{self._RESULT_KEY}_status": aggregate_status,
            f"{self._RESULT_KEY}_threshold": 1,
            f"{self._RESULT_KEY}_properties": aggregate_properties,
            f"{self._RESULT_KEY}_evaluators": normalized_evaluators,
        }
        result.update({f"{self._RESULT_KEY}_{key}": value for key, value in token_metadata.items()})
        return result

    def _return_not_applicable_result(self, error_message: str) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a skipped result for all six evaluators."""
        evaluators = {
            evaluator["name"]: {
                "score": None,
                "status": "skipped",
                "reason": f"Not applicable: {error_message}",
            }
            for evaluator in self._EVALUATORS
        }
        return self._build_aggregate_result(evaluators, self._get_token_metadata({}))

    @staticmethod
    def _get_token_metadata(prompty_output: Dict) -> Dict:
        """Extract token usage and model metadata from a prompty output."""
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
    async def _real_call(self, **kwargs):
        """Validate and execute the evaluator."""
        if "response" in kwargs and "query" not in kwargs and "messages" not in kwargs:
            kwargs["query"] = []

        if self._evaluation_level == EvaluationLevel.CONVERSATION and not kwargs.get("messages"):
            query = kwargs.get("query")
            response = kwargs.get("response")
            if isinstance(query, str) and isinstance(response, str) and query and response:
                query, response = _wrap_string_messages(query, response)
            if isinstance(query, list) and isinstance(response, list):
                kwargs["messages"] = _merge_query_response_messages(query, response)
        elif self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            messages = kwargs["messages"]
            if any(isinstance(m, dict) and m.get("role") == MessageRole.USER for m in messages):
                query_messages, response_messages = _split_messages_at_latest_user(messages)
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages
                kwargs.pop("messages", None)
            elif any(isinstance(m, dict) and m.get("role") == MessageRole.ASSISTANT for m in messages):
                kwargs["query"] = []
                kwargs["response"] = messages
                kwargs.pop("messages", None)

        self._validator.validate_eval_input(kwargs)
        return await self._the_super_real_call(**kwargs)

    async def _the_super_real_call(self, **kwargs):
        """Execute one or more evaluation inputs."""
        try:
            eval_input_list = self._convert_kwargs_to_eval_input(**kwargs)
        except Exception as exc:
            logger.error("Error converting kwargs to eval_input_list: %s", exc)
            raise

        per_turn_results = []
        for eval_input in eval_input_list:
            per_turn_results.append(await self._do_eval(eval_input))
        if len(per_turn_results) == 1:
            return per_turn_results[0]
        if not per_turn_results:
            return {}
        return self._aggregate_results(per_turn_results=per_turn_results)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:
        """Evaluate one turn or one full conversation."""
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        if eval_input.get("response") is None:
            raise EvaluationException(
                message=(
                    "A response must be provided as input to the Quality Evaluation Suite."
                ),
                internal_message=(
                    "A response must be provided as input to the Quality Evaluation Suite."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.QUALITY_EVALUATION_SUITE,
            )
        if eval_input.get("query") is None:
            eval_input["query"] = []
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation."
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        eval_input.pop("messages", None)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[int, str]]:
        """Evaluate a full conversation with the multi-turn prompty."""
        messages = _preprocess_messages(eval_input["messages"])
        conversation_text = serialize_messages(messages)
        prompty_kwargs: Dict[str, Any] = {"messages": conversation_text}
        tool_definitions = eval_input.get("tool_definitions")
        if tool_definitions:
            prompty_kwargs["tool_definitions"] = reformat_tool_definitions(tool_definitions, logger)

        prompty_output_dict = await self._multi_turn_flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_kwargs)
        return self._parse_prompty_output(prompty_output_dict, is_multi_turn=True)

    def _parse_prompty_output(
        self, prompty_output_dict: Dict, is_multi_turn: bool = False
    ) -> Dict[str, Union[int, str]]:
        """Parse the LLM response into aggregate and raw evaluator results."""
        del is_multi_turn
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)
        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.QUALITY_EVALUATION_SUITE,
            )

        evaluators: Dict[str, Dict] = {}
        for evaluator in self._EVALUATORS:
            name = evaluator["name"]
            evaluator_output = llm_output.get(name)
            if isinstance(evaluator_output, dict):
                evaluators[name] = evaluator_output
            else:
                evaluators[name] = {
                    "score": None,
                    "status": "skipped",
                    "reason": "Evaluator did not return a result for this evaluator.",
                }
        return self._build_aggregate_result(evaluators, self._get_token_metadata(prompty_output_dict))
