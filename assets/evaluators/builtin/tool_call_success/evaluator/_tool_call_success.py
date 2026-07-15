# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import logging
from enum import Enum
from typing import Dict, Union, List, Optional
from typing_extensions import overload, override
from azure.ai.evaluation._exceptions import (
    EvaluationException,
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
)
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

from azure.ai.evaluation._common.utils import _format_value
# ConversationValidator is re-exported for the test suite / capability surface (unused here).
from azure.ai.evaluation._evaluators._common._validators import (  # noqa: F401
    ValidatorInterface,
    ConversationValidator,
    ToolDefinitionsValidator,
)

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
    from azure.ai.evaluation._common.utils import (
        _FAILED_RUNTIME_STATUSES,
        _stringify_tool_result,
        _log_safe_summary,
        _collect_failed_tool_calls,
        _get_tool_calls_results,
        _reformat_tool_calls_results,
    )
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    # Bodies below are copied from azure-ai-evaluation 1.18.1 (the earliest release
    # that ships these symbols).
    _FAILED_RUNTIME_STATUSES = frozenset({"incomplete", "failed"})

    def _stringify_tool_result(result):
        """Render a tool_result value as a string the LLM judge can read.

        Tool outputs arrive in mixed shapes depending on the producer: function/MCP tools usually
        emit a plain ``str``, while built-in grounding tools (``azure_ai_search``, ``azure_fabric``,
        ``sharepoint_grounding``) emit a list/dict. Falling back to ``f"{result}"`` for the latter
        produced a Python ``repr`` (single quotes, trailing commas) that the LLM had to
        reverse-engineer. Strings are passed through unchanged (zero behavior change for function/MCP
        tools), anything else is serialized as JSON with ``default=str`` so non-JSON-native values do
        not raise, and ``None`` renders as the empty string.

        :param result: The raw tool_result value.
        :type result: Any
        :return: A string representation suitable for an LLM prompt.
        :rtype: str
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)

    def _log_safe_summary(obj):
        """Return a non-sensitive structural summary of a payload for safe logging.

        The raw payload may contain customer-controlled data (tool arguments, tool results, assistant
        text, database rows, file content, etc.) which can include credentials or PII. Logging the
        payload itself risks leaking that data into telemetry sinks at any log level. This helper
        returns shape-only metadata - type, length, top-level keys/roles - which is sufficient to
        diagnose schema drift without exposing values.

        :param obj: The payload to summarize.
        :type obj: Any
        :return: A shape-only, non-sensitive summary string.
        :rtype: str
        """
        try:
            type_name = type(obj).__name__
            if isinstance(obj, list):
                roles = []
                for item in obj[:10]:
                    if isinstance(item, dict):
                        role = item.get("role")
                        if isinstance(role, str):
                            roles.append(role)
                roles_summary = roles if roles else "n/a"
                return f"type={type_name} len={len(obj)} roles={roles_summary}"
            if isinstance(obj, dict):
                keys = sorted(k for k in obj.keys() if isinstance(k, str))[:10]
                return f"type={type_name} top_keys={keys}"
            length = len(obj) if hasattr(obj, "__len__") else "n/a"
            return f"type={type_name} len={length}"
        except Exception:  # pylint: disable=broad-except
            return f"type={type(obj).__name__} (summary unavailable)"

    def _collect_failed_tool_calls(messages):
        """Return ordered, unique tool names whose runtime status indicates failure.

        A tool call is treated as a runtime failure when either its assistant ``tool_call`` content
        block or its matched tool ``tool_result`` content block carries a ``status`` field in
        ``{failed, incomplete}``. This lets callers short-circuit deterministically and skip the LLM
        judge on the failure path. When the failing block carries no resolvable function name, the
        tool ``tool_call_id`` is used as a stable identifier instead.

        :param messages: The list of conversation messages to scan.
        :type messages: Any
        :return: Ordered, de-duplicated list of failed tool names (or ids).
        :rtype: List
        """
        if not isinstance(messages, list):
            return []

        id_to_name = {}
        failed_ids = []
        failed_names_without_id = []

        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for content in msg.get("content", []) or []:
                if not isinstance(content, dict) or content.get("type") != "tool_call":
                    continue
                if "tool_call" in content and "function" in content.get("tool_call", {}):
                    tc = content["tool_call"]
                    name = tc.get("function", {}).get("name", "") or ""
                    call_id = tc.get("id")
                else:
                    name = content.get("name", "") or ""
                    call_id = content.get("tool_call_id")
                if call_id is not None:
                    id_to_name[call_id] = name
                status = content.get("status")
                if isinstance(status, str) and status in _FAILED_RUNTIME_STATUSES:
                    if call_id is not None:
                        failed_ids.append(call_id)
                    elif name:
                        failed_names_without_id.append(name)

        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            call_id = msg.get("tool_call_id")
            for content in msg.get("content", []) or []:
                if not isinstance(content, dict) or content.get("type") != "tool_result":
                    continue
                status = content.get("status")
                if isinstance(status, str) and status in _FAILED_RUNTIME_STATUSES and call_id is not None:
                    failed_ids.append(call_id)

        ordered = []
        seen = set()
        for call_id in failed_ids:
            label = id_to_name.get(call_id) or call_id
            if label and label not in seen:
                seen.add(label)
                ordered.append(label)
        for name in failed_names_without_id:
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _get_tool_calls_results(agent_response_msgs):
        """Extract formatted agent tool calls and results from a response.

        The output uses the ``[TOOL_CALL]`` / ``[TOOL_RESULT]`` line format. Tool results are rendered
        via :func:`_stringify_tool_result` so list/dict grounding outputs are readable JSON.

        :param agent_response_msgs: The agent response messages to scan.
        :type agent_response_msgs: List[dict]
        :return: A list of formatted tool-call/result lines.
        :rtype: List[str]
        """
        agent_response_text = []
        tool_results = {}

        for msg in agent_response_msgs:
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                for content in msg.get("content", []):
                    if content.get("type") == "tool_result":
                        result = content.get("tool_result")
                        tool_results[msg["tool_call_id"]] = f"[TOOL_RESULT] {_stringify_tool_result(result)}"

        for msg in agent_response_msgs:
            if "role" in msg and msg.get("role") == "assistant" and "content" in msg:
                for content in msg.get("content", []):
                    if content.get("type") == "tool_call":
                        if "tool_call" in content and "function" in content.get("tool_call", {}):
                            tc = content.get("tool_call", {})
                            func_name = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", {})
                            tool_call_id = tc.get("id")
                        else:
                            tool_call_id = content.get("tool_call_id")
                            func_name = content.get("name", "")
                            args = content.get("arguments", {})
                        args_str = ", ".join(f"{k}={_format_value(v)}" for k, v in args.items())
                        call_line = f"[TOOL_CALL] {func_name}({args_str})"
                        agent_response_text.append(call_line)
                        if tool_call_id in tool_results:
                            agent_response_text.append(tool_results[tool_call_id])

        return agent_response_text

    def _reformat_tool_calls_results(response, logger=None):
        """Reformat an agent response into tool-call/result lines, with a safe fallback.

        :param response: The agent response to reformat.
        :type response: Union[None, List[dict], str]
        :param logger: Optional logger for warning messages.
        :type logger: Optional[logging.Logger]
        :return: The formatted string, or the original response if parsing fails.
        :rtype: Union[str, List[dict]]
        """
        try:
            if response is None or response == []:
                return ""
            agent_response = _get_tool_calls_results(response)
            if agent_response == []:
                if logger:
                    logger.warning(
                        "Empty agent response extracted, likely due to input schema change. "
                        "Falling back to using the original response. %s",
                        _log_safe_summary(response),
                    )
                return response
            return "\n".join(agent_response)
        except Exception as e:  # pylint: disable=broad-except
            if logger:
                logger.warning(
                    "Agent response could not be parsed, falling back to original response. Error: %s. %s",
                    e,
                    _log_safe_summary(response),
                )
            return response


logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TOOL_CALL_SUCCESS_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TOOL_CALL_SUCCESS_EVALUATOR"] = "ToolCallSuccessEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


@experimental
class ToolCallSuccessEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Tool Call Success evaluator determines whether tool calls done by an AI agent includes failures or not.

    This evaluator focuses solely on tool call results and tool definitions, disregarding user's query to
    the agent, conversation history and agent's final response. Although tool definitions is optional,
    providing them can help the evaluator better understand the context of the tool calls made by the
    agent. Please note that this evaluator validates tool calls for potential technical failures like
    errors, exceptions, timeouts and empty results (only in cases where empty results could indicate a
    failure). It does not assess the correctness or the tool result itself, like mathematical errors and
    unrealistic field values like name="668656".

    Scoring is binary:
    - TRUE: All tool calls were successful
    - FALSE: At least one tool call failed

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START TOOL_CALL_SUCCESS_EVALUATOR]
            :end-before: [END TOOL_CALL_SUCCESS_EVALUATOR]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolCallSuccessEvaluator with a tool definitions and response.

    .. admonition:: Example using Azure AI Project URL:

    .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
        :start-after: [START TOOL_CALL_SUCCESS_EVALUATOR]
        :end-before: [END TOOL_CALL_SUCCESS_EVALUATOR]
        :language: python
        :dedent: 8
        :caption: Initialize and call ToolCallSuccessEvaluator using Azure AI Project URL in the following
            format https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "tool_call_success.prompty"
    _RESULT_KEY = "tool_call_success"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/tool_call_success"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        """Initialize the Tool Call Success evaluator."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", 1)
        higher_is_better_value = kwargs.pop("_higher_is_better", True)

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            check_for_unsupported_tools=True,
        )
        # azure_ai_search, azure_fabric and sharepoint_grounding are supported by this
        # evaluator. They were removed from the SDK's UNSUPPORTED_TOOLS list in
        # azure-ai-evaluation >= 1.18.1 but are still listed on 1.17.x, so we drop them
        # from this validator instance to keep the behavior consistent across SDK
        # versions. This override can be removed once 1.17.x is no longer supported.
        self._validator.UNSUPPORTED_TOOLS = [
            tool
            for tool in self._validator.UNSUPPORTED_TOOLS
            if tool not in ("azure_ai_search", "azure_fabric", "sharepoint_grounding")
        ]

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold_value,
            credential=credential,
            _higher_is_better=higher_is_better_value,
            **kwargs,
        )

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
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
                                self._threshold.get(base_key)
                                if isinstance(self._threshold, dict)
                                else self._threshold
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
                                        result[result_key] = (
                                            EVALUATION_PASS_FAIL_MAPPING[True]
                                        )
                                    else:
                                        result[result_key] = (
                                            EVALUATION_PASS_FAIL_MAPPING[False]
                                        )
                                else:
                                    if float(score_value) <= threshold_value:
                                        result[result_key] = (
                                            EVALUATION_PASS_FAIL_MAPPING[True]
                                        )
                                    else:
                                        result[result_key] = (
                                            EVALUATION_PASS_FAIL_MAPPING[False]
                                        )
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

    @overload
    def __call__(
        self,
        *,
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate tool call success for a given response, and optionally tool definitions.

        Example with list of messages:
            evaluator = ToolCallSuccessEvaluator(model_config)
            response = [{'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant',
            'content': [{'type': 'text', 'text': '**Day 1:** Morning: Visit Louvre Museum (9 AM - 12 PM)...'}]}]

            result = evaluator(response=response, )

        :keyword response: The response being evaluated, either a string or a list of messages (full agent
            response potentially including tool calls)
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: Optional tool definitions to use for evaluation.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :return: A dictionary with the tool success evaluation results.
        :rtype: Dict[str, Union[str, float]]
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
        result.update(
            {
                f"{self._result_key}_{key}": value
                for key, value in token_metadata.items()
            }
        )
        return result

    def _return_short_circuit_failure_result(
        self, failed_tools: List[str]
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a deterministic fail result without invoking the LLM judge.

        Used when the runtime explicitly marks one or more tool calls as
        failed/incomplete via the ``status`` field on a ``tool_call`` or
        ``tool_result`` content block. The LLM call is skipped because the
        runtime signal is authoritative; token-metadata fields are emitted
        with zero/empty values for schema compatibility with the LLM path.
        """
        failed_list = ",".join(failed_tools)
        reason = (
            f"Tool call(s) [{failed_list}] reported a non-success runtime status "
            "(failed or incomplete)."
        )
        token_metadata = self._get_token_metadata({})
        result = {
            self._result_key: 0.0,
            f"{self._result_key}_score": 0.0,
            f"{self._result_key}_passed": False,
            f"{self._result_key}_result": "fail",
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": "completed",
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": {
                "failed_tools": failed_list,
                **token_metadata,
            },
        }
        # Add top-level token metadata fields for backward compatibility with the LLM path.
        result.update(
            {
                f"{self._result_key}_{key}": value
                for key, value in token_metadata.items()
            }
        )
        return result

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
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[str, float]]:  # type: ignore[override]
        """Do Tool Call Success evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are
        needed for the _flow method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "response" not in eval_input:
            raise EvaluationException(
                message="response, is a required inputs to the Tool Call Success evaluator.",
                internal_message="response, is a required inputs to the Tool Call Success evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            )
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if eval_input["response"] is None or eval_input["response"] == []:
            raise EvaluationException(
                message="response cannot be None or empty for the Tool Call Success evaluator.",
                internal_message="response cannot be None or empty for the Tool Call Success evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            )

        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
            # Short-circuit: when the runtime explicitly marks any tool_call or
            # tool_result with a non-success status (e.g. ``failed`` or
            # ``incomplete``) there is no point asking the LLM judge to
            # re-derive the failure from the payload -- the runtime signal is
            # authoritative. Return a deterministic fail result and skip the
            # LLM call entirely. The prompty rubric is now only consulted on
            # the success path (status ``completed`` or absent).
            failed_tools = _collect_failed_tool_calls(eval_input["response"])
            if failed_tools:
                return self._return_short_circuit_failure_result(failed_tools)
            eval_input["tool_calls"] = _reformat_tool_calls_results(
                eval_input["response"], logger
            )
        # If response is a string, pass directly without reformatting
        elif isinstance(eval_input["response"], str):
            eval_input["tool_calls"] = eval_input["response"]
        else:
            raise EvaluationException(
                message="response must be either a list of messages or a string.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            )

        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        # If tool definitions are string, pass directly without reformatting, else format it.
        if "tool_definitions" in eval_input and not isinstance(
            eval_input["tool_definitions"], str
        ):
            tool_definitions = eval_input["tool_definitions"]
            # Only if response is not a string, we filter tool definitions to only tools needed.
            if not isinstance(eval_input["response"], str):
                tool_definitions = _filter_to_used_tools(
                    tool_definitions=tool_definitions,
                    msgs_list=eval_input["response"],
                    logger=logger,
                )
            eval_input["tool_definitions"] = _reformat_tool_definitions(
                tool_definitions, logger
            )

        prompty_output_dict = await self._flow(
            timeout=self._LLM_CALL_TIMEOUT, **eval_input
        )
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            # Handle skipped status from LLM
            llm_status = llm_output.get("status", "completed")
            if llm_status == "skipped":
                reason = llm_output.get("reason", "")
                return self._return_not_applicable_result(reason, self._threshold)

            llm_properties = llm_output.get("properties", {}) or {}

            score = float(llm_output.get("score", 0))
            success_result = "pass" if score >= 1.0 else "fail"
            reason = llm_output.get("reason", "")
            token_metadata = self._get_token_metadata(prompty_output_dict)
            llm_properties.update(token_metadata)
            result = {
                self._result_key: score,
                f"{self._result_key}_score": score,
                f"{self._result_key}_passed": success_result == "pass",
                f"{self._result_key}_result": success_result,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_status": "completed",
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_properties": llm_properties,
            }
            # Add top-level token metadata fields for backward compatibility.
            result.update(
                {
                    f"{self._result_key}_{key}": value
                    for key, value in token_metadata.items()
                }
            )
            return result
        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
        )


def _filter_to_used_tools(tool_definitions, msgs_list, logger=None):
    """Filter the tool definitions to only include those that were actually used in the messages lists."""
    try:
        used_tool_names = set()
        any_tools_used = False

        for msg in msgs_list:
            if msg.get("role") == "assistant" and "content" in msg:
                for content in msg.get("content", []):
                    if content.get("type") == "tool_call":
                        any_tools_used = True
                        if (
                            "tool_call" in content
                            and "function" in content["tool_call"]
                        ):
                            used_tool_names.add(content["tool_call"]["function"])
                        elif "name" in content:
                            used_tool_names.add(content["name"])

        filtered_tools = [
            tool for tool in tool_definitions if tool.get("name") in used_tool_names
        ]
        if any_tools_used and not filtered_tools:
            if logger:
                logger.warning(
                    "No tool definitions matched the tools used in the messages. Returning original list."
                )
            filtered_tools = tool_definitions

        return filtered_tools
    except Exception as e:
        if logger:
            logger.warning(
                f"Failed to filter tool definitions, returning original list. Error: {e}"
            )
        return tool_definitions


def _reformat_tool_definitions(tool_definitions, logger=None):
    try:
        output_lines = ["TOOL_DEFINITIONS:"]
        for tool in tool_definitions:
            name = tool.get("name", "unnamed_tool")
            desc = tool.get("description", "").strip()
            params = tool.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys()) if params else "no parameters"
            output_lines.append(f"- {name}: {desc} (inputs: {param_names})")
        return "\n".join(output_lines)
    except Exception as e:
        # If the tool definitions cannot be parsed for whatever reason, the original tool definitions are returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(
                "Tool definitions could not be parsed; falling back to raw definitions. "
                "Input shape: %s. Error: %s",
                _log_safe_summary(tool_definitions),
                e,
            )
        return tool_definitions
