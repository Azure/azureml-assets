# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import os
import logging
from typing import Dict, Union, List, Optional
from typing_extensions import overload, override
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common._experimental import experimental
from enum import Enum


logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TOOL_CALL_SUCCESS_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TOOL_CALL_SUCCESS_EVALUATOR"] = "ToolCallSuccessEvaluator"

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
                if isinstance(last_content, dict) and last_content.get("type") in ("function_call", "mcp_approval_request"):
                    return True
    return False


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
    """Normalize function_call/function_call_output types to tool_call/tool_result."""
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
    return messages


def _preprocess_messages(messages):
    """Drop MCP approval messages and normalize function call types."""
    messages = _drop_mcp_approval_messages(messages)
    messages = _normalize_function_call_types(messages)
    return messages


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

    id = "azureai://built-in/evaluators/tool_call_success"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        """Initialize the Tool Call Success evaluator."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", 1)
        higher_is_better_value = kwargs.pop("_higher_is_better", True)
        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold_value,
            credential=credential,
            _higher_is_better=higher_is_better_value,
            **kwargs,
        )

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

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: threshold,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
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
            return self._not_applicable_result(
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
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        # If response is a string, pass directly without reformatting
        if isinstance(eval_input["response"], str):
            # Unless tool calls are explicitly provided, then keep it as is
            if "tool_calls" not in eval_input or not eval_input["tool_calls"]:
                eval_input["tool_calls"] = eval_input["response"]
        else:
            eval_input["tool_calls"] = _reformat_tool_calls_results(eval_input["response"], logger)

        # If tool definitions are string, pass directly without reformatting, else format it.
        if "tool_definitions" in eval_input and not isinstance(eval_input["tool_definitions"], str):
            tool_definitions = eval_input["tool_definitions"]
            # Only if response is not a string, we filter tool definitions to only tools needed.
            if not isinstance(eval_input["response"], str):
                tool_definitions = _filter_to_used_tools(
                    tool_definitions=tool_definitions,
                    msgs_list=eval_input["response"],
                    logger=logger,
                )
            eval_input["tool_definitions"] = _reformat_tool_definitions(tool_definitions, logger)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            success = llm_output.get("success", False)
            details = llm_output.get('details', {})

            if "success" not in llm_output and "success" in details:
                success = details.get("success", False)

            if isinstance(success, str):
                success = success.upper() == "TRUE"

            success_result = "pass" if success else "fail"
            reason = llm_output.get("explanation", "")
            return {
                f"{self._result_key}": success * 1.0,
                f"{self._result_key}_result": success_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": f"{reason} {llm_output.get('details', '')}",
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
        if logger:
            logger.warning("LLM output is not a dictionary, returning NaN for the score.")

        score = math.nan
        binary_result = self._get_binary_result(score)
        return {
            self._result_key: float(score),
            f"{self._result_key}_result": binary_result,
            f"{self._result_key}_threshold": self._threshold,
        }


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
                        if "tool_call" in content and "function" in content["tool_call"]:
                            used_tool_names.add(content["tool_call"]["function"])
                        elif "name" in content:
                            used_tool_names.add(content["name"])

        filtered_tools = [tool for tool in tool_definitions if tool.get("name") in used_tool_names]
        if any_tools_used and not filtered_tools:
            if logger:
                logger.warning("No tool definitions matched the tools used in the messages. Returning original list.")
            filtered_tools = tool_definitions

        return filtered_tools
    except Exception as e:
        if logger:
            logger.warning(f"Failed to filter tool definitions, returning original list. Error: {e}")
        return tool_definitions


def _format_value(v):
    if v is None:
        return "None"
    if isinstance(v, str):
        return f'"{v}"'
    return v


def _get_tool_calls_results(agent_response_msgs):
    """Extract formatted agent tool calls and results from response."""
    agent_response_text = []
    tool_results = {}

    # First pass: collect tool results

    for msg in agent_response_msgs:
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            for content in msg.get("content", []):
                if content.get("type") == "tool_result":
                    result = content.get("tool_result")
                    tool_results[msg["tool_call_id"]] = f"[TOOL_RESULT] {result}"

    # Second pass: parse assistant messages and tool calls
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
                    args_str = ", ".join(f'{k}={_format_value(v)}' for k, v in args.items())
                    call_line = f"[TOOL_CALL] {func_name}({args_str})"
                    agent_response_text.append(call_line)
                    if tool_call_id in tool_results:
                        agent_response_text.append(tool_results[tool_call_id])

    return agent_response_text


def _reformat_tool_calls_results(response, logger=None):
    try:
        if response is None or response == []:
            return ""
        agent_response = _get_tool_calls_results(response)
        if agent_response == []:
            # If no message could be extracted, likely the format changed,
            # fallback to the original response in that case
            if logger:
                logger.warning(
                    f"Empty agent response extracted, likely due to input schema change. "
                    f"Falling back to using the original response: {response}"
                )
            return response
        return "\n".join(agent_response)
    except Exception:
        # If the agent response cannot be parsed for whatever
        # reason (e.g. the converter format changed), the original response is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(f"Agent response could not be parsed, falling back to original response: {response}")
        return response


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
    except Exception:
        # If the tool definitions cannot be parsed for whatever reason, the original tool definitions are returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(
                f"Tool definitions could not be parsed, falling back to original definitions: {tool_definitions}"
            )
        return tool_definitions
