# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import logging
from enum import Enum
from typing import Dict, Union, List

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import (
    EvaluationException,
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
    ErrorMessage,
)
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import _extract_text_from_content
from azure.ai.evaluation._common._experimental import experimental

logger = logging.getLogger(__name__)


# ``` updated _exceptions.py
# Extend ErrorTarget enum if needed
def _create_extended_error_target(ErrorTarget):
    """Create an extended ErrorTarget enum that includes TOOL_OUTPUT_UTILIZATION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TOOL_OUTPUT_UTILIZATION_EVALUATOR"] = "ToolOutputUtilizationEvaluator"

    ErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ErrorTarget


ErrorTarget = _create_extended_error_target(ErrorTarget)
# ```


# ``` updated utils.py
def _filter_to_used_tools(tool_definitions, msgs_lists, logger=None):
    """Filter the tool definitions to only include those that were actually used in the messages lists."""
    try:
        used_tool_names = set()
        any_tools_used = False
        for msgs in msgs_lists:
            for msg in msgs:
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


def _get_conversation_history(query, include_system_messages=False, include_tool_messages=False):
    """Parse conversation history from a list of messages into structured format.

    :param query: List of message dictionaries containing the conversation history
    :type query: List[dict]
    :param include_system_messages: Whether to include system messages in the output
    :type include_system_messages: bool
    :param include_tool_messages: Whether to include tool-related messages in agent responses
    :type include_tool_messages: bool
    :return: Dict containing parsed user_queries, agent_responses, and optionally system_message
    :rtype: Dict[str, Union[List[List[str]], str]]
    :raises EvaluationException: If conversation history is malformed (mismatched user/agent turns
    """
    all_user_queries, all_agent_responses = [], []
    cur_user_query, cur_agent_response = [], []
    system_message = None

    for msg in query:
        role = msg.get("role")
        if not role:
            continue
        if include_system_messages and role == "system":
            system_message = msg.get("content", "")

        elif role == "user" and "content" in msg:
            if cur_agent_response:
                formatted_agent_response = _get_agent_response(
                    cur_agent_response, include_tool_messages=include_tool_messages
                )
                all_agent_responses.append([formatted_agent_response])
                cur_agent_response = []
            text_in_msg = _extract_text_from_content(msg["content"])
            if text_in_msg:
                cur_user_query.append(text_in_msg)

        elif role in ("assistant", "tool"):
            if cur_user_query:
                all_user_queries.append(cur_user_query)
                cur_user_query = []
            cur_agent_response.append(msg)

    if cur_user_query:
        all_user_queries.append(cur_user_query)
    if cur_agent_response:
        formatted_agent_response = _get_agent_response(cur_agent_response, include_tool_messages=include_tool_messages)
        all_agent_responses.append([formatted_agent_response])

    if len(all_user_queries) != len(all_agent_responses) + 1:
        raise EvaluationException(
            message=ErrorMessage.MALFORMED_CONVERSATION_HISTORY,
            internal_message=ErrorMessage.MALFORMED_CONVERSATION_HISTORY,
            target=ErrorTarget.CONVERSATION_HISTORY_PARSING,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )

    result = {"user_queries": all_user_queries, "agent_responses": all_agent_responses}
    if include_system_messages and system_message:
        result["system_message"] = system_message
    return result


def _pretty_format_conversation_history(conversation_history):
    """Format the conversation history for better readability."""
    formatted_history = ""
    if conversation_history.get("system_message"):
        formatted_history += "SYSTEM_PROMPT:\n"
        formatted_history += "  " + conversation_history["system_message"] + "\n\n"
    for i, (user_query, agent_response) in enumerate(
        zip(
            conversation_history["user_queries"],
            conversation_history["agent_responses"] + [None],
        )
    ):
        formatted_history += f"User turn {i+1}:\n"
        for msg in user_query:
            formatted_history += "  " + "\n  ".join(msg)
        formatted_history += "\n\n"
        if agent_response:
            formatted_history += f"Agent turn {i+1}:\n"
            for msg in agent_response:
                formatted_history += "  " + "\n  ".join(msg)
            formatted_history += "\n\n"
    return formatted_history


def reformat_conversation_history(query, logger=None, include_system_messages=False, include_tool_messages=False):
    """Reformats the conversation history to a more compact representation."""
    try:
        conversation_history = _get_conversation_history(
            query,
            include_system_messages=include_system_messages,
            include_tool_messages=include_tool_messages,
        )
        return _pretty_format_conversation_history(conversation_history)
    except Exception as e:
        # If the conversation history cannot be parsed for whatever reason, the original query is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # However the accuracy of the evaluation will be affected.
        # From our tests the negative impact on IntentResolution is:
        #   Higher intra model variance (0.142 vs 0.046)
        #   Higher inter model variance (0.345 vs 0.607)
        #   Lower percentage of mode in Likert scale (73.4% vs 75.4%)
        #   Lower pairwise agreement between LLMs (85% vs 90% at the pass/fail level with threshold of 3)
        if logger:
            logger.warning(f"Conversation history could not be parsed, falling back to original query: {query}")
            print(e)
        return query


def _get_agent_response(agent_response_msgs, include_tool_messages=False):
    """Extract formatted agent response including text, and optionally tool calls/results."""
    agent_response_text = []
    tool_results = {}

    # First pass: collect tool results
    if include_tool_messages:
        for msg in agent_response_msgs:
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                for content in msg.get("content", []):
                    if content.get("type") == "tool_result":
                        result = content.get("tool_result")
                        tool_results[msg["tool_call_id"]] = f"[TOOL_RESULT] {result}"

    # Second pass: parse assistant messages and tool calls
    for msg in agent_response_msgs:
        if "role" in msg and msg.get("role") == "assistant" and "content" in msg:
            text = _extract_text_from_content(msg["content"])
            if text:
                agent_response_text.extend(text)
            if include_tool_messages:
                for content in msg.get("content", []):
                    # Todo: Verify if this is the correct way to handle tool calls
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
                        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                        call_line = f"[TOOL_CALL] {func_name}({args_str})"
                        agent_response_text.append(call_line)
                        if tool_call_id in tool_results:
                            agent_response_text.append(tool_results[tool_call_id])

    return agent_response_text


def reformat_agent_response(response, logger=None, include_tool_messages=False):
    """Reformat agent response to a standardized string format.

    :param response: The agent response to reformat, can be None, empty list, or list of messages
    :type response: Union[None, List[dict], str]
    :param logger: Optional logger for warning messages
    :type logger: Optional[logging.Logger]
    :param include_tool_messages: Whether to include tool call and result information
    :type include_tool_messages: bool
    :return: Formatted agent response as a string, or original response if parsing fails
    :rtype: str
    """
    try:
        if response is None or response == []:
            return ""
        agent_response = _get_agent_response(response, include_tool_messages=include_tool_messages)
        if agent_response == []:
            # If no message could be extracted, fallback to the original response in that case
            if logger:
                logger.warning(
                    "Empty agent response extracted, likely due to input schema change. "
                    f"Falling back to using the original response: {response}"
                )
            return response
        return "\n".join(agent_response)
    except Exception as e:
        # If the agent response cannot be parsed for whatever reason (e.g. the converter format changed),
        # the original response is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(f"Agent response could not be parsed, falling back to original response. Error: {e}")
        return response


def reformat_tool_definitions(tool_definitions, logger=None):
    """Reformat tool definitions into a human-readable string format.

    :param tool_definitions: List of tool definition dictionaries containing name, description, and parameters
    :type tool_definitions: List[dict]
    :param logger: Optional logger for warning messages
    :type logger: Optional[logging.Logger]
    :return: Formatted tool definitions as a string, or original definitions if parsing fails
    :rtype: str
    """
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
                "Tool definitions could not be parsed, falling back to original definitions"
                f": {tool_definitions}. Error: {e}"
            )
        return tool_definitions


# ```


@experimental
class ToolOutputUtilizationEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """Evaluate how effectively an AI agent uses tool outputs.

    This evaluator checks whether the agent correctly incorporates information from tools into its responses.

    Scoring is based on two levels:
    1. Pass - effectively utilizes tool outputs and accurately incorporates the information into its response.
    2. Fail - fails to properly utilize tool outputs or incorrectly incorporates the information into its response.

    The evaluation includes the score, a brief explanation, and a final pass/fail result.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START tool_output_utilization_evaluator]
            :end-before: [END tool_output_utilization_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolOutputUtilizationEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START tool_output_utilization_evaluator]
            :end-before: [END tool_output_utilization_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call ToolOutputUtilizationEvaluator
                using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}
    """

    _PROMPTY_FILE = "tool_output_utilization.prompty"
    _RESULT_KEY = "tool_output_utilization"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _DEFAULT_TOOL_OUTPUT_UTILIZATION_SCORE = 3

    id = "azureai://built-in/evaluators/tool_output_utilization"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(
        self,
        model_config,
        *,
        threshold=_DEFAULT_TOOL_OUTPUT_UTILIZATION_SCORE,
        credential=None,
        **kwargs,
    ):
        """Initialize the Tool Output Utilization Evaluator."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self.threshold = threshold
        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Union[dict, List[dict]],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate tool output utilization for a given query, response, and optional tool defintions.

        The query and response can be either a string or a list of messages.
        Example with string inputs and no tools:
            evaluator = ToolOutputUtilizationEvaluator(model_config)
            query = "What is the weather today?"
            response = "The weather is sunny."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = ToolOutputUtilizationEvaluator(model_config)
            query = [
                {
                    "role": "system",
                    "content": "You are a helpful customer service assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hi, can you check the status of my last order?",
                        }
                    ],
                },
            ]

            response = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Sure! Let me look that up for you."}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call": {
                                "id": "tool_1",
                                "type": "function",
                                "function": {
                                    "name": "get_order_status",
                                    "arguments": {"order_id": "123"},
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"order_id": "123", "status": "shipped"}',
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your order 123 has been shipped and is on its way!",
                        }
                    ],
                },
            ]

            tool_definitions = [
                {
                    "name": "get_order_status",
                    "description": "Retrieve the status of an order by its ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "The order ID to check.",
                            }
                        },
                    },
                }
            ]


            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        (full agent response potentially including tool calls)
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :return: A dictionary with the tool output utilization evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Invoke the instance using the overloaded __call__ signature.

        For detailed parameter types and return value documentation, see the overloaded __call__ definition.
        """
        return super().__call__(*args, **kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do Tool Output Utilization evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are needed for the _flow
            method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # we override the _do_eval method as we want the output to be a dictionary,
        # which is a different schema than _base_prompty_eval.py
        if ("query" not in eval_input) and ("response" not in eval_input) and ("tool_definitions" not in eval_input):
            raise EvaluationException(
                message=(
                    "Query, response, and tool_definitions are required inputs to "
                    "the Tool Output Utilization evaluator."
                ),
                internal_message=(
                    "Query, response, and tool_definitions are required inputs "
                    "to the Tool Output Utilization evaluator."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            )

        tool_definitions = eval_input["tool_definitions"]
        filtered_tool_definitions = _filter_to_used_tools(
            tool_definitions=tool_definitions,
            msgs_lists=[eval_input["query"], eval_input["response"]],
            logger=logger,
        )
        eval_input["tool_definitions"] = reformat_tool_definitions(filtered_tool_definitions, logger)

        eval_input["query"] = reformat_conversation_history(
            eval_input["query"],
            logger,
            include_system_messages=True,
            include_tool_messages=True,
        )
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)

        llm_output = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        if isinstance(llm_output, dict):
            output_label = llm_output.get("label", None)
            if output_label is None:
                if logger:
                    logger.warning("LLM output does not contain 'label' key, returning NaN for the score.")
                output_label = "fail"

            output_label = output_label.lower()
            if output_label not in ["pass", "fail"]:
                if logger:
                    logger.warning(
                        (
                            f"LLM output label is not 'pass' or 'fail' (got '{output_label}'), "
                            "returning NaN for the score."
                        )
                    )

            score = 1.0 if output_label == "pass" else 0.0
            score_result = output_label
            reason = llm_output.get("reason", "")

            faulty_details = llm_output.get("faulty_details", [])
            if faulty_details:
                reason += " Issues found: " + "; ".join(faulty_details)

            return {
                f"{self._result_key}": score,
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_reason": reason,
            }
        if logger:
            logger.warning("LLM output is not a dictionary, returning NaN for the score.")
        return {self._result_key: math.nan}
