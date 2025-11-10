# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Dict, List, Union, TypeVar
from typing_extensions import override
from itertools import chain
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import (
    ErrorBlame,
    ErrorCategory,
    ErrorMessage,
    ErrorTarget,
    EvaluationException,
)
from azure.ai.evaluation._common._experimental import experimental
from enum import Enum

logger = logging.getLogger(__name__)

T_EvalValue = TypeVar("T_EvalValue")


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TOOL_SELECTION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members['TOOL_SELECTION_EVALUATOR'] = 'ToolSelectionEvaluator'

    ExtendedErrorTarget = Enum('ExtendedErrorTarget', existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


def _extract_tool_names_from_calls(tool_calls: List[Dict]) -> List[str]:
    """Extract just the tool names from tool calls, removing parameters."""
    tool_names = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_type = tool_call.get("type")
            if tool_type == "tool_call":
                tool_name = tool_call.get("name")
                if tool_name:
                    tool_names.append(tool_name)
            elif tool_call.get("function", {}).get("name"):
                # Handle function call format
                tool_names.append(tool_call["function"]["name"])
            elif tool_call.get("name"):
                # Handle direct name format
                tool_names.append(tool_call["name"])
    return tool_names


def _get_built_in_tool_definition(tool_name: str):
    """Get the definition for the built-in tool."""
    try:
        from azure.ai.evaluation._converters._models import _BUILT_IN_DESCRIPTIONS, _BUILT_IN_PARAMS

        if tool_name in _BUILT_IN_DESCRIPTIONS:
            return {
                "type": tool_name,
                "description": _BUILT_IN_DESCRIPTIONS[tool_name],
                "name": tool_name,
                "parameters": _BUILT_IN_PARAMS.get(tool_name, {}),
            }
    except ImportError:
        pass
    return None


def _get_needed_built_in_tool_definitions(tool_calls: List[Dict]) -> List[Dict]:
    """Extract tool definitions needed for the given built-in tool calls."""
    needed_definitions = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_type = tool_call.get("type")

            # Only support converter format: {type: "tool_call", name: "bing_custom_search", arguments: {...}}
            if tool_type == "tool_call":
                tool_name = tool_call.get("name")
                if tool_name:
                    definition = _get_built_in_tool_definition(tool_name)
                    if definition and definition not in needed_definitions:
                        needed_definitions.append(definition)

    return needed_definitions


def _extract_needed_tool_definitions(
        tool_calls: List[Dict], tool_definitions: List[Dict], error_target: ErrorTarget
) -> List[Dict]:
    """Extract the tool definitions that are needed for the provided tool calls.

    :param tool_calls: The tool calls that need definitions
    :type tool_calls: List[Dict]
    :param tool_definitions: User-provided tool definitions
    :type tool_definitions: List[Dict]
    :return: List of needed tool definitions
    :rtype: List[Dict]
    :raises EvaluationException: If validation fails
    """
    needed_tool_definitions = []

    # Add all user-provided tool definitions
    needed_tool_definitions.extend(tool_definitions)

    # Add the needed built-in tool definitions (if they are called)
    built_in_definitions = _get_needed_built_in_tool_definitions(tool_calls)
    needed_tool_definitions.extend(built_in_definitions)

    # OpenAPI tool is a collection of functions, so we need to expand it
    tool_definitions_expanded = list(
        chain.from_iterable(
            tool.get("functions", []) if tool.get("type") == "openapi" else [tool]
            for tool in needed_tool_definitions
        )
    )

    # Validate that all tool calls have corresponding definitions
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_type = tool_call.get("type")

            if tool_type == "tool_call":
                tool_name = tool_call.get("name")
                if tool_name and _get_built_in_tool_definition(tool_name):
                    # This is a built-in tool from converter, already handled above
                    continue
                elif tool_name:
                    # This is a regular function tool from converter
                    tool_definition_exists = any(
                        tool.get("name") == tool_name and tool.get("type", "function") == "function"
                        for tool in tool_definitions_expanded
                    )
                    if not tool_definition_exists:
                        raise EvaluationException(
                            message=f"Tool definition for {tool_name} not found",
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=error_target,
                        )
                else:
                    raise EvaluationException(
                        message=f"Tool call missing name: {tool_call}",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=error_target,
                    )
            else:
                # Unsupported tool format - only converter format is supported
                raise EvaluationException(
                    message=f"Unsupported tool call format. Only converter format is supported: {tool_call}",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=error_target,
                )
        else:
            # Tool call is not a dictionary
            raise EvaluationException(
                message=f"Tool call is not a dictionary: {tool_call}",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=error_target,
            )

    return needed_tool_definitions


def _extract_text_from_content(content):
    text = []
    for msg in content:
        if "text" in msg:
            text.append(msg["text"])
    return text


def _get_conversation_history(query, include_system_messages=False, include_tool_calls=False):
    all_user_queries = []
    cur_user_query = []
    all_agent_responses = []
    cur_agent_response = []
    system_message = None

    # Track tool calls and results for grouping with assistant messages
    tool_results = {}

    # First pass: collect all tool results if include_tool_calls is True
    if include_tool_calls:
        for msg in query:
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                tool_call_id = msg["tool_call_id"]
                for content in msg.get("content", []):
                    if content.get("type") == "tool_result":
                        result = content.get("tool_result")
                        tool_results[tool_call_id] = f"[TOOL_RESULT] {result}"

    # Second pass: process messages and build conversation history
    for msg in query:
        if "role" not in msg:
            continue

        if include_system_messages and msg["role"] == "system" and "content" in msg:
            system_message = msg.get("content", "")

        if msg["role"] == "user" and "content" in msg:
            # Start new user turn, close previous agent response if exists
            if cur_agent_response != []:
                all_agent_responses.append(cur_agent_response)
                cur_agent_response = []
            text_in_msg = _extract_text_from_content(msg["content"])
            if text_in_msg:
                cur_user_query.append(text_in_msg)

        if msg["role"] == "assistant" and "content" in msg:
            # Start new agent response, close previous user query if exists
            if cur_user_query != []:
                all_user_queries.append(cur_user_query)
                cur_user_query = []

            # Add text content
            text_in_msg = _extract_text_from_content(msg["content"])
            if text_in_msg:
                cur_agent_response.append(text_in_msg)

            # Handle tool calls in assistant messages
            if include_tool_calls:
                for content in msg.get("content", []):
                    if content.get("type") == "tool_call":
                        # Handle the format from your sample data
                        tool_call_id = content.get("tool_call_id")
                        func_name = content.get("name", "")
                        args = content.get("arguments", {})

                        # Also handle the nested tool_call format
                        if "tool_call" in content and "function" in content.get("tool_call", {}):
                            tc = content.get("tool_call", {})
                            func_name = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", {})
                            tool_call_id = tc.get("id")

                        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                        tool_call_text = f"[TOOL_CALL] {func_name}({args_str})"
                        cur_agent_response.append(tool_call_text)

                        # Immediately add tool result if available
                        if tool_call_id and tool_call_id in tool_results:
                            cur_agent_response.append(tool_results[tool_call_id])

    # Close any remaining open queries/responses
    if cur_user_query != []:
        all_user_queries.append(cur_user_query)
    if cur_agent_response != []:
        all_agent_responses.append(cur_agent_response)

    if len(all_user_queries) != len(all_agent_responses) + 1:
        raise EvaluationException(
            message=ErrorMessage.MALFORMED_CONVERSATION_HISTORY,
            internal_message=ErrorMessage.MALFORMED_CONVERSATION_HISTORY,
            target=ErrorTarget.CONVERSATION_HISTORY_PARSING,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )
    result = {"user_queries": all_user_queries, "agent_responses": all_agent_responses}
    if include_system_messages:
        result["system_message"] = system_message
    return result


def _pretty_format_conversation_history(conversation_history):
    """Format the conversation history for better readability."""
    formatted_history = ""
    if "system_message" in conversation_history and conversation_history["system_message"] is not None:
        formatted_history += "SYSTEM_PROMPT:\n"
        formatted_history += "  " + conversation_history["system_message"] + "\n\n"
    for i, (user_query, agent_response) in enumerate(
        zip(conversation_history["user_queries"], conversation_history["agent_responses"] + [None])
    ):
        formatted_history += f"User turn {i+1}:\n"
        for msg in user_query:
            if isinstance(msg, list):
                for submsg in msg:
                    formatted_history += "  " + "\n  ".join(submsg.split("\n")) + "\n"
            else:
                formatted_history += "  " + "\n  ".join(msg.split("\n")) + "\n"
        formatted_history += "\n"
        if agent_response:
            formatted_history += f"Agent turn {i+1}:\n"
            for msg in agent_response:
                if isinstance(msg, list):
                    for submsg in msg:
                        formatted_history += "  " + "\n  ".join(submsg.split("\n")) + "\n"
                else:
                    formatted_history += "  " + "\n  ".join(msg.split("\n")) + "\n"
            formatted_history += "\n"
    return formatted_history


def reformat_conversation_history(query, logger=None, include_system_messages=False, include_tool_calls=False):
    """Reformats the conversation history to a more compact representation."""
    try:
        conversation_history = _get_conversation_history(
            query, include_system_messages=include_system_messages, include_tool_calls=include_tool_calls
        )
        return _pretty_format_conversation_history(conversation_history)
    except Exception:
        # If the conversation history cannot be parsed for whatever reason (e.g. the converter format change),
        # the original query is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # However the accuracy of the evaluation will be affected.
        # From our tests the negative impact on IntentResolution is:
        #   Higher intra model variance (0.142 vs 0.046)
        #   Higher inter model variance (0.345 vs 0.607)
        #   Lower percentage of mode in Likert scale (73.4% vs 75.4%)
        #   Lower pairwise agreement between LLMs (85% vs 90% at the pass/fail level with threshold of 3)
        if logger:
            logger.warning(f"Conversation history could not be parsed, falling back to original query: {query}")
        return query


@experimental
class ToolSelectionEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Tool Selection evaluator assesses the appropriateness and efficiency of tool choices made by an AI agent.

       The evaluator assesses the following:
        - Relevance of selected tools to the conversation.
        - Completeness of tool selection according to task requirements.
        - Efficiency in avoiding unnecessary or redundant tools.

    The evaluator uses a binary scoring system:
        - Score 0 (Fail): Tools selected are irrelevant, incorrect, or missing essential tools
        - Score 1 (Pass): All needed tools are selected, even if there are redundant tools

    This evaluation focuses on measuring whether the right tools were chosen for the task,
    regardless of how those tools were executed or their parameter correctness.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START tool_selection_evaluator]
            :end-before: [END tool_selection_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolSelectionEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START tool_selection_evaluator]
            :end-before: [END tool_selection_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call ToolSelectionEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE = "tool_selection.prompty"
    _RESULT_KEY = "tool_selection"

    _NO_TOOL_CALLS_MESSAGE = "No tool calls found in response or provided tool_calls."
    _NO_TOOL_DEFINITIONS_MESSAGE = "Tool definitions must be provided."
    _TOOL_DEFINITIONS_MISSING_MESSAGE = "Tool definitions for all tool calls must be provided."
    _INVALID_SCORE_MESSAGE = "Tool selection score must be 0 or 1."

    id = "azureai://built-in/evaluators/tool_selection"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=1, credential=None, **kwargs):
        """Initialize the Tool Selection evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation. Binary score (0 or 1), so threshold should be 1 for pass.
        :type threshold: int
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        """
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

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate tool selection quality for a given query, tool definitions, and tool calls.

        For detailed parameter types and return value documentation, see the class documentation.
        """
        return super().__call__(*args, **kwargs)

    def _convert_kwargs_to_eval_input(self, **kwargs):
        """Convert an arbitrary input into a list of inputs for evaluators.

        It is assumed that evaluators generally make use of their inputs in one of two ways.
        Either they receive a collection of keyname inputs that are all single values
        (like a query and response), or they receive conversation that iss a list of dictionary
        values.

        The self._singleton_inputs list assigned during initialization is used to find and extract
        singleton keywords, and self._allow_conversation_input is used to determine if a conversation
        is a valid input.

        If both conversations and singletons are allowed, the function will raise an exception if both
        are inputted.

        This function must be overridden by child classes IF they need to both a conversation and
        other inputs to be passed in.

        :keyword kwargs: The inputs to convert.
        :type kwargs: Dict
        :return: A list of arbitrary values that are valid inputs for this evaluator's do_eval function.
        :rtype: List
        """
        # Collect inputs
        tool_calls = kwargs.get("tool_calls")
        tool_definitions = kwargs.get("tool_definitions", [])  # Default to empty list
        query = kwargs.get("query")
        response = kwargs.get("response")

        # Extract tool calls from response if not provided directly
        if response:
            parsed_tool_calls = self._parse_tools_from_response(response)
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls

        if not tool_calls:
            return {"error_message": self._NO_TOOL_CALLS_MESSAGE}

        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []

        try:
            needed_tool_definitions = _extract_needed_tool_definitions(
                            tool_calls, tool_definitions, ExtendedErrorTarget.TOOL_SELECTION_EVALUATOR)
        except EvaluationException:
            # Check if this is because no tool definitions were provided at all
            if len(tool_definitions) == 0:
                return {"error_message": self._NO_TOOL_DEFINITIONS_MESSAGE}
            else:
                return {"error_message": self._TOOL_DEFINITIONS_MISSING_MESSAGE}

        if len(needed_tool_definitions) == 0:
            return {"error_message": self._NO_TOOL_DEFINITIONS_MESSAGE}

        # Extract only tool names from tool calls, removing parameters and results
        tool_names = _extract_tool_names_from_calls(tool_calls)

        return {
            "query": query,
            "tool_calls": tool_names,  # Only tool names, no parameters
            "tool_definitions": needed_tool_definitions,
        }

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Do Tool Selection evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: A dictionary containing the result of the evaluation.
        :rtype: Dict[str, Union[str, float]]
        """
        if "query" not in eval_input:
            raise EvaluationException(
                message=(
                    "Query is a required input to the Tool Selection evaluator."
                ),
                internal_message=(
                    "Query is a required inputto the Tool Selection evaluator."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.TOOL_SELECTION_EVALUATOR,
            )

        # Format conversation history for cleaner evaluation
        else:
            eval_input["query"] = reformat_conversation_history(
                eval_input["query"], logger, include_system_messages=True, include_tool_calls=True
            )

        # Call the LLM to evaluate
        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", {})

        if isinstance(llm_output, dict):
            score = llm_output.get("score", None)
            if score not in [0, 1]:
                raise EvaluationException(
                    message=f"Invalid score value: {score}. Expected 0 or 1.",
                    internal_message="Invalid score value.",
                    category=ErrorCategory.FAILED_EXECUTION,
                    blame=ErrorBlame.SYSTEM_ERROR,
                )

            # Format the output
            explanation = llm_output.get("explanation", "")
            score = int(score)  # Keep as int since it's binary (0 or 1)
            score_result = "pass" if score == 1 else "fail"

            # Add tool selection accuracy post-processing
            details = llm_output.get("details", {})
            if details:
                tool_selection_accuracy = self._calculate_tool_selection_accuracy(details)
                details["tool_selection_accuracy"] = tool_selection_accuracy

            response_dict = {
                self._result_key: score,
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": explanation,
                f"{self._result_key}_details": details,
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
            return response_dict

        else:
            raise EvaluationException(
                message="Tool selection evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TOOL_SELECTION_EVALUATOR,
            )

    async def _real_call(self, **kwargs):
        """Perform the asynchronous call for the real end-to-end evaluation logic.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Convert inputs into list of evaluable inputs.
        eval_input = self._convert_kwargs_to_eval_input(**kwargs)
        if isinstance(eval_input, dict) and eval_input.get("error_message"):
            return self._not_applicable_result(eval_input.get("error_message"), 1)

        result = await self._do_eval(eval_input)

        return result

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable.

        :param error_message: The error message explaining why evaluation is not applicable.
        :type error_message: str
        :param threshold: The threshold value for the evaluator.
        :type threshold: Union[int, float]
        :return: A dictionary containing the result of the evaluation.
        :rtype: Dict[str, Union[str, float, Dict]]
        """
        # If no tool calls were made or tool call type is not supported, return not applicable result
        return {
            self._result_key: self._NOT_APPLICABLE_RESULT,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": error_message,
            f"{self._result_key}_details": {},
        }

    def _calculate_tool_selection_accuracy(self, details):
        """Calculate tool selection accuracy from the evaluation details.

        :param details: The details dictionary from the LLM evaluation output
        :type details: Dict
        :return: Tool selection accuracy as a percentage
        :rtype: float
        """
        correct_tool_selections = details.get("correct_tool_selections", 0)
        wrong_tool_selections = details.get("wrong_tool_selections", 0)
        total_tools_called = correct_tool_selections + wrong_tool_selections

        if total_tools_called > 0:
            accuracy = (correct_tool_selections / total_tools_called) * 100
            return round(accuracy, 2)
