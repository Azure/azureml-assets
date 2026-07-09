# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from enum import Enum
from typing import Dict, List, Union, TypeVar, cast
from typing_extensions import override
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import (
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
    EvaluationException,
)
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._common.utils import (
    reformat_agent_response,
    reformat_conversation_history,
    _is_intermediate_response,
    _preprocess_messages,
)
from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ToolDefinitionsValidator,
)


logger = logging.getLogger(__name__)

T_EvalValue = TypeVar("T_EvalValue")


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TOOL_INPUT_ACCURACY_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TOOL_INPUT_ACCURACY_EVALUATOR"] = "ToolInputAccuracyEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


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
                    # This is a regular function tool from converter or built-in tool from agent v2
                    tool_definition_exists = any(tool.get("name") == tool_name for tool in needed_tool_definitions)
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


@experimental
class ToolInputAccuracyEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Tool Input Accuracy evaluator performs an evaluations of parameters passed to tool calls.

       The evaluation criteria are as follows:
        - Parameter grounding: All parameters must be derived from conversation history/query
        - Type compliance: All parameters must match exact types specified in tool definitions
        - Format compliance: All parameters must follow exact format and structure requirements
        - Completeness: All required parameters must be provided
        - No unexpected parameters: Only defined parameters are allowed

    The evaluator uses strict binary evaluation:
        - PASS: Only when ALL criteria are satisfied perfectly for ALL parameters
        - FAIL: When ANY criterion fails for ANY parameter

    This evaluation focuses on ensuring tool call parameters are completely correct without any tolerance
    for partial correctness.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START tool_input_accuracy_evaluator]
            :end-before: [END tool_input_accuracy_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolInputAccuracyEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START tool_input_accuracy_evaluator]
            :end-before: [END tool_input_accuracy_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call ToolInputAccuracyEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE = "tool_input_accuracy.prompty"
    _RESULT_KEY = "tool_input_accuracy"

    _validator: ValidatorInterface

    _NO_TOOL_CALLS_MESSAGE = "No tool calls found in response or provided tool_calls."
    _NO_TOOL_DEFINITIONS_MESSAGE = "Tool definitions must be provided."
    _TOOL_DEFINITIONS_MISSING_MESSAGE = "Tool definitions for all tool calls must be provided."

    def __init__(
        self,
        model_config,
        *,
        credential=None,
        **kwargs,
    ):
        """Initialize the Tool Input Accuracy evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_INPUT_ACCURACY_EVALUATOR, optional_tool_definitions=False,
            check_for_unsupported_tools=False,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=1,
            **kwargs,
        )

    def _convert_kwargs_to_eval_input(self, **kwargs):
        """Convert kwargs to evaluation input format.

        :keyword kwargs: The inputs to convert.
        :type kwargs: Dict
        :return: The formatted evaluation input.
        :rtype: Dict
        """
        # Collect inputs
        tool_definitions = kwargs.get("tool_definitions", [])  # Default to empty list
        query = kwargs.get("query")
        response = kwargs.get("response")

        if not response:
            return {"error_message": "Response parameter is required to extract tool calls."}

        # Try to parse tool calls from response
        tool_calls = self._parse_tools_from_response(response)

        if not tool_calls:
            # If no tool calls found and response is string, use response string as tool calls as is
            if isinstance(response, str):
                tool_calls = response
            else:
                return {"error_message": self._NO_TOOL_CALLS_MESSAGE}

        # Normalize tool_calls and tool_definitions (skip for strings)
        if not isinstance(tool_calls, list) and not isinstance(tool_calls, str):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list) and not isinstance(tool_definitions, str):
            tool_definitions = [tool_definitions] if tool_definitions else []

        # Cross-validation (skip when either is string)
        if isinstance(tool_calls, str) or isinstance(tool_definitions, str):
            needed_tool_definitions = tool_definitions
        else:
            try:
                # Type cast to satisfy static type checker
                tool_calls_typed = cast(List[Dict], tool_calls)
                needed_tool_definitions = _extract_needed_tool_definitions(
                    tool_calls_typed, tool_definitions, ExtendedErrorTarget.TOOL_INPUT_ACCURACY_EVALUATOR
                )
            except EvaluationException:
                # Check if this is because no tool definitions were provided at all
                if len(tool_definitions) == 0:
                    return {"error_message": self._NO_TOOL_DEFINITIONS_MESSAGE}
                else:
                    return {"error_message": self._TOOL_DEFINITIONS_MISSING_MESSAGE}

        if not needed_tool_definitions:
            return {"error_message": self._NO_TOOL_DEFINITIONS_MESSAGE}

        # Reformat response for LLM (skip for strings - already a string)
        if isinstance(tool_calls, str):
            agent_response_with_tools = tool_calls
        else:
            agent_response_with_tools = reformat_agent_response(response, include_tool_messages=True)

        return {
            "query": query,
            "tool_calls": agent_response_with_tools,
            "tool_definitions": needed_tool_definitions,
        }

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Do Tool Input Accuracy evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: A dictionary containing the result of the evaluation.
        :rtype: Dict[str, Union[str, float]]
        """
        if eval_input.get("query") is None:
            raise EvaluationException(
                message=(
                    "Query is a required input to "
                    "the Tool Input Accuracy evaluator."
                ),
                internal_message=(
                    "Query is a required input "
                    "to the Tool Input Accuracy evaluator."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.TOOL_INPUT_ACCURACY_EVALUATOR,
            )

        # Format conversation history for cleaner evaluation
        eval_input["query"] = reformat_conversation_history(
            eval_input["query"], logger, include_system_messages=True, include_tool_calls=True
        )

        # Call the LLM to evaluate
        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            # Handle skipped status from LLM
            llm_status = llm_output.get("status", "completed")
            if llm_status == "skipped":
                reason = llm_output.get("reason", "")
                return self._return_not_applicable_result(reason, self._threshold)

            score = llm_output.get("score", None)
            if score not in [0, 1]:
                raise EvaluationException(
                    message=f"Invalid score value: {score}. Expected 0 or 1.",
                    internal_message="Invalid score value.",
                    category=ErrorCategory.FAILED_EXECUTION,
                    blame=ErrorBlame.SYSTEM_ERROR,
                )

            # Add parameter extraction accuracy post-processing
            llm_properties = llm_output.get("properties", {}) or {}
            if llm_properties:
                parameter_extraction_accuracy = self._calculate_parameter_extraction_accuracy(llm_properties)
                llm_properties["parameter_extraction_accuracy"] = parameter_extraction_accuracy

            # Format the output
            reason = llm_output.get("reason", "")
            score = float(score)
            score_result = "pass" if score == 1 else "fail"
            token_metadata = self._get_token_metadata(prompty_output_dict)
            llm_properties.update(token_metadata)
            response_dict = {
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
            response_dict.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
            return response_dict

        else:
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TOOL_INPUT_ACCURACY_EVALUATOR,
            )

    async def _real_call(self, **kwargs):
        """Perform the asynchronous call for the real end-to-end evaluation logic.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
        self._validator.validate_eval_input(kwargs)

        response = kwargs.get("response")
        if _is_intermediate_response(response):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if "response" in kwargs:
            kwargs["response"] = _preprocess_messages(kwargs["response"])
        if "query" in kwargs:
            kwargs["query"] = _preprocess_messages(kwargs["query"])
        # Convert inputs into list of evaluable inputs.
        eval_input = self._convert_kwargs_to_eval_input(**kwargs)
        if isinstance(eval_input, dict) and eval_input.get("error_message"):
            # If there is an error message, return not applicable result
            error_message = eval_input.get("error_message", "Unknown error")
            return self._return_not_applicable_result(error_message, self._threshold)
        # Do the evaluation
        result = await self._do_eval(eval_input)
        # Return the result
        return result

    def _calculate_parameter_extraction_accuracy(self, details):
        """Calculate parameter extraction accuracy from the evaluation details.

        :param details: The details dictionary from the LLM evaluation output
        :type details: Dict
        :return: Parameter extraction accuracy as a percentage
        :rtype: float
        """
        total_parameters = details.get("total_parameters_passed", 0)
        correct_parameters = details.get("correct_parameters_passed", 0)

        if total_parameters == 0:
            return 100.0  # If no parameters were passed, accuracy is 100%

        accuracy = (correct_parameters / total_parameters) * 100
        return round(accuracy, 2)

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
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate parameter correctness of tool calls.

        :keyword query: Query or Chat history up to the message that has the tool call being evaluated.
        :paramtype query: Union[str, List[dict]]
        :keyword tool_definitions: List of tool definitions whose calls are being evaluated.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :keyword response: Response containing tool calls to be evaluated.
        :paramtype response: Union[str, List[dict]]
        :return: The tool input accuracy evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """
        return super().__call__(*args, **kwargs)
