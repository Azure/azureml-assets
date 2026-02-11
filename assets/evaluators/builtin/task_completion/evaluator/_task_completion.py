# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import reformat_conversation_history, reformat_agent_response
from azure.ai.evaluation._common.utils import reformat_tool_definitions
from azure.ai.evaluation._common._experimental import experimental
from enum import Enum

logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TASK_COMPLETION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members['TASK_COMPLETION_EVALUATOR'] = 'TaskCompletionEvaluator'

    ExtendedErrorTarget = Enum('ExtendedErrorTarget', existing_members)
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
                if (isinstance(last_content, dict) and
                        last_content.get("type") in ("function_call", "mcp_approval_request")):
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
class TaskCompletionEvaluator(PromptyEvaluatorBase[Union[str, int]]):
    """The Task Completion evaluator determines whether an AI agent successfully completed the requested task.

    This evaluator assesses task completion based on:
        - Final outcome and deliverable of the task
        - Completeness of task requirements

    This evaluator focuses solely on task completion and success, not on task adherence or intent understanding.

    Scoring is binary:
    - 1 (Pass): Task fully completed with usable deliverable that meets all user requirements
    - 0 (Fail): Task incomplete, partially completed, or deliverable does not meet requirements

    The evaluation includes task requirement analysis, outcome assessment, and completion gap identification.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a TaskCompletionEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call TaskCompletionEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "task_completion.prompty"
    _RESULT_KEY = "task_completion"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    id = "azureai://built-in/evaluators/task_completion"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        """Initialize the TaskCompletionEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop('threshold', 1)
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
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int]]:
        """Evaluate task completion for a given query, response, and optionally tool definitions.

        The query and response can be either a string or a list of messages.

        Example with string inputs and no tools:
            evaluator = TaskCompletionEvaluator(model_config)
            query = "Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine."
            response = "**Day 1:** Morning: Louvre Museum, Lunch: Le Comptoir du Relais..."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = TaskCompletionEvaluator(model_config)
            query = [{'role': 'system', 'content': 'You are a helpful travel planning assistant.'},
                     {'createdAt': 1700000060, 'role': 'user',
                     'content': [{'type': 'text', 'text': 'Plan a 3-day Paris itinerary with cultural
                     landmarks and cuisine'}]}]
            response = [{'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant',
                        'content': [{'type': 'text', 'text': '*Day 1:* Morning: Visit Louvre Museum (9 AM-12 PM)'}]}]
            tool_definitions = [{'name': 'get_attractions', 'description': 'Get tourist attractions for a city.',
                                'parameters': {'type': 'object', 'properties':
                                  {'city': {'type': 'string', 'description': 'City name.'}}}}]

            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the task completion evaluation results.
        :rtype: Dict[str, Union[str, int]]
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
            f"{self._result_key}_details": {},
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
        }

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:  # type: ignore[override]
        """Do Task Completion evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # we override the _do_eval method as we want the output to be a dictionary,
        # which is a different schema than _base_prompty_eval.py
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Task Completion evaluator.",
                internal_message="Both query and response must be provided as input to the Task Completion evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
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
        eval_input["query"] = reformat_conversation_history(eval_input["query"], logger, include_system_messages=True)
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)
        if "tool_definitions" in eval_input and eval_input["tool_definitions"]:
            eval_input["tool_definitions"] = reformat_tool_definitions(eval_input["tool_definitions"], logger)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            success_value = llm_output.get("success", False)
            if isinstance(success_value, str):
                success = 1 if success_value.lower() == "true" else 0
            else:
                success = 1 if success_value else 0
            success_result = "pass" if success == 1 else "fail"
            reason = llm_output.get("explanation", "")
            return {
                self._result_key: success,
                f"{self._result_key}_result": success_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_details": llm_output.get("details", {}),
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
        if logger:
            logger.warning("LLM output is not a dictionary, returning 0 for the success.")
        return {self._result_key: 0}
