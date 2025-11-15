# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from collections import Counter
import json
import copy
from typing import Dict, List, Union, Any, Tuple
from typing_extensions import overload, override

from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._exceptions import (
    ErrorCategory,
    ErrorTarget,
    EvaluationException,
)


# Extend ErrorTarget enum if needed
def _create_extended_error_target(ErrorTarget):
    """Create an extended ErrorTarget enum that includes TOOL_INPUT_ACCURACY_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members['TOOL_INPUT_ACCURACY_EVALUATOR'] = 'ToolInputAccuracyEvaluator'

    ExtendedErrorTarget = Enum('ExtendedErrorTarget', existing_members)
    return ExtendedErrorTarget


ErrorTarget = _create_extended_error_target(ErrorTarget)


class TaskNavigationEfficiencyMatchingMode(str, Enum):
    """
    Enumeration of task navigation efficiency matching mode.

    This enum allows you to specify which single matching technique should be used when evaluating
    the efficiency of an agent's tool calls sequence against a expected actions path.
    """

    EXACT_MATCH = "exact_match"
    """
    Binary metric indicating whether the agent's tool calls exactly match the expected actions.

    Returns True only if the agent's tool calls sequence is identical to the expected sequence
    in both order and content (no extra steps, no missing steps, correct order).
    """

    IN_ORDER_MATCH = "in_order_match"
    """
    Binary metric allowing extra steps but requiring correct order of required tool calls.

    Returns True if all expected actions steps appear in the agent's sequence in the correct
    order, even if there are additional steps interspersed.
    """

    ANY_ORDER_MATCH = "any_order_match"
    """
    Binary metric allowing both extra steps and different ordering.

    Returns True if all expected actions steps appear in the agent's sequence with sufficient
    frequency, regardless of order. Most lenient matching criterion.
    """


@experimental
class TaskNavigationEfficiencyEvaluator(EvaluatorBase):
    """
    Evaluates whether an agent's sequence of actions is efficient and follows optimal decision-making patterns.

    The Task Navigation Efficiency Evaluator returns binary matching results between the agent's tool usage trajectory
    and the expected actions expected steps.
    It has three matching techniques: exact match, in-order match (allows extra steps),
    and any-order match (allows extra steps and ignores order).
    It also returns precision, recall, and F1 scores in properties bag.

    :param matching_mode: The matching mode to use. Default is "exact_match".
    :type matching_mode: enum[str, TaskNavigationEfficiencyMatchingMode]

    .. admonition:: Example:

        .. code-block:: python

            from azure.ai.evaluation._task_navigation_efficiency import TaskNavigationEfficiencyEvaluator

            task_navigation_efficiency_eval = TaskNavigationEfficiencyEvaluator(
                matching_mode=TaskNavigationEfficiencyMatchingMode.EXACT_MATCH
            )

            # Example 1: Using simple tool names list
            result = path_efficiency_eval(
                actions=[
                    {"role": "assistant", "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "identify_tools_to_call",
                         "arguments": {}}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "tool_call", "tool_call_id": "call_2", "name": "call_tool_A", "arguments": {}}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "tool_call", "tool_call_id": "call_3", "name": "call_tool_B", "arguments": {}}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "tool_call", "tool_call_id": "call_4", "name": "response_synthesis", "arguments": {}}
                    ]}
                ],
                expected_actions=["identify_tools_to_call", ""call_tool_A", "call_tool_B", "response_synthesis"]
            )

            # Example 2: Using tool names with parameters (exact parameter matching required)
            result = path_efficiency_eval(
                actions=[
                    {"role": "assistant", "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "search",
                         "arguments": {"query": "weather", "location": "NYC"}}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "tool_call", "tool_call_id": "call_2", "name": "format_result",
                         "arguments": {"format": "json"}}
                    ]}
                ],
                expected_actions=(
                    ["search", "format_result"],
                    {
                        "search": {"query": "weather", "location": "NYC"},
                        "format_result": {"format": "json"}
                    }
                )
            )
    """

    id = "azureai://built-in/evaluators/task_navigation_efficiency"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    matching_mode: TaskNavigationEfficiencyMatchingMode
    """The matching mode to use."""

    @override
    def __init__(
        self,
        *,
        matching_mode: Union[
            str, TaskNavigationEfficiencyMatchingMode
        ] = TaskNavigationEfficiencyMatchingMode.EXACT_MATCH,
    ):
        """Initialize a TaskNavigationEfficiencyEvaluator instance.

        :param matching_mode: The matching mode to use. Default is "exact_match".
        :type matching_mode: enum[str, TaskNavigationEfficiencyMatchingMode]
        """
        # Type checking for metric parameter
        if isinstance(matching_mode, str):
            try:
                self.matching_mode = TaskNavigationEfficiencyMatchingMode(matching_mode)
            except ValueError:
                raise ValueError(
                    f"matching_mode must be one of {[m.value for m in TaskNavigationEfficiencyMatchingMode]}, "
                    f"got '{matching_mode}'"
                )
        elif isinstance(matching_mode, TaskNavigationEfficiencyMatchingMode):
            self.matching_mode = matching_mode
        else:
            raise EvaluationException(
                f"matching_mode must be a string with one of {[m.value for m in TaskNavigationEfficiencyMatchingMode]}"
                f" or TaskNavigationEfficiencyMatchingMode enum, got {type(matching_mode)}",
                internal_message=str(matching_mode),
                target=ErrorTarget.TASK_NAVIGATION_EFFICIENCY_EVALUATOR,
                category=ErrorCategory.INVALID_VALUE,
            )

        super().__init__()

    def _prepare_steps_for_comparison(
        self,
        agent_tool_pairs: List[Tuple[str, Dict[str, Any]]],
        expected_actions: List[str],
        expected_actions_params: Dict[str, Dict[str, Any]],
        use_parameter_matching: bool,
    ) -> Tuple[
        List[Union[str, Tuple[str, Tuple]]],
        List[Union[str, Tuple[str, Tuple]]],
    ]:
        """Prepare agent and expected actions steps for comparison based on parameter matching mode."""
        agent_steps: List[Union[str, Tuple[str, Tuple]]] = []
        expected_actions_steps: List[Union[str, Tuple[str, Tuple]]] = []
        if use_parameter_matching:
            # When parameter matching is enabled, we need to match both tool name and parameters
            agent_steps = [(pair[0], tuple(sorted(pair[1].items()))) for pair in agent_tool_pairs]
            expected_actions_steps = [
                (name, tuple(sorted(expected_actions_params.get(name, {}).items()))) for name in expected_actions
            ]
        else:
            # When parameter matching is disabled, only compare tool names
            agent_steps = [name for name, _ in agent_tool_pairs]
            expected_actions_steps = [step for step in expected_actions]

        return agent_steps, expected_actions_steps

    def _calculate_precision_recall_f1_scores(
        self, agent_steps: List, expected_actions_steps: List
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 scores."""
        if not agent_steps:
            return {"precision_score": 0.0, "recall_score": 0.0, "f1_score": 0.0}

        # Count occurrences of each step in both lists to handle duplicates
        agent_steps_counts = Counter(agent_steps)
        expected_actions_counts = Counter(expected_actions_steps)

        # Calculate true positives by taking the minimum count for each common element
        # For each step, count the intersection (min count) of agent and expected actions steps
        true_positives = sum(
            min(agent_steps_counts[step], expected_actions_counts[step])
            for step in agent_steps_counts
            if step in expected_actions_counts
        )

        # Calculate false positives (agent steps not in expected actions or excess occurrences)
        # For each step, count the excess occurrences of agent steps not in (minus) expected actions
        # or zero (agent steps minus agent steps) if agent steps is less than expected actions
        false_positives = sum(
            agent_steps_counts[step] - min(agent_steps_counts[step], expected_actions_counts.get(step, 0))
            for step in agent_steps_counts
        )

        # Calculate false negatives (expected actions steps not in agent or missing occurrences)
        # For each step, count the excess occurrences of expected actions steps not in (minus) agent steps
        # or zero (expected actions steps minus expected actions steps) if expected actions steps is less than
        # agent steps
        false_negatives = sum(
            expected_actions_counts[step] - min(expected_actions_counts[step], agent_steps_counts.get(step, 0))
            for step in expected_actions_counts
        )

        # Calculate precision, recall, F1
        precision = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        )
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision_score": precision,
            "recall_score": recall,
            "f1_score": f1_score,
        }

    def _calculate_exact_match(self, agent_steps: List, expected_actions_steps: List) -> bool:
        """Check if agent steps exactly match expected actions (order and content)."""
        return agent_steps == expected_actions_steps

    def _calculate_in_order_match(self, agent_steps: List, expected_actions_steps: List) -> bool:
        """Check if all expected actions steps appear in agent steps in correct order (extra steps allowed)."""
        if not expected_actions_steps:
            return True

        gt_index = 0
        for step in agent_steps:
            if gt_index < len(expected_actions_steps) and step == expected_actions_steps[gt_index]:
                gt_index += 1

        return gt_index == len(expected_actions_steps)

    def _calculate_any_order_match(self, agent_steps: List, expected_actions_steps: List) -> bool:
        """Check if all expected actions steps appear in agent steps with sufficient frequency.

        any order, extra steps allowed.
        """
        # Count occurrences of each step in both lists to handle duplicates
        agent_counts = Counter(agent_steps)
        expected_actions_counts = Counter(expected_actions_steps)

        # Check if agent has at least as many occurrences of each expected actions step
        return all(agent_counts[step] >= expected_actions_counts[step] for step in expected_actions_counts)

    _TASK_NAVIGATION_EFFICIENCY_MATCHING_MODE_TO_FUNCTIONS = {
        TaskNavigationEfficiencyMatchingMode.EXACT_MATCH: _calculate_exact_match,
        TaskNavigationEfficiencyMatchingMode.IN_ORDER_MATCH: _calculate_in_order_match,
        TaskNavigationEfficiencyMatchingMode.ANY_ORDER_MATCH: _calculate_any_order_match,
    }

    def _parse_tools_from_actions(self, actions):
        """Parse the actions to extract tool calls and results.

        :param actions: The actions to parse.
        :type actions: Union[str, List[dict]]
        :return: List of tool calls extracted from the actions.
        :rtype: List[dict]
        """
        tool_calls = []
        tool_results_map = {}

        # Work on a deep copy to avoid modifying the original object
        actions_copy = copy.deepcopy(actions)

        if isinstance(actions_copy, list):
            for message in actions_copy:
                # Extract tool calls from assistant messages
                if message.get("role") == "assistant" and isinstance(message.get("content"), list):
                    for content_item in message.get("content"):
                        if isinstance(content_item, dict) and content_item.get("type") == "tool_call":
                            tool_calls.append(content_item)

                # Extract tool results from tool messages
                elif message.get("role") == "tool" and message.get("tool_call_id"):
                    tool_call_id = message.get("tool_call_id")
                    if isinstance(message.get("content"), list) and len(message.get("content")) > 0:
                        result_content = message.get("content")[0]
                        if isinstance(result_content, dict) and result_content.get("type") == "tool_result":
                            tool_results_map[tool_call_id] = result_content

        # Attach results to their corresponding calls
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("tool_call_id")
            if tool_call_id in tool_results_map:
                tool_call["tool_result"] = tool_results_map[tool_call_id]["tool_result"]

        return tool_calls

    def _extract_tool_names_and_params_from_actions(self, actions) -> List[Tuple[str, Dict[str, str]]]:
        """Extract tool names and parameters from the actions.

        :param actions: The actions to parse.
        :type actions: Union[str, List[dict]]
        :return: List of tuples containing (tool_name, parameters_dict) extracted from the actions.
        :rtype: List[Tuple[str, Dict[str, str]]]
        """
        tool_calls = self._parse_tools_from_actions(actions)
        tool_name_param_pairs = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise EvaluationException(
                    "Tool call must be a dictionary.",
                    internal_message=str(tool_call),
                    target=ErrorTarget.EVALUATE,
                    category=ErrorCategory.UNKNOWN,
                )
            if tool_call.get("type") != "tool_call":
                raise EvaluationException(
                    "Tool call must have 'type' set to 'tool_call'.",
                    internal_message=str(tool_call),
                    target=ErrorTarget.EVALUATE,
                    category=ErrorCategory.INVALID_VALUE,
                )

            if "name" not in tool_call:
                raise EvaluationException(
                    "Tool call missing 'name' field.",
                    internal_message=str(tool_call),
                    target=ErrorTarget.EVALUATE,
                    category=ErrorCategory.MISSING_FIELD,
                )

            tool_name = str(tool_call["name"]).strip()

            # Extract parameters/arguments
            parameters = {}
            if "arguments" in tool_call:
                args = tool_call["arguments"]
                if isinstance(args, dict):
                    # Convert all values to strings for consistent comparison
                    parameters = {str(k): str(v) for k, v in args.items()}
                elif isinstance(args, str):
                    # If arguments is a string, try to parse it as JSON
                    try:
                        parsed_args = json.loads(args)
                        if isinstance(parsed_args, dict):
                            parameters = {str(k): str(v) for k, v in parsed_args.items()}
                    except json.JSONDecodeError:
                        raise EvaluationException(
                            "Failed to parse tool call arguments as JSON.",
                            internal_message=str(tool_call),
                            target=ErrorTarget.EVALUATE,
                            category=ErrorCategory.INVALID_VALUE,
                        )

            tool_name_param_pairs.append((tool_name, parameters))

        return tool_name_param_pairs

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str, Dict[str, float]]]:
        """Produce a path efficiency evaluation result.

        :param eval_input: The input to the evaluation function. Must contain "actions" and "expected_actions".
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict[str, Union[float, str, Dict[str, float]]]
        """
        actions = eval_input["actions"]
        expected_actions = eval_input["expected_actions"]

        # Value and type checking for expected actions steps
        if not expected_actions:
            raise ValueError("expected_actions cannot be empty")

        # Check if expected_actions is a tuple (tool names + parameters) or list (tool names only)
        use_parameter_matching = False
        expected_actions_names = []
        expected_actions_params_dict: Dict[str, Dict[str, Any]] = {}

        if isinstance(expected_actions, list) and all(isinstance(step, str) for step in expected_actions):
            # List format: just tool names
            expected_actions_names = [step.strip() for step in expected_actions]
            use_parameter_matching = False
        elif (
            isinstance(expected_actions, tuple) or isinstance(expected_actions, list)
        ) and len(expected_actions) == 2:
            # Tuple format: (tool_names, parameters_dict)
            tool_names_list, params_dict = expected_actions

            if not isinstance(tool_names_list, list) or not all(isinstance(name, str) for name in tool_names_list):
                raise TypeError("expected_actions tuple first element must be a list of strings (tool names)")

            if not isinstance(params_dict, dict):
                raise TypeError(
                    "expected_actions tuple second element must be a dictionary mapping tool names to parameters"
                )

            # Validate that all values in params_dict are dictionaries with string keys and values
            for tool_name, params in params_dict.items():
                if not isinstance(tool_name, str):
                    raise TypeError("expected_actions parameters dictionary keys must be strings (tool names)")
                if not isinstance(params, dict):
                    raise TypeError(f"expected_actions parameters for tool '{tool_name}' must be a dictionary")
                for k, v in params.items():
                    if not isinstance(k, str):
                        raise TypeError(f"expected_actions parameters for tool '{tool_name}' must have string keys")
                    try:
                        json.dumps(v)
                    except (TypeError, ValueError):
                        raise TypeError(
                            f"expected_actions parameters for tool '{tool_name}' must have JSON-serializable values "
                            f"(got type {type(v)} for key '{k}')"
                        )

            expected_actions_names = [name.strip() for name in tool_names_list]
            expected_actions_params_dict = params_dict
            use_parameter_matching = True
        else:
            raise TypeError(
                "expected_actions must be a list of strings or a tuple of (list[str], dict[str, dict[str, str]])"
            )

        # Extract tool information from the actions
        agent_tool_pairs = self._extract_tool_names_and_params_from_actions(actions)

        # Prepare steps for comparison
        agent_steps, expected_actions_steps = self._prepare_steps_for_comparison(
            agent_tool_pairs,
            expected_actions_names,
            expected_actions_params_dict,
            use_parameter_matching,
        )

        # Calculate precision, recall, and F1 scores
        additional_properties_metrics = self._calculate_precision_recall_f1_scores(agent_steps, expected_actions_steps)

        # Convert metrics to floats, using nan for None or non-convertible values
        for metric, score in additional_properties_metrics.items():
            additional_properties_metrics[metric] = float(score) if score is not None else float("nan")

        if self.matching_mode in self._TASK_NAVIGATION_EFFICIENCY_MATCHING_MODE_TO_FUNCTIONS:
            # Calculate binary match metrics
            match_result = self._TASK_NAVIGATION_EFFICIENCY_MATCHING_MODE_TO_FUNCTIONS[self.matching_mode](
                self, agent_steps, expected_actions_steps
            )

            return {
                "task_navigation_efficiency_label": match_result,
                "task_navigation_efficiency_result": EVALUATION_PASS_FAIL_MAPPING[match_result],
                "task_navigation_efficiency_details": additional_properties_metrics,
            }
        else:
            raise EvaluationException(
                f"Unsupported matching_mode '{self.matching_mode}'",
                internal_message=str(self.matching_mode),
                target=ErrorTarget.TASK_NAVIGATION_EFFICIENCY_EVALUATOR,
                category=ErrorCategory.INVALID_VALUE,
            )

    @overload
    def __call__(  # type: ignore
        self, *, actions: Union[str, List[Dict[str, Any]]], expected_actions: List[str]
    ) -> Dict[str, Union[float, str, Dict[str, float]]]:
        """
        Evaluate the task navigation efficiency of an agent's action sequence.

        :keyword actions: The agent's actions containing tool calls.
        :paramtype actions: Union[str, List[Dict[str, Any]]]
        :keyword expected_actions: List of expected tool/action steps.
        :paramtype expected_actions: List[str]
        :return: The task navigation efficiency scores and results.
        :rtype: Dict[str, Union[float, str, Dict[str, float]]]
        """

    @overload
    def __call__(  # type: ignore
        self,
        *,
        actions: Union[str, List[Dict[str, Any]]],
        expected_actions: Tuple[List[str], Dict[str, Dict[str, str]]],
    ) -> Dict[str, Union[float, str, Dict[str, float]]]:
        """
        Evaluate the task navigation efficiency of an agent's action sequence with tool parameters.

        :keyword actions: The agent's actions containing tool calls.
        :paramtype actions: Union[str, List[Dict[str, Any]]]
        :keyword expected_actions: Tuple of (tool names list, parameters dict) where parameters must match exactly.
        :paramtype expected_actions: Tuple[List[str], Dict[str, Dict[str, str]]]
        :return: The task navigation efficiency scores and results.
        :rtype: Dict[str, Union[float, str, Dict[str, float]]]
        """

    @override
    def __call__(
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate task navigation efficiency.

        :keyword actions: The agent's actions containing tool calls.
        :paramtype actions: Union[str, List[Dict[str, Any]]]
        :keyword expected_actions: List of expected tool/action steps or tuple of (tool names, parameters dict).
        :paramtype expected_actions: Union[List[str], Tuple[List[str], Dict[str, Dict[str, str]]]]
        :return: The task navigation efficiency scores and results.
        :rtype: Dict[str, Union[float, str, Dict[str, float]]]
        """
        return super().__call__(*args, **kwargs)
