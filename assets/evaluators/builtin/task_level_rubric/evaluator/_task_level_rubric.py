# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Row-specific rubric evaluation with a temporary, testing-only judge prompt."""

import hashlib
import json
import logging
import math
import os
from copy import deepcopy
from fractions import Fraction
from typing import Any, TypedDict, TypeVar

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import (
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
    EvaluationException,
)

logger = logging.getLogger(__name__)

_MAX_DIMENSIONS = 10
_MESSAGE_ROLES = {"system", "developer", "user", "assistant", "tool"}
_InputT = TypeVar("_InputT")
_USAGE_FIELDS = {
    "prompt_tokens": "input_token_count",
    "completion_tokens": "output_token_count",
    "total_tokens": "total_token_count",
    "cached_tokens": "cached_tokens",
}


class _RubricDimension(TypedDict):
    id: str
    description: str
    weight: int


class _JudgeInput(TypedDict):
    messages: list[dict[str, Any]]
    rubric_dimensions: list[_RubricDimension]
    tool_definitions: dict[str, Any] | list[dict[str, Any]]
    test_case_description: str


def _input_error(message: str, *, missing: bool = False) -> EvaluationException:
    return EvaluationException(
        message=message,
        blame=ErrorBlame.USER_ERROR,
        category=ErrorCategory.MISSING_FIELD
        if missing
        else ErrorCategory.INVALID_VALUE,
        target=ErrorTarget.EVALUATE,
    )


def _judge_error(message: str) -> EvaluationException:
    return EvaluationException(
        message=message,
        blame=ErrorBlame.SYSTEM_ERROR,
        category=ErrorCategory.FAILED_EXECUTION,
        target=ErrorTarget.EVALUATE,
    )


def _copy_task_input(value: _InputT) -> _InputT:
    try:
        return deepcopy(value)
    except RecursionError as exc:
        raise _input_error("Task inputs are too deeply nested to process.") from exc


def _validate_dimensions(value: Any) -> list[_RubricDimension]:
    if value is None:
        raise _input_error(
            "rubric_dimensions is required for every Task.", missing=True
        )
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_DIMENSIONS:
        raise _input_error(
            "rubric_dimensions must be a structured array containing 1-10 dimensions."
        )

    dimensions: list[_RubricDimension] = []
    seen_ids: set[str] = set()
    for index, dimension in enumerate(value):
        if not isinstance(dimension, dict):
            raise _input_error(f"rubric_dimensions[{index}] must be an object.")
        dimension_id = dimension.get("id")
        description = dimension.get("description")
        weight = dimension.get("weight")
        if not isinstance(dimension_id, str) or not dimension_id.strip():
            raise _input_error(
                f"rubric_dimensions[{index}].id must be a nonblank string."
            )
        if dimension_id in seen_ids:
            raise _input_error(
                "rubric_dimensions must contain unique, case-sensitive IDs."
            )
        if not isinstance(description, str) or not description.strip():
            raise _input_error(
                f"rubric_dimensions[{index}].description must be a nonblank string."
            )
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise _input_error(
                f"rubric_dimensions[{index}].weight must be a positive integer."
            )
        seen_ids.add(dimension_id)
        dimensions.append(
            {
                "id": dimension_id,
                "description": description,
                "weight": weight,
            }
        )
    return dimensions


def _validate_messages(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise _input_error(f"{field} must be a nonempty array of messages.")
    for message in value:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message["role"] not in _MESSAGE_ROLES
        ):
            raise _input_error(f"Each {field} message must have a supported role.")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if tool_calls is not None and (
            not isinstance(tool_calls, list)
            or not all(isinstance(call, dict) and call for call in tool_calls)
        ):
            raise _input_error(f"{field} tool_calls must be an array of objects.")

        has_content = False
        if isinstance(content, str):
            has_content = bool(content.strip())
        elif isinstance(content, list):
            for block in content:
                if (
                    not isinstance(block, dict)
                    or not isinstance(block.get("type"), str)
                    or not block["type"]
                ):
                    raise _input_error(
                        f"{field} content blocks must be objects with a nonempty type."
                    )
                if block["type"] == "text":
                    if not isinstance(block.get("text"), str):
                        raise _input_error(
                            f"{field} text blocks must contain string text."
                        )
                    has_content = has_content or bool(block["text"].strip())
                else:
                    has_content = True
        elif content is not None:
            raise _input_error(
                f"{field} content must be text or an array of content blocks."
            )
        if not has_content and not (message["role"] == "assistant" and tool_calls):
            raise _input_error(
                f"Each {field} message must contain content or assistant tool calls."
            )
    return _copy_task_input(value)


def _trace_messages(eval_input: dict[str, Any]) -> list[dict[str, Any]]:
    messages = eval_input.get("messages")
    query = eval_input.get("query")
    response = eval_input.get("response")
    if messages is not None:
        if query is not None or response is not None:
            raise _input_error(
                "Provide messages or query and response, not both input forms."
            )
        trace = _validate_messages(messages, "messages")
        assistant_messages = trace
    else:
        if query is None or response is None:
            raise _input_error(
                "Provide messages or both query and response.", missing=True
            )
        query_messages = (
            [{"role": "user", "content": query}] if isinstance(query, str) else query
        )
        response_messages = (
            [{"role": "assistant", "content": response}]
            if isinstance(response, str)
            else response
        )
        trace = _validate_messages(query_messages, "query")
        assistant_messages = _validate_messages(response_messages, "response")
        trace.extend(assistant_messages)
    if not any(message["role"] == "assistant" for message in assistant_messages):
        raise _input_error("The Task trace must include an assistant response.")
    return trace


def _prepare_judge_input(eval_input: dict[str, Any]) -> _JudgeInput:
    # Dataset column alignment can fill omitted fields with NaN.
    eval_input = {
        field: value
        for field, value in eval_input.items()
        if not (
            field
            in {
                "messages",
                "query",
                "response",
                "tool_definitions",
                "test_case_description",
            }
            and isinstance(value, float)
            and math.isnan(value)
        )
    }
    dimensions = _validate_dimensions(eval_input.get("rubric_dimensions"))
    messages = _trace_messages(eval_input)
    tool_definitions = eval_input.get("tool_definitions")
    if tool_definitions is None:
        tool_definitions = []
    if not isinstance(tool_definitions, dict) and not (
        isinstance(tool_definitions, list)
        and all(isinstance(tool, dict) for tool in tool_definitions)
    ):
        raise _input_error("tool_definitions must be an object or an array of objects.")
    description = eval_input.get("test_case_description", "")
    if not isinstance(description, str):
        raise _input_error("test_case_description must be a string.")
    judge_input: _JudgeInput = {
        "messages": messages,
        "rubric_dimensions": dimensions,
        "tool_definitions": _copy_task_input(tool_definitions),
        "test_case_description": description,
    }
    try:
        json.dumps(judge_input, allow_nan=False)
    except RecursionError as exc:
        raise _input_error("Task inputs are too deeply nested to process.") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise _input_error(
            "Task inputs must contain JSON-serializable values without NaN or infinity."
        ) from exc
    return judge_input


def _criteria_hash(dimensions: list[_RubricDimension]) -> str:
    normalized = json.dumps(
        sorted(dimensions, key=lambda dimension: dimension["id"]),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class TaskLevelRubricEvaluator(PromptyEvaluatorBase[bool]):
    """Evaluate a full trace, returning verdicts, explanations, and run metadata.

    The judge uses a temporary placeholder prompt for integration testing,
    not an approved production grading prompt.

    :param model_config: Model configuration used by the temporary judge flow.
    :keyword credential: Optional credential for the judge deployment.
    :keyword threshold: Internal weighted pass threshold in [0, 1], default 1.
    :keyword is_reasoning_model: Whether the deployment is a reasoning model.
    """

    _RESULT_KEY = "task_level_rubric"
    _PROMPTY_FILE = "task_level_rubric.prompty"
    _VERSION = 1
    id = "azureai://built-in/evaluators/task_level_rubric"

    def __init__(
        self,
        model_config: dict,
        *,
        credential=None,
        threshold: float = 1,
        is_reasoning_model: bool = False,
    ):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
            or not math.isfinite(threshold)
        ):
            raise _input_error("threshold must be a finite number between 0 and 1.")
        super().__init__(
            model_config=model_config,
            credential=credential,
            threshold=threshold,
            result_key=self._RESULT_KEY,
            prompty_file=os.path.join(os.path.dirname(__file__), self._PROMPTY_FILE),
            _higher_is_better=True,
            is_reasoning_model=is_reasoning_model,
        )
        judge_deployment = model_config.get("azure_deployment") or model_config.get(
            "model"
        )
        if not isinstance(judge_deployment, str) or not judge_deployment.strip():
            raise _input_error(
                "model_config must specify the judge deployment or model."
            )
        self._judge_deployment = judge_deployment

    def __call__(
        self,
        *,
        rubric_dimensions: list[dict[str, Any]] | None = None,
        query: str | list[dict[str, Any]] | None = None,
        response: str | list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        tool_definitions: dict[str, Any] | list[dict[str, Any]] | None = None,
        test_case_description: str = "",
    ) -> dict[str, Any]:
        """Evaluate one Task using the temporary, testing-only judge prompt."""
        return super().__call__(
            rubric_dimensions=rubric_dimensions,
            query=query,
            response=response,
            messages=messages,
            tool_definitions=tool_definitions,
            test_case_description=test_case_description,
        )

    async def _real_call(self, **kwargs):
        # A row is one complete trace; do not use the base class's per-turn conversion.
        return await self._do_eval(kwargs)

    async def _do_eval(self, eval_input: dict[str, Any]) -> dict[str, Any]:
        judge_input = _prepare_judge_input(eval_input)
        dimensions = judge_input["rubric_dimensions"]
        logger.debug(
            "Task Level rubric criteria_source=dataset_row criteria_hash=%s",
            _criteria_hash(dimensions),
        )
        # Retain a private snapshot even if the future judge adapter modifies its request.
        judge_output = await self._invoke_judge(_copy_task_input(judge_input))
        return self._parse_judge_output(judge_output, dimensions)

    async def _invoke_judge(self, judge_input: _JudgeInput) -> dict[str, Any]:
        """Call the temporary judge, preserving the response metadata envelope."""
        prompt_input = {
            field: json.dumps(value, allow_nan=False)
            for field, value in judge_input.items()
        }
        try:
            return await self._flow(timeout=self._LLM_CALL_TIMEOUT, **prompt_input)
        except json.JSONDecodeError as exc:
            raise _judge_error("Judge output must be valid JSON.") from exc

    def _build_sample(self, metadata: dict[str, Any]) -> dict[str, Any]:
        usage: dict[str, int | None] = {}
        for output_field, source_field in _USAGE_FIELDS.items():
            count = metadata.get(source_field)
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise _judge_error(
                    f"Judge usage.{output_field} must be a nonnegative integer or null."
                )
            usage[output_field] = count
        return {"model": self._judge_deployment, "usage": usage}

    def _parse_judge_output(
        self, judge_output: Any, dimensions: list[_RubricDimension]
    ) -> dict[str, Any]:
        if not isinstance(judge_output, dict):
            raise _judge_error("Judge output must contain a dimensions array.")
        llm_output = judge_output.get("llm_output", judge_output)
        if not isinstance(llm_output, dict) or not isinstance(
            llm_output.get("dimensions"), list
        ):
            raise _judge_error("Judge output must contain a dimensions array.")
        results = llm_output["dimensions"]
        if len(results) != len(dimensions):
            raise _judge_error(
                "Judge output must contain exactly one result for every supplied dimension."
            )
        expected_ids = {dimension["id"] for dimension in dimensions}
        by_id: dict[str, dict[str, Any]] = {}
        for result in results:
            if not isinstance(result, dict):
                raise _judge_error("Each judge dimension result must be an object.")
            dimension_id = result.get("id")
            if (
                not isinstance(dimension_id, str)
                or dimension_id not in expected_ids
                or dimension_id in by_id
            ):
                raise _judge_error(
                    "Judge dimension IDs must match the supplied rubric exactly, without duplicates."
                )
            score = result.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or score not in (1, 2, 3, 4, 5)
            ):
                raise _judge_error(
                    "Each judge dimension score must be a whole number from 1 to 5."
                )
            reason = result.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise _judge_error(
                    "Each judge dimension result must include a nonblank reason."
                )
            by_id[dimension_id] = result

        dimension_passed = {
            dimension_id: result["score"] >= 3 for dimension_id, result in by_id.items()
        }
        # Exact arithmetic prevents a tiny failed weight from rounding an almost-perfect score to 1.
        weighted_score = Fraction(
            sum(
                dimension["weight"]
                for dimension in dimensions
                if dimension_passed[dimension["id"]]
            ),
            sum(dimension["weight"] for dimension in dimensions),
        )
        passed = weighted_score >= self._threshold
        all_passed = all(dimension_passed.values())
        reason = (
            "All rubric dimensions passed."
            if all_passed
            else "The Task passed overall, although some rubric dimensions failed."
            if passed
            else "The Task did not meet the rubric requirements."
        )
        prefix = self._RESULT_KEY
        # Bare judge JSON is grading content, not a trusted source of usage telemetry.
        sample = self._build_sample(
            judge_output if "llm_output" in judge_output else {}
        )
        return {
            prefix: passed,
            f"{prefix}_passed": passed,
            f"{prefix}_result": "pass" if passed else "fail",
            f"{prefix}_reason": reason,
            f"{prefix}_status": "completed",
            "total_dimension_count": len(dimensions),
            "evaluator_version": self._VERSION,
            "sample": sample,
            f"{prefix}_properties": {
                "dimensions": [
                    {
                        "id": dimension["id"],
                        "result": "pass"
                        if dimension_passed[dimension["id"]]
                        else "fail",
                        "reason": by_id[dimension["id"]]["reason"],
                    }
                    for dimension in dimensions
                ]
            },
        }
