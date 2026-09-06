# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
IFEval (Instruction-Following Evaluation) Evaluator.

This evaluator implements the IFEval benchmark from Google Research for
evaluating how well LLMs follow natural language instructions. It checks
verifiable instructions like word count, format constraints, keyword usage, etc.

Reference: https://github.com/google-research/google-research/tree/master/instruction_following_eval
"""

import json
import logging
from typing import Any, Dict
from typing_extensions import overload, override

from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING

from ._instructions import get_checker

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import _preprocess_messages
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import _preprocess_messages

logger = logging.getLogger(__name__)


def _extract_final_text_response(messages):
    """Extract only the plain text of assistant messages, dropping tool calls/results.

    Iterates the preprocessed messages and collects the ``text`` content of every
    ``assistant`` role message, ignoring any ``tool_call``/``tool_result`` content blocks
    and any non-assistant messages (e.g. ``tool`` role messages). This ensures only the
    final plain-text agent response is used for evaluation, never intermediate tool call
    or tool result content.

    :param messages: The preprocessed list of chat-message dicts.
    :type messages: list
    :return: The joined plain text of all assistant messages, or an empty string if none.
    :rtype: str
    """
    text_lines = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for content in msg.get("content", []) or []:
                if isinstance(content, dict) and "text" in content and content.get("type", "text") == "text":
                    text_lines.append(content["text"])
    return "\n".join(text_lines)


def _parse_response_for_evaluation(response):
    """Flatten ``response`` into a plain string, parsing a JSON-encoded list of messages if needed.

    If ``response`` is a string, attempt to ``json.loads`` it. When it (or an already-parsed
    ``response``) is a list of chat-message dicts, only the plain text of assistant messages
    is extracted (tool calls, tool results, and other message types are dropped). Any other
    input (a plain string that is not JSON, or a string/list that fails to parse or yields no
    assistant text) is returned unchanged.

    :param response: The raw response value from the eval input.
    :type response: Any
    :return: A plain string ready for deterministic scoring.
    :rtype: str
    """
    parsed = response
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except (ValueError, TypeError):
            return response
    if isinstance(parsed, list):
        try:
            messages = _preprocess_messages(parsed)
            text = _extract_final_text_response(messages)
            if text:
                return text
        except Exception:
            logger.debug("Could not extract plain text from response messages; falling back to original response")
    return response


class IFEvalEvaluator(EvaluatorBase):
    r"""Evaluator for Instruction-Following Eval (IFEval) benchmark.

    This evaluator checks whether model outputs comply with verifiable
    instructions such as word count constraints, format requirements, keyword
    usage, and language specifications. It returns both strict (exact compliance)
    and loose (minor tolerance) accuracy scores.

    The evaluator supports 25+ instruction types including:
    - Language constraints (respond in specific language)
    - Length constraints (word count, sentence count)
    - Format constraints (JSON, bullet lists, sections, paragraphs)
    - Content constraints (keywords, forbidden words, placeholders)
    - Style constraints (all caps, all lowercase, quotation marks)

    :return: A dictionary with the evaluation results.
    :rtype: Dict[str, any]

    .. admonition:: Example:

        .. code-block:: python

            from azure.ai.evaluation import IFEvalEvaluator

            evaluator = IFEvalEvaluator()

            bullets = "* Point 1\n* Point 2\n* Point 3\n* Point 4\n* Point 5"
            result = evaluator(
                response=f"Here is my response with exactly five bullet points:\n{bullets}",
                instruction_id_list='["detectable_format:number_bullet_lists"]',
                instruction_kwargs='[{"num_bullets": 5}]'
            )
            # result: {"ifeval_strict": True, "ifeval_loose": True, "ifeval_result": "pass"}

    .. admonition:: Example with multiple instructions:

        .. code-block:: python

            result = evaluator(
                response="This is a response with at least 10 words and no commas used anywhere.",
                instruction_id_list='["length_constraints:number_words", "punctuation:no_comma"]',
                instruction_kwargs='[{"num_words": 10, "relation": "at least"}, {}]'
            )
            # result: {"ifeval_strict": True, "ifeval_loose": True, "ifeval_result": "pass"}
    """

    id = "azureai://built-in/evaluators/ifeval"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    def __init__(self):
        """Initialize the IFEval evaluator."""
        super().__init__()
        logger.debug("IFEvalEvaluator initialized")

    @staticmethod
    def _parse_json_field(value: Any, field_name: str) -> Any:
        """Parse a JSON field that may be a string or already parsed.

        :param value: The value to parse (string or already parsed).
        :type value: Any
        :param field_name: Name of the field for error messages.
        :type field_name: str
        :return: The parsed value.
        :rtype: Any
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse %s as JSON: %s", field_name, e)
                return None
        return value

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, any]:
        """Produce an IFEval evaluation result.

        Evaluates whether the model response follows all the specified
        instructions, returning both strict and loose accuracy.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result with strict and loose scores.
        :rtype: Dict
        """
        response = _parse_response_for_evaluation(eval_input.get("response", ""))
        instruction_id_list = self._parse_json_field(
            eval_input.get("instruction_id_list"), "instruction_id_list"
        )
        kwargs_list = self._parse_json_field(
            eval_input.get("instruction_kwargs"), "instruction_kwargs"
        )

        # Validate inputs
        if not response:
            logger.debug("Empty response, returning False")
            return {
                "ifeval_strict": False,
                "ifeval_loose": False,
                "ifeval_result": EVALUATION_PASS_FAIL_MAPPING[False],
            }

        if not instruction_id_list:
            logger.warning("Empty or invalid instruction_id_list")
            return {
                "ifeval_strict": False,
                "ifeval_loose": False,
                "ifeval_result": EVALUATION_PASS_FAIL_MAPPING[False],
            }

        if not kwargs_list:
            kwargs_list = [{} for _ in instruction_id_list]

        if len(instruction_id_list) != len(kwargs_list):
            logger.error(
                "Mismatch: %d instruction IDs vs %d kwargs entries",
                len(instruction_id_list),
                len(kwargs_list),
            )
            return {
                "ifeval_strict": False,
                "ifeval_loose": False,
                "ifeval_result": EVALUATION_PASS_FAIL_MAPPING[False],
            }

        # Check each instruction
        strict_results = []
        loose_results = []

        for i, (inst_id, inst_kwargs) in enumerate(zip(instruction_id_list, kwargs_list)):
            checker = get_checker(inst_id, inst_kwargs or {})
            if checker is None:
                logger.warning("Skipping unknown instruction: %s", inst_id)
                strict_results.append(False)
                loose_results.append(False)
                continue

            try:
                strict_pass = checker.check_following(response)
                loose_pass = checker.check_following_loose(response)
            except Exception as e:
                logger.error("Error checking instruction %s: %s", inst_id, e)
                strict_pass = False
                loose_pass = False

            strict_results.append(strict_pass)
            loose_results.append(loose_pass)
            logger.debug(
                "Instruction %d (%s): strict=%s, loose=%s",
                i, inst_id, strict_pass, loose_pass
            )

        # All instructions must pass for overall pass
        all_strict = all(strict_results)
        all_loose = all(loose_results)

        logger.debug(
            "IFEval evaluation: strict=%s (%d/%d), loose=%s (%d/%d)",
            all_strict,
            sum(strict_results),
            len(strict_results),
            all_loose,
            sum(loose_results),
            len(loose_results),
        )

        return {
            "ifeval_strict": all_strict,
            "ifeval_loose": all_loose,
            "ifeval_result": EVALUATION_PASS_FAIL_MAPPING[all_strict],
        }

    @overload  # type: ignore
    def __call__(
        self,
        *,
        response: str,
        instruction_id_list: str,
        instruction_kwargs: str
    ) -> Dict[str, any]:
        """
        Evaluate whether the response follows all specified instructions.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :keyword instruction_id_list: JSON array of instruction IDs.
        :paramtype instruction_id_list: str
        :keyword instruction_kwargs: JSON array of parameter dicts for each instruction.
        :paramtype instruction_kwargs: str
        :return: The evaluation result containing strict and loose accuracy.
        :rtype: Dict[str, any]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kw,
    ):
        """
        Evaluate whether the response follows all specified instructions.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :keyword instruction_id_list: JSON array of instruction IDs.
        :paramtype instruction_id_list: str
        :keyword instruction_kwargs: JSON array of parameter dicts for each instruction.
        :paramtype instruction_kwargs: str
        :return: The evaluation result containing strict and loose accuracy.
        :rtype: Dict[str, any]
        """
        return super().__call__(*args, **kw)
