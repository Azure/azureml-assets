# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
from typing import Dict, List, Optional, Union

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import ErrorTarget
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ConversationValidator,
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
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)
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
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import _coerce_bool, _coerce_number, _log_safe_summary
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)
    # Bodies below are copied from azure-ai-evaluation 1.18.1 (the earliest release
    # that ships these symbols).
    def _coerce_bool(value) -> Optional[bool]:
        """Coerce an LLM output value to bool or None.

        Handles Python booleans and string variants like 'true', 'false'.

        :param value: The value to coerce.
        :type value: Any
        :return: The coerced boolean, or None if it cannot be interpreted.
        :rtype: Optional[bool]
        """
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower == "true":
                return True
            if lower == "false":
                return False
        return None

    def _coerce_number(value) -> Optional[float]:
        """Coerce an LLM output value to a number or None.

        Handles Python ints/floats and string variants like '3', '2.5', 'null'.

        :param value: The value to coerce.
        :type value: Any
        :return: The coerced number, or None if it cannot be interpreted.
        :rtype: Optional[float]
        """
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in ("null", "none", ""):
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

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

try:
    from azure.ai.evaluation._user_agent import UserAgentSingleton
except ImportError:

    class UserAgentSingleton:
        """Fallback singleton for user agent when import fails."""

        @property
        def value(self) -> str:
            """Return the user agent value."""
            return "None"


try:
    from azure.ai.evaluation._common.utils import construct_prompty_model_config, validate_model_config
except ImportError:
    from ..._common.utils import construct_prompty_model_config, validate_model_config

logger = logging.getLogger(__name__)


# Use the SDK's ErrorTarget member when the installed version defines it; otherwise fall back to EVALUATE.
_ERROR_TARGET = getattr(ErrorTarget, "QUALITY_GRADER_EVALUATOR", ErrorTarget.EVALUATE)


# Thresholds for response quality checks (first prompt)
_QUALITY_RELEVANCE_THRESHOLD = 2.5
_ANSWER_COMPLETENESS_THRESHOLD = 1.5

# Thresholds for groundedness checks (second prompt)
_GROUNDEDNESS_THRESHOLD = 2.5
_CONTEXT_COVERAGE_THRESHOLD = 1.5


class QualityGraderEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """Evaluates overall response quality using a two-stage grading pipeline.

    Stage 1 (Response Quality): Evaluates the response for relevance, abstention, and answer completeness.
    The response must satisfy:
        - abstention must be false
        - relevance must be greater than 2.5 (on a 1-5 scale)
        - answerCompleteness must be greater than 1.5

    Stage 2 (Groundedness, only if context is provided): Evaluates whether the response is grounded in the
    provided context and covers the key information. The response must satisfy:
        - groundedness must be greater than 2.5
        - contextCoverage must exceed 1.5

    If all checks pass, the evaluator returns "pass". Otherwise, it returns "fail" with failure reasons.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, updates config parameters for reasoning models. Defaults to False.
    :paramtype is_reasoning_model: bool
    """

    _RESPONSE_QUALITY_PROMPTY = "quality_grader_response_quality.prompty"
    _GROUNDEDNESS_PROMPTY = "quality_grader_groundedness.prompty"
    _RESULT_KEY = "quality_grader"
    _OPTIONAL_PARAMS = ["context"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/quality_grader"

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        """Initialize a QualityGraderEvaluator instance."""
        current_dir = os.path.dirname(__file__)
        response_quality_prompty_path = os.path.join(current_dir, self._RESPONSE_QUALITY_PROMPTY)

        self._higher_is_better = True
        self._model_config = model_config
        self._credential = credential

        # Initialize input validator
        self._validator = ConversationValidator(
            error_target=_ERROR_TARGET,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=response_quality_prompty_path,
            result_key=self._RESULT_KEY,
            threshold=1,
            credential=credential,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

        # Load the second prompty flow for groundedness evaluation
        groundedness_prompty_path = os.path.join(current_dir, self._GROUNDEDNESS_PROMPTY)
        subclass_name = self.__class__.__name__
        user_agent = f"{UserAgentSingleton().value} (type=evaluator subtype={subclass_name})"
        prompty_model_config = construct_prompty_model_config(
            validate_model_config(model_config),
            self._DEFAULT_OPEN_API_VERSION,
            user_agent,
        )
        self._groundedness_flow = AsyncPrompty.load(
            source=groundedness_prompty_path,
            model=prompty_model_config,
            token_credential=credential,
            is_reasoning_model=kwargs.get("is_reasoning_model", False),
        )

    @overload
    def __call__(
        self,
        *,
        query: str,
        response: str,
        context: Optional[str] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate quality for a given query, response, and optional context.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword context: The context (retrieved documents) to evaluate groundedness against. Optional.
        :paramtype context: Optional[str]
        :return: The quality grader result.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate quality for a conversation.

        :keyword conversation: The conversation to evaluate.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The quality grader result.
        :rtype: Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]
        """

    @override
    def __call__(self, *args, **kwargs):
        """Evaluate quality for a query/response pair with optional context for groundedness or a conversation.

        :return: The quality grader result.
        :rtype: Dict[str, Union[str, float]]
        """
        return super().__call__(*args, **kwargs)

    def _return_not_applicable_result(
        self, error_message: str,
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: None,
            f"{self._result_key}_score": None,
            f"{self._result_key}_result": "not_applicable",
            f"{self._result_key}_passed": None,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_status": "skipped",
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": None,
        }

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
        self._validator.validate_eval_input(kwargs)

        return await super()._real_call(**kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str, Dict]]:  # type: ignore[override]
        """Run the two-stage quality grading pipeline.

        Stage 1: Call the response quality prompt and check thresholds.
        Stage 2 (if context provided): Call the groundedness prompt and check thresholds.
        """
        # Handle intermediate responses
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
            )

        # Preprocess messages
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        query = eval_input.get("query", "")
        response = eval_input.get("response", "")
        context = eval_input.get("context", None)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        model_id = ""

        # --- Stage 1: Response Quality ---
        stage1_input = {"question": query, "response": response}
        stage1_output = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **stage1_input)

        stage1_parsed = self._parse_prompty_json_output(stage1_output)
        total_prompt_tokens += stage1_output.get("input_token_count", 0) if stage1_output else 0
        total_completion_tokens += stage1_output.get("output_token_count", 0) if stage1_output else 0
        total_tokens += stage1_output.get("total_token_count", 0) if stage1_output else 0
        model_id = stage1_output.get("model_id", "") if stage1_output else ""

        # If stage 1 was skipped (conversationIncomplete = true), return not applicable
        if stage1_parsed.get("status") == "skipped":
            return self._return_not_applicable_result(
                "Conversation is incomplete or consists only of greetings/closings with no task to evaluate.",
            )

        # Check stage 1 conditions
        failure_reasons = []
        stage1_props = stage1_parsed.get("properties", {})
        abstention = _coerce_bool(stage1_props.get("abstention"))
        relevance = _coerce_number(stage1_props.get("relevance"))
        answer_completeness = _coerce_number(stage1_props.get("answerCompleteness"))

        if abstention is True:
            failure_reasons.append("abstention is true (expected false)")
        if isinstance(relevance, (int, float)) and relevance <= _QUALITY_RELEVANCE_THRESHOLD:
            failure_reasons.append(
                f"relevance is {relevance} (must be > {_QUALITY_RELEVANCE_THRESHOLD})"
            )
        elif relevance is None:
            failure_reasons.append(f"relevance is null (must be > {_QUALITY_RELEVANCE_THRESHOLD})")
        if isinstance(answer_completeness, (int, float)) and answer_completeness <= _ANSWER_COMPLETENESS_THRESHOLD:
            failure_reasons.append(
                f"answerCompleteness is {answer_completeness} (must be > {_ANSWER_COMPLETENESS_THRESHOLD})"
            )
        elif answer_completeness is None:
            failure_reasons.append(f"answerCompleteness is null (must be > {_ANSWER_COMPLETENESS_THRESHOLD})")

        if failure_reasons:
            return self._build_result(
                passed=False,
                failure_reasons=failure_reasons,
                stage1_parsed=stage1_parsed,
                stage2_parsed=None,
                stage1_output=stage1_output,
                stage2_output=None,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                model_id=model_id,
            )

        # --- Stage 2: Groundedness (only if context is provided) ---
        stage2_parsed = None
        stage2_output = None
        if context:
            stage2_input = {"question": query, "response": response, "context": context}
            stage2_output = await self._groundedness_flow(timeout=self._LLM_CALL_TIMEOUT, **stage2_input)

            stage2_parsed = self._parse_prompty_json_output(stage2_output)
            total_prompt_tokens += stage2_output.get("input_token_count", 0) if stage2_output else 0
            total_completion_tokens += stage2_output.get("output_token_count", 0) if stage2_output else 0
            total_tokens += stage2_output.get("total_token_count", 0) if stage2_output else 0

            stage2_props = stage2_parsed.get("properties", {})
            groundedness = _coerce_number(stage2_props.get("groundedness"))
            context_coverage = _coerce_number(stage2_props.get("contextCoverage"))

            if isinstance(groundedness, (int, float)) and groundedness <= _GROUNDEDNESS_THRESHOLD:
                failure_reasons.append(
                    f"groundedness is {groundedness} (must be > {_GROUNDEDNESS_THRESHOLD})"
                )
            elif groundedness is None:
                failure_reasons.append(f"groundedness is null (must be > {_GROUNDEDNESS_THRESHOLD})")

            if isinstance(context_coverage, (int, float)) and context_coverage <= _CONTEXT_COVERAGE_THRESHOLD:
                failure_reasons.append(
                    f"contextCoverage is {context_coverage} (must exceed {_CONTEXT_COVERAGE_THRESHOLD})"
                )
            elif context_coverage is None:
                failure_reasons.append(f"contextCoverage is null (must exceed {_CONTEXT_COVERAGE_THRESHOLD})")

            if failure_reasons:
                return self._build_result(
                    passed=False,
                    failure_reasons=failure_reasons,
                    stage1_parsed=stage1_parsed,
                    stage2_parsed=stage2_parsed,
                    stage1_output=stage1_output,
                    stage2_output=stage2_output,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_tokens,
                    model_id=model_id,
                )

        # All checks passed
        return self._build_result(
            passed=True,
            failure_reasons=[],
            stage1_parsed=stage1_parsed,
            stage2_parsed=stage2_parsed,
            stage1_output=stage1_output,
            stage2_output=stage2_output,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            model_id=model_id,
        )

    @staticmethod
    def _parse_prompty_json_output(prompty_output: Optional[Dict]) -> Dict:
        """Parse the JSON output from a prompty flow call.

        :param prompty_output: The raw output dict from the prompty flow.
        :return: Parsed JSON dict from the LLM output.
        """
        if not prompty_output:
            return {}
        llm_output = prompty_output.get("llm_output", prompty_output)
        if not llm_output:
            return {}
        if isinstance(llm_output, dict):
            return llm_output
        try:
            return json.loads(llm_output)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Failed to parse LLM output as JSON. Output shape: %s. Error: %s",
                _log_safe_summary(llm_output),
                e,
            )
            return {}

    def _build_result(
        self,
        *,
        passed: bool,
        failure_reasons: List[str],
        stage1_parsed: Optional[Dict],
        stage2_parsed: Optional[Dict],
        stage1_output: Optional[Dict],
        stage2_output: Optional[Dict],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model_id: str,
    ) -> Dict[str, Union[str, float, Dict]]:
        """Build the standardized result dictionary.

        :param passed: Whether the evaluation passed.
        :param failure_reasons: List of reasons for failure (empty if passed).
        :param stage1_parsed: Parsed output from stage 1 (response quality).
        :param stage2_parsed: Parsed output from stage 2 (groundedness), or None if not run.
        :param stage1_output: Raw prompty output from stage 1.
        :param stage2_output: Raw prompty output from stage 2, or None if not run.
        :param prompt_tokens: Total prompt tokens used.
        :param completion_tokens: Total completion tokens used.
        :param total_tokens: Total tokens used.
        :param model_id: The model ID used.
        :return: Standardized result dict.
        """
        score = 1.0 if passed else 0.0
        result_label = self._PASS_RESULT if passed else self._FAIL_RESULT

        # Build reason from LLM reasoning fields, concatenating both stages when available.
        reasoning_parts = []
        if stage1_parsed and stage1_parsed.get("reasoning"):
            reasoning_parts.append(stage1_parsed["reasoning"])
        if stage2_parsed and stage2_parsed.get("reasoning"):
            reasoning_parts.append(stage2_parsed["reasoning"])
        llm_reasoning = " ".join(reasoning_parts)

        reason = llm_reasoning if llm_reasoning else (
            "All quality checks passed." if passed else "; ".join(failure_reasons)
        )

        properties = {}
        if stage1_parsed:
            stage1_props = stage1_parsed.get("properties", {})
            properties["abstention"] = stage1_props.get("abstention")
            properties["relevance"] = stage1_props.get("relevance")
            properties["answerCompleteness"] = stage1_props.get("answerCompleteness")
            properties["queryType"] = stage1_props.get("queryType")
            properties["conversationIncomplete"] = stage1_props.get("conversationIncomplete")
            properties["judgeConfidence"] = stage1_props.get("judgeConfidence")
            properties["stage1_explanation"] = stage1_props.get("explanation", {})
            properties["stage1_reasoning"] = stage1_parsed.get("reasoning", "")
            properties["stage1_score"] = stage1_parsed.get("score")
            properties["stage1_status"] = stage1_parsed.get("status", "")

        if stage2_parsed:
            stage2_props = stage2_parsed.get("properties", {})
            properties["groundedness"] = stage2_props.get("groundedness")
            properties["contextCoverage"] = stage2_props.get("contextCoverage")
            properties["documentUtility"] = stage2_props.get("documentUtility")
            properties["missingContextParts"] = stage2_props.get("missingContextParts", [])
            properties["unsupportedClaims"] = stage2_props.get("unsupportedClaims", [])
            properties["stage2_explanation"] = stage2_props.get("explanation", {})
            properties["stage2_reasoning"] = stage2_parsed.get("reasoning", "")
            properties["stage2_score"] = stage2_parsed.get("score")
            properties["stage2_status"] = stage2_parsed.get("status", "")

        # Build per-stage diagnostic lists; only include entries for stages that ran.
        finish_reasons = []
        sample_inputs = []
        sample_outputs = []
        for raw in [stage1_output, stage2_output]:
            if raw:
                finish_reasons.append(raw.get("finish_reason", ""))
                sample_inputs.append(raw.get("sample_input", ""))
                sample_outputs.append(raw.get("sample_output", ""))

        properties["prompt_tokens"] = prompt_tokens
        properties["completion_tokens"] = completion_tokens
        properties["total_tokens"] = total_tokens
        properties["finish_reason"] = finish_reasons
        properties["model"] = model_id
        properties["sample_input"] = sample_inputs
        properties["sample_output"] = sample_outputs

        return {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result_label,
            f"{self._result_key}_passed": passed,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": "completed",
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": properties,
        }
