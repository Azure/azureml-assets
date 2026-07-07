# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Composable low-level unit tests for the repeated evaluator validator/util code.

The repeated building blocks (``ConversationValidator``,
``MessagesOrQueryResponseInputValidator``, ``serialize_messages``,
``_preprocess_messages`` ...) are byte-for-byte copies in every evaluator's
``_<name>.py`` source. Their corner-case tests live here once, split into
capability-scoped mixins so each evaluator's behavior file composes exactly the
mixins matching the surface it exposes - no test is skipped for a capability the
evaluator does not have::

    class TestCoherenceValidatorUnit(
        CorePromptyValidatorUnitTests,
        SuperDoEvalNotApplicableUnitTests,
        MessagePreprocessUnitTests,
        ConversationValidatorUnitTests,
        ConversationValidatorToolCheckUnitTests,
        ConversationSerializationUnitTests,
        MessagesOrQueryResponseUnitTests,
    ):
        evaluator_class = CoherenceEvaluator

Every mixin derives from :class:`_ValidatorUnitTestSupport`, which holds the
shared configuration (``evaluator_class`` and the derived ``validator_module`` /
``error_target``), the capability manifest, the per-evaluator divergence maps and
all the construction/resolution helpers. Utilities are resolved by *name* on the
evaluator's module (see ``_module_fn``) and base methods fall back to
``super()._<name>`` (see ``_super_impl``), so the tests keep working whether a
util is inlined in the source or imported from the ``azure-ai-evaluation`` SDK.

``CorePromptyValidatorUnitTests`` carries ``test_capability_manifest``, which
pins the exact capability surface of each evaluator (``_CAPABILITY_BASELINE``)
and fails loudly on drift: a removed capability is a likely regression, an added
one means the source is converging and both the baseline and the mixins composed
by that evaluator's behavior file must be updated deliberately.
"""

import asyncio
import inspect
import json
import logging
import os
import sys
from typing import Any

import pytest
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException, ErrorCategory

from ..common.evaluator_mock_config import OutputType, create_flow_side_effect


class _ValidatorUnitTestSupport:
    """Shared configuration, capability manifest, divergence maps and helpers.

    Every ``*UnitTests`` mixin inherits from this base so that, however a
    behavior file composes them, the construction/resolution helpers and the
    per-evaluator data are always available. The base class name is
    underscore-prefixed and no mixin name starts with ``Test``, so pytest never
    collects them directly - only the composed ``Test<Name>ValidatorUnit``
    subclasses run.
    """

    validator_module: Any = None
    evaluator_class: Any = None
    error_target: Any = None

    # ------------------------------------------------------------------
    # Per-evaluator behaviour maps for currently-divergent code paths.
    # These pin the inconsistent behaviour catalogued in
    # EVALUATOR_DISCREPANCIES.md. Once the source is harmonised these maps
    # should collapse to a single expected outcome and can be removed.
    # ------------------------------------------------------------------

    # customer_satisfaction reports a skipped prompty status as "skipped" rather
    # than the conventional "not_applicable".
    _PARSE_SKIPPED_RESULT = {"customer_satisfaction": "skipped"}

    # Expected outcome of _parse_prompty_output for three malformed llm_outputs,
    # in order: (missing-score dict, non-numeric-score dict, non-dict output).
    # A string is the returned "<key>_status"; an exception class means it raises.
    _PARSE_MALFORMED_EXPECTATIONS = {
        "coherence": ("error", "error", "error"),
        "customer_satisfaction": ("completed", TypeError, "error"),
        "groundedness": ("completed", "error", "error"),
        "task_adherence": ("completed", ValueError, EvaluationException),
        "task_completion": ("completed", ValueError, EvaluationException),
    }

    # _do_eval with a missing score: groundedness/tool_output_utilization raise
    # KeyError (they require extra inputs before scoring); the evaluators below
    # degrade to a result dict; every other evaluator raises EvaluationException.
    _DO_EVAL_MISSING_SCORE_KEYERROR = frozenset({"groundedness", "tool_output_utilization"})
    _DO_EVAL_MISSING_SCORE_RETURNS_DICT = frozenset(
        {
            "relevance",
            "customer_satisfaction",
            "task_adherence",
            "task_completion",
            "deflection_rate",
            "quality_grader",
            "tool_call_success",
        }
    )

    # quality_grader overrides _return_not_applicable_result with a 1-arg
    # signature incompatible with the base _do_eval (which passes a threshold),
    # so the super()._do_eval fallback cannot exercise its not-applicable path.
    _SUPER_DO_EVAL_NOT_APPLICABLE_UNSUPPORTED = frozenset({"quality_grader"})

    # ------------------------------------------------------------------
    # Capability manifest (skip-hiding guard).
    #
    # Most per-capability tests below auto-skip when the probed helper/validator
    # is absent from the evaluator under test. That is convenient (the same mixin
    # fits every evaluator) but risky: if a helper that *should* exist is removed
    # by mistake, its tests would silently skip instead of fail. To stop a
    # regression from hiding behind a skip, ``test_capability_manifest`` pins the
    # exact set of probed capabilities each evaluator currently exposes and fails
    # loudly on any drift (removed => possible regression; added => source
    # converging, update the baseline).
    # ------------------------------------------------------------------

    # Module-level names probed from the evaluator's ``_<name>.py`` source module.
    _MODULE_CAPABILITIES = (
        "ConversationValidator",
        "MessagesOrQueryResponseInputValidator",
        "EvaluationLevel",
        "serialize_messages",
        "_merge_query_response_messages",
        "_split_messages_at_latest_user",
        "_wrap_string_messages",
        "_resolve_evaluation_level",
        "_is_intermediate_response",
        "_drop_mcp_approval_messages",
        "_normalize_function_call_types",
        "_preprocess_messages",
    )
    # Instance methods probed on the evaluator object.
    _METHOD_CAPABILITIES = (
        "_get_token_metadata",
        "_return_not_applicable_result",
        "_should_use_conversation_level",
        "_build_result",
        "_parse_prompty_output",
        "_do_eval",
        "_real_call",
        "_the_super_do_eval",
        "_the_super_real_call",
        "_convert_kwargs_to_eval_input",
    )
    # Frozen per-evaluator capability surface (space-separated for compact diffs).
    # Regenerate deliberately when the source converges; never loosen to silence
    # a failure without confirming the capability change is intentional.
    _CAPABILITY_BASELINE = {
        "coherence": (
            "ConversationValidator EvaluationLevel MessagesOrQueryResponseInputValidator _build_result"
            " _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _merge_query_response_messages _normalize_function_call_types"
            " _parse_prompty_output _preprocess_messages _real_call _resolve_evaluation_level"
            " _return_not_applicable_result _should_use_conversation_level _split_messages_at_latest_user"
            " _the_super_do_eval _the_super_real_call _wrap_string_messages serialize_messages"
        ),
        "fluency": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result _the_super_do_eval"
            " _the_super_real_call"
        ),
        "relevance": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result _the_super_real_call"
        ),
        "retrieval": (
            "_convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _normalize_function_call_types _preprocess_messages _real_call"
            " _return_not_applicable_result _the_super_do_eval"
        ),
        "similarity": (
            "_convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _normalize_function_call_types _preprocess_messages _real_call"
            " _return_not_applicable_result _the_super_do_eval"
        ),
        "response_completeness": (
            "_convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _normalize_function_call_types _preprocess_messages _real_call"
            " _return_not_applicable_result"
        ),
        "groundedness": (
            "ConversationValidator EvaluationLevel MessagesOrQueryResponseInputValidator _build_result"
            " _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _merge_query_response_messages _normalize_function_call_types"
            " _parse_prompty_output _preprocess_messages _real_call _resolve_evaluation_level"
            " _return_not_applicable_result _should_use_conversation_level _split_messages_at_latest_user"
            " _the_super_do_eval _the_super_real_call _wrap_string_messages serialize_messages"
        ),
        "intent_resolution": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result _the_super_real_call"
        ),
        "customer_satisfaction": (
            "ConversationValidator EvaluationLevel _build_result _convert_kwargs_to_eval_input _do_eval"
            " _drop_mcp_approval_messages _get_token_metadata _is_intermediate_response"
            " _merge_query_response_messages _normalize_function_call_types _parse_prompty_output"
            " _preprocess_messages _real_call _resolve_evaluation_level _return_not_applicable_result"
            " _should_use_conversation_level _split_messages_at_latest_user _the_super_real_call"
            " _wrap_string_messages serialize_messages"
        ),
        "task_adherence": (
            "ConversationValidator EvaluationLevel MessagesOrQueryResponseInputValidator _build_result"
            " _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _merge_query_response_messages _normalize_function_call_types"
            " _parse_prompty_output _preprocess_messages _real_call _resolve_evaluation_level"
            " _return_not_applicable_result _should_use_conversation_level _split_messages_at_latest_user"
            " _the_super_real_call _wrap_string_messages serialize_messages"
        ),
        "task_completion": (
            "ConversationValidator EvaluationLevel MessagesOrQueryResponseInputValidator _build_result"
            " _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages _get_token_metadata"
            " _is_intermediate_response _merge_query_response_messages _normalize_function_call_types"
            " _parse_prompty_output _preprocess_messages _real_call _resolve_evaluation_level"
            " _return_not_applicable_result _should_use_conversation_level _split_messages_at_latest_user"
            " _the_super_real_call _wrap_string_messages serialize_messages"
        ),
        "deflection_rate": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _get_token_metadata"
            " _is_intermediate_response _real_call _return_not_applicable_result"
        ),
        "quality_grader": (
            "ConversationValidator _build_result _convert_kwargs_to_eval_input _do_eval"
            " _drop_mcp_approval_messages _get_token_metadata _is_intermediate_response"
            " _normalize_function_call_types _preprocess_messages _real_call _return_not_applicable_result"
        ),
        "tool_call_accuracy": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result"
        ),
        "tool_call_success": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result _the_super_real_call"
        ),
        "tool_input_accuracy": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result"
        ),
        "tool_output_utilization": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result _the_super_real_call"
        ),
        "tool_selection": (
            "ConversationValidator _convert_kwargs_to_eval_input _do_eval _drop_mcp_approval_messages"
            " _get_token_metadata _is_intermediate_response _normalize_function_call_types"
            " _preprocess_messages _real_call _return_not_applicable_result"
        ),
    }

    def _module(self):
        """Return the evaluator's source module (explicit or derived from the class)."""
        if self.validator_module is not None:
            return self.validator_module
        if self.evaluator_class is None:
            pytest.skip("set validator_module or evaluator_class")
        return sys.modules[self.evaluator_class.__module__]

    def _target(self):
        """Return the evaluator's ErrorTarget (explicit or derived from its validator)."""
        if self.error_target is not None:
            return self.error_target
        return self._make_evaluator()._validator.error_target

    def _module_fn(self, name):
        """Return a module-level function by name, or skip if the module lacks it.

        Resolution is by attribute lookup on the evaluator's module, so it finds
        the util whether it is *defined* in the ``_<name>.py`` source or *imported*
        into it under the same name (e.g. from ``azure.ai.evaluation``). These
        tests therefore keep exercising the util unchanged after the source is
        migrated to import the shared SDK copy instead of inlining it.
        """
        module = self._module()
        fn = getattr(module, name, None)
        if fn is None:
            pytest.skip(f"{name} not present in {module.__name__}")
        return fn

    def _validator(self, **kwargs):
        """Construct the asset's ConversationValidator, skipping unsupported kwargs."""
        cls = getattr(self._module(), "ConversationValidator", None)
        if cls is None:
            pytest.skip("ConversationValidator not present")
        try:
            return cls(error_target=self._target(), **kwargs)
        except TypeError:
            pytest.skip("ConversationValidator does not accept the requested kwargs")

    def _messages_or_query_response_validator(self):
        """Construct the asset's MessagesOrQueryResponseInputValidator or skip."""
        cls = getattr(self._module(), "MessagesOrQueryResponseInputValidator", None)
        if cls is None:
            pytest.skip("MessagesOrQueryResponseInputValidator not present")
        return cls(error_target=self._target())

    def _make_evaluator(self, **kwargs):
        """Construct the asset's evaluator with a dummy (mockable) model config."""
        if self.evaluator_class is None:
            pytest.skip("evaluator_class not set")
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
        )
        return self.evaluator_class(model_config=model_config, **kwargs)

    @staticmethod
    def _assert_exc(result, category=None):
        """Assert that a validator helper returned an EvaluationException."""
        assert isinstance(result, EvaluationException)
        if category is not None:
            assert result.category == category

    @staticmethod
    def _run_async(coro):
        """Run a coroutine to completion for async method tests."""
        return asyncio.run(coro)

    def _super_impl(self, evaluator, name):
        """Resolve the evaluator's copy of the base ``_<name>`` implementation.

        ``_the_super_<name>`` and the SDK's ``super()._<name>`` are the same code,
        so prefer the inlined ``_the_super_<name>`` copy the evaluator delegates
        to when present. Some evaluators instead inline that same base copy
        directly as their own ``_<name>`` override - a byte copy the SDK entry
        point bypasses, so it is otherwise never exercised; detect it by its
        ``per_turn_results`` aggregation body and drive it directly. Otherwise
        fall back to the bound ``super()._<name>`` method (resolved from the
        evaluator's parent class) so the base behaviour is still exercised.
        """
        inlined = getattr(evaluator, f"_the_super_{name}", None)
        if inlined is not None:
            return inlined
        own = type(evaluator).__dict__.get(f"_{name}")
        if own is not None:
            try:
                is_super_copy = "per_turn_results" in inspect.getsource(own)
            except (OSError, TypeError):
                is_super_copy = False
            if is_super_copy:
                return getattr(evaluator, f"_{name}")
        return getattr(super(type(evaluator), evaluator), f"_{name}", None)

    def _actual_capabilities(self, evaluator):
        """Return the set of probed capabilities the evaluator currently exposes."""
        module = self._module()
        caps = {c for c in self._MODULE_CAPABILITIES if getattr(module, c, None) is not None}
        caps |= {m for m in self._METHOD_CAPABILITIES if hasattr(evaluator, m)}
        return caps

    def _extract_tool_defs_callable(self):
        """Return an ``extract(tool_calls, tool_definitions)`` for the asset, or None.

        Matches only evaluators that inline their own copy of the util: the
        module-level 3-arg function (tool_input_accuracy, tool_selection) or the
        class-level override (tool_call_accuracy's 2-arg method). The inherited
        3-arg base method is deliberately ignored - it lives in the SDK (not the
        asset source) and its per-asset coverage impact is nil. The explicit
        ``error_target`` is supplied only when the callable's signature needs it.
        """
        fn = getattr(self._module(), "_extract_needed_tool_definitions", None)
        if fn is None:
            evaluator = self._make_evaluator()
            if type(evaluator).__dict__.get("_extract_needed_tool_definitions") is None:
                return None
            fn = evaluator._extract_needed_tool_definitions
        positional = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 3:
            target = self._target()
            return lambda calls, defs: fn(calls, defs, target)
        return lambda calls, defs: fn(calls, defs)

    def _tool_reformat_module(self):
        """Return the module if it inlines ``reformat_agent_response``, else skip.

        Only the tool evaluators inline both ``_get_agent_response`` and the
        public ``reformat_agent_response`` copy; other evaluators expose neither
        (or only the private helper), so gate on both.
        """
        module = self._module()
        if getattr(module, "_get_agent_response", None) is None:
            pytest.skip("no inlined tool-family reformat helpers")
        if getattr(module, "reformat_agent_response", None) is None:
            pytest.skip("no inlined reformat_agent_response")
        return module

    def _tool_definitions_validator_cls(self):
        """Return the asset's ``ToolDefinitionsValidator`` class, or skip."""
        cls = getattr(self._module(), "ToolDefinitionsValidator", None)
        if cls is None:
            pytest.skip("no ToolDefinitionsValidator")
        return cls


class CorePromptyValidatorUnitTests(_ValidatorUnitTestSupport):
    """Capability manifest plus base token/not-applicable/do_eval/real_call tests every prompty evaluator shares."""

    def test_capability_manifest(self):
        """Fail (not skip) when an evaluator gains or loses a probed capability.

        The per-capability tests below auto-skip when their helper/validator is
        absent, which would let a helper removed by mistake hide behind a skip.
        This test pins the exact capability surface per evaluator so any drift is
        loud: a removed capability is a likely regression; an added one means the
        source is converging and the baseline must be updated deliberately.
        """
        evaluator = self._make_evaluator()
        result_key = evaluator._result_key
        actual = self._actual_capabilities(evaluator)
        assert result_key in self._CAPABILITY_BASELINE, (
            f"No capability baseline for '{result_key}'. A newly wired evaluator must be registered in "
            f"_CAPABILITY_BASELINE. Current capabilities: {sorted(actual)}"
        )
        expected = set(self._CAPABILITY_BASELINE[result_key].split())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        assert not missing and not extra, (
            f"Capability drift for '{result_key}' (see EVALUATOR_DISCREPANCIES.md):\n"
            f"  removed (possible regression now hidden as skips): {missing}\n"
            f"  added (source converging; update the baseline): {extra}"
        )

    def test_get_token_metadata(self):
        """Cover token-metadata extraction with and without counts."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_get_token_metadata"):
            pytest.skip("no _get_token_metadata")
        md = evaluator._get_token_metadata(
            {"input_token_count": 3, "output_token_count": 5, "total_token_count": 8, "model_id": "m"}
        )
        assert md["prompt_tokens"] == 3
        assert md["completion_tokens"] == 5
        assert md["total_tokens"] == 8
        assert md["model"] == "m"
        assert evaluator._get_token_metadata({})["prompt_tokens"] == 0

    def test_return_not_applicable_result(self):
        """Cover the not-applicable result builder."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_return_not_applicable_result"):
            pytest.skip("no _return_not_applicable_result")
        result_key = evaluator._result_key
        # quality_grader takes only (reason); every other evaluator takes (reason, threshold).
        if result_key == "quality_grader":
            res = evaluator._return_not_applicable_result("because reasons")
        else:
            res = evaluator._return_not_applicable_result("because reasons", evaluator._threshold)
        assert res[result_key] is None
        assert res[f"{result_key}_result"] == "not_applicable"
        assert res[f"{result_key}_status"] == "skipped"
        assert "because reasons" in res[f"{result_key}_reason"]

    def test_is_intermediate_response(self):
        """Cover intermediate-response detection."""
        fn = self._module_fn("_is_intermediate_response")
        assert fn([{"role": "assistant", "content": [{"type": "function_call", "name": "f"}]}]) is True
        assert fn([{"role": "assistant", "content": [{"type": "mcp_approval_request"}]}]) is True
        assert fn([{"role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]) is False
        assert fn("x") is False
        assert fn([]) is False

    def test_do_eval_nan_output_raises(self):
        """A turn-level score that evaluates to NaN must not crash with TypeError."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_do_eval"):
            pytest.skip("no _do_eval")
        result_key = evaluator._result_key
        level_enum = getattr(self._module(), "EvaluationLevel", None)
        if level_enum is not None:
            evaluator._evaluation_level = level_enum.TURN

        async def flow(timeout=None, **kwargs):
            return {"llm_output": {"status": "completed", "reason": "no score field"}}

        evaluator._flow = MagicMock(side_effect=flow)
        # The invariant: a missing score is handled deterministically (never an
        # unhandled TypeError from math.isnan(None)). The exact handling diverges
        # by evaluator and is tracked in EVALUATOR_DISCREPANCIES.md.
        if result_key in self._DO_EVAL_MISSING_SCORE_KEYERROR:
            with pytest.raises(KeyError):
                self._run_async(evaluator._do_eval({"query": "q", "response": "r"}))
        elif result_key in self._DO_EVAL_MISSING_SCORE_RETURNS_DICT:
            res = self._run_async(evaluator._do_eval({"query": "q", "response": "r"}))
            assert isinstance(res, dict)
        else:
            with pytest.raises(EvaluationException):
                self._run_async(evaluator._do_eval({"query": "q", "response": "r"}))

    def test_the_super_do_eval_missing_inputs_raises(self):
        """Raise when required inputs are missing."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        with pytest.raises(EvaluationException):
            self._run_async(do_eval({}))

    def test_the_super_do_eval_dict_output(self):
        """Parse a dict llm_output from the prompty flow."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        result_key = evaluator._result_key
        evaluator._flow = MagicMock(side_effect=create_flow_side_effect(4, OutputType.DICT))
        res = self._run_async(do_eval({"query": "q", "response": "r"}))
        assert res[f"{result_key}_score"] == 4
        # List inputs exercise the message-preprocessing branch before the flow call.
        res_list = self._run_async(
            do_eval(
                {
                    "query": [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
                    "response": [{"role": "assistant", "content": [{"type": "text", "text": "r"}]}],
                }
            )
        )
        assert res_list[f"{result_key}_score"] == 4

    def test_the_super_do_eval_json_string_output(self):
        """Parse a JSON-string llm_output from the prompty flow."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        result_key = evaluator._result_key

        async def flow(timeout=None, **kwargs):
            return {"llm_output": json.dumps({"status": "completed", "score": 3, "reason": "ok"})}

        evaluator._flow = MagicMock(side_effect=flow)
        res = self._run_async(do_eval({"query": "q", "response": "r"}))
        assert res[f"{result_key}_score"] == 3

    def test_the_super_do_eval_plain_string_output(self):
        """Parse a free-text llm_output for a reason-based evaluator."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        result_key = evaluator._result_key

        async def flow(timeout=None, **kwargs):
            return {"llm_output": "The score is 4 because the response is coherent."}

        evaluator._flow = MagicMock(side_effect=flow)
        res = self._run_async(do_eval({"query": "q", "response": "r"}))
        assert f"{result_key}_score" in res

    def test_the_super_do_eval_regex_score_branch(self):
        """Cover the regex single-digit branch for a non-reason evaluator."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        # Force the non-reason-evaluator string path (regex single-digit extraction).
        evaluator._result_key = "zzz_unit_not_a_reason_evaluator"
        result_key = evaluator._result_key

        async def flow(timeout=None, **kwargs):
            return {"llm_output": "score 4 overall"}

        evaluator._flow = MagicMock(side_effect=flow)
        res = self._run_async(do_eval({"query": "q", "response": "r"}))
        assert res[f"{result_key}_score"] == 4

    def test_the_super_do_eval_empty_output_raises(self):
        """Raise when the prompty flow returns no output."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")

        async def flow(timeout=None, **kwargs):
            return None

        evaluator._flow = MagicMock(side_effect=flow)
        with pytest.raises(EvaluationException):
            self._run_async(do_eval({"query": "q", "response": "r"}))

    def test_the_super_real_call_convert_raises(self):
        """Propagate exceptions raised while converting kwargs to eval input."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        evaluator._convert_kwargs_to_eval_input = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            self._run_async(real_call(query="q", response="r"))

    def test_the_super_real_call_aggregates_and_fills_threshold(self):
        """Aggregate per-turn results and fill in result/threshold keys."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        result_key = evaluator._result_key
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 5}, {"x": 2}])

        async def fake_do_eval(eval_input):
            return {f"{result_key}_score": eval_input["x"]}

        evaluator._do_eval = fake_do_eval
        res = self._run_async(real_call(query="q", response="r"))
        assert isinstance(res, dict)

    def test_the_super_real_call_lower_is_better_fallback(self):
        """Cover the lower-is-better fallback when filling result keys."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        result_key = evaluator._result_key
        evaluator._higher_is_better = False
        evaluator._threshold = 3
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 2}, {"x": 5}])

        async def fake_do_eval(eval_input):
            return {f"{result_key}_score": eval_input["x"]}

        evaluator._do_eval = fake_do_eval
        res = self._run_async(real_call(query="q", response="r"))
        assert isinstance(res, dict)

    def test_the_super_real_call_higher_is_better_fills_result(self):
        """Cover the higher-is-better pass and fail branches when filling result keys."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        result_key = evaluator._result_key
        evaluator._higher_is_better = True
        evaluator._threshold = 3
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 5}, {"x": 1}])

        async def fake_do_eval(eval_input):
            return {f"{result_key}_score": eval_input["x"]}

        evaluator._do_eval = fake_do_eval
        res = self._run_async(real_call(query="q", response="r"))
        assert isinstance(res, dict)

    def test_the_super_real_call_invalid_threshold_is_swallowed(self):
        """Swallow the invalid-threshold exception while aggregating."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        result_key = evaluator._result_key
        evaluator._threshold = "not a number"
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 4}])

        async def fake_do_eval(eval_input):
            return {f"{result_key}_score": eval_input["x"]}

        evaluator._do_eval = fake_do_eval
        res = self._run_async(real_call(query="q", response="r"))
        assert res[f"{result_key}_score"] == 4

    def test_the_super_real_call_single_result(self):
        """Return a single fully-populated result unchanged."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        result_key = evaluator._result_key
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"x": 5}])

        async def fake_do_eval(eval_input):
            return {
                f"{result_key}_score": eval_input["x"],
                f"{result_key}_result": "pass",
                f"{result_key}_threshold": evaluator._threshold,
            }

        evaluator._do_eval = fake_do_eval
        res = self._run_async(real_call(query="q", response="r"))
        assert res[f"{result_key}_score"] == 5

    def test_the_super_real_call_empty_result(self):
        """Return an empty dict when there are no eval inputs."""
        evaluator = self._make_evaluator()
        real_call = self._super_impl(evaluator, "real_call")
        if real_call is None:
            pytest.skip("no _real_call on super")
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[])

        async def fake_do_eval(eval_input):
            return {}

        evaluator._do_eval = fake_do_eval
        assert self._run_async(real_call(query="q", response="r")) == {}


class SuperDoEvalNotApplicableUnitTests(_ValidatorUnitTestSupport):
    """Base ``_do_eval`` not-applicable paths.

    For evaluators whose ``_return_not_applicable_result`` matches the base signature.
    """

    def test_the_super_do_eval_intermediate_response(self):
        """Return not-applicable for an intermediate (tool-call) response."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        if evaluator._result_key in self._SUPER_DO_EVAL_NOT_APPLICABLE_UNSUPPORTED:
            pytest.skip("_return_not_applicable_result signature is incompatible with the base _do_eval")
        result_key = evaluator._result_key
        intermediate = [{"role": "assistant", "content": [{"type": "function_call", "name": "f"}]}]
        res = self._run_async(do_eval({"query": "q", "response": intermediate}))
        assert res[f"{result_key}_result"] == "not_applicable"

    def test_the_super_do_eval_skipped_output(self):
        """Map a skipped llm_output to a not-applicable result."""
        evaluator = self._make_evaluator()
        do_eval = self._super_impl(evaluator, "do_eval")
        if do_eval is None:
            pytest.skip("no _do_eval on super")
        if evaluator._result_key in self._SUPER_DO_EVAL_NOT_APPLICABLE_UNSUPPORTED:
            pytest.skip("_return_not_applicable_result signature is incompatible with the base _do_eval")
        result_key = evaluator._result_key

        async def flow(timeout=None, **kwargs):
            return {"llm_output": {"status": "skipped", "reason": "nope"}}

        evaluator._flow = MagicMock(side_effect=flow)
        res = self._run_async(do_eval({"query": "q", "response": "r"}))
        assert res[f"{result_key}_result"] == "not_applicable"


class MessagePreprocessUnitTests(_ValidatorUnitTestSupport):
    """Tests for the message-preprocessing utils (drop-mcp, normalize, preprocess)."""

    def test_drop_mcp_approval_messages(self):
        """Cover removal of MCP approval request/response messages."""
        fn = self._module_fn("_drop_mcp_approval_messages")
        msgs = [
            {"role": "assistant", "content": [{"type": "mcp_approval_request"}]},
            {"role": "tool", "content": [{"type": "mcp_approval_response"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        assert fn(msgs) == [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
        assert fn("x") == "x"

    def test_normalize_function_call_types(self):
        """Cover normalization of function/openapi call types and non-dict items."""
        fn = self._module_fn("_normalize_function_call_types")
        msgs = [
            {"role": "assistant", "content": [{"type": "function_call", "name": "f"}, "non-dict-item"]},
            {"role": "tool", "content": [{"type": "function_call_output", "function_call_output": {"x": 1}}]},
            {"role": "assistant", "content": [{"type": "openapi_call", "name": "g"}]},
            {"role": "tool", "content": [{"type": "openapi_call_output", "openapi_call_output": {"y": 2}}]},
            "non-dict-message",
            {"role": "user", "content": "string-content-not-a-list"},
        ]
        out = fn(msgs)
        assert out[0]["content"][0]["type"] == "tool_call"
        assert out[1]["content"][0]["type"] == "tool_result"
        assert out[1]["content"][0]["tool_result"] == {"x": 1}
        assert out[2]["content"][0]["type"] == "tool_call"
        assert out[3]["content"][0]["type"] == "tool_result"
        assert out[3]["content"][0]["tool_result"] == {"y": 2}
        # Non-dict messages and messages whose content is not a list are skipped unchanged.
        assert out[4] == "non-dict-message"
        assert out[5]["content"] == "string-content-not-a-list"
        assert fn("x") == "x"

    def test_preprocess_messages(self):
        """Cover the combined preprocessing pipeline."""
        fn = self._module_fn("_preprocess_messages")
        msgs = [
            {"role": "assistant", "content": [{"type": "mcp_approval_request"}]},
            {"role": "assistant", "content": [{"type": "function_call", "name": "f"}]},
        ]
        assert fn(msgs) == [{"role": "assistant", "content": [{"type": "tool_call", "name": "f"}]}]


class ConversationValidatorUnitTests(_ValidatorUnitTestSupport):
    """Corner-case tests for the ``ConversationValidator`` field/content/message/input helpers."""

    def test_validate_field_exists(self):
        """Cover required-field presence checks."""
        validator = self._validator()
        assert validator._validate_field_exists({"a": 1}, "a", "ctx") is None
        self._assert_exc(validator._validate_field_exists({}, "a", "ctx"), ErrorCategory.INVALID_VALUE)

    def test_validate_string_field(self):
        """Cover string field type/presence checks."""
        validator = self._validator()
        assert validator._validate_string_field({"a": "x"}, "a", "ctx") is None
        self._assert_exc(validator._validate_string_field({}, "a", "ctx"), ErrorCategory.INVALID_VALUE)
        self._assert_exc(validator._validate_string_field({"a": 1}, "a", "ctx"), ErrorCategory.INVALID_VALUE)

    def test_validate_list_field(self):
        """Cover list field type/presence checks."""
        validator = self._validator()
        assert validator._validate_list_field({"a": []}, "a", "ctx") is None
        self._assert_exc(validator._validate_list_field({}, "a", "ctx"), ErrorCategory.INVALID_VALUE)
        self._assert_exc(validator._validate_list_field({"a": 1}, "a", "ctx"), ErrorCategory.INVALID_VALUE)

    def test_validate_dict_field(self):
        """Cover dict field type/presence checks."""
        validator = self._validator()
        assert validator._validate_dict_field({"a": {}}, "a", "ctx") is None
        self._assert_exc(validator._validate_dict_field({}, "a", "ctx"), ErrorCategory.INVALID_VALUE)
        self._assert_exc(validator._validate_dict_field({"a": 1}, "a", "ctx"), ErrorCategory.INVALID_VALUE)

    def test_validate_text_content_item(self):
        """Cover text content-item validation."""
        validator = self._validator()
        assert validator._validate_text_content_item({"text": "hi"}, "user") is None
        self._assert_exc(validator._validate_text_content_item({}, "user"))
        self._assert_exc(validator._validate_text_content_item({"text": 1}, "user"))

    def test_validate_tool_call_content_item(self):
        """Cover tool_call content-item validation paths."""
        validator = self._validator()
        # Missing / invalid type.
        self._assert_exc(validator._validate_tool_call_content_item({}))
        self._assert_exc(validator._validate_tool_call_content_item({"type": "text"}))
        # mcp_approval_request short-circuits to valid.
        assert validator._validate_tool_call_content_item({"type": "mcp_approval_request"}) is None
        # Fully valid tool_call.
        valid = {"type": "tool_call", "name": "f", "arguments": {}, "tool_call_id": "c1"}
        assert validator._validate_tool_call_content_item(valid) is None
        # Missing name / non-dict arguments / missing tool_call_id.
        self._assert_exc(
            validator._validate_tool_call_content_item({"type": "tool_call", "arguments": {}, "tool_call_id": "c1"})
        )
        self._assert_exc(
            validator._validate_tool_call_content_item(
                {"type": "tool_call", "name": "f", "arguments": 1, "tool_call_id": "c1"}
            )
        )
        self._assert_exc(
            validator._validate_tool_call_content_item({"type": "tool_call", "name": "f", "arguments": {}})
        )

    def test_validate_user_or_system_message(self):
        """Cover user/system message content validation."""
        validator = self._validator()
        assert validator._validate_user_or_system_message({"content": "hi"}, "user") is None
        assert validator._validate_user_or_system_message(
            {"content": [{"type": "text", "text": "hi"}]}, "user"
        ) is None
        assert validator._validate_user_or_system_message(
            {"content": [{"type": "input_text", "text": "hi"}]}, "user"
        ) is None
        self._assert_exc(
            validator._validate_user_or_system_message({"content": [{"type": "tool_call", "text": "x"}]}, "user")
        )
        self._assert_exc(
            validator._validate_user_or_system_message({"content": [{"type": "text", "text": 1}]}, "user")
        )

    def test_validate_assistant_message(self):
        """Cover assistant message content validation."""
        validator = self._validator()
        assert validator._validate_assistant_message({"content": "hi"}) is None
        assert validator._validate_assistant_message({"content": [{"type": "output_text", "text": "ok"}]}) is None
        assert validator._validate_assistant_message(
            {"content": [{"type": "tool_call", "name": "f", "arguments": {}, "tool_call_id": "c1"}]}
        ) is None
        self._assert_exc(validator._validate_assistant_message({"content": [{"type": "input_text", "text": "x"}]}))

    def test_validate_tool_message(self):
        """Cover tool message validation paths."""
        validator = self._validator()
        self._assert_exc(validator._validate_tool_message({"content": "x", "tool_call_id": "c1"}))
        self._assert_exc(validator._validate_tool_message({"content": [{"type": "tool_result", "tool_result": {}}]}))
        self._assert_exc(validator._validate_tool_message({"tool_call_id": "c1", "content": [{"type": "text"}]}))
        self._assert_exc(
            validator._validate_tool_message({"tool_call_id": "c1", "content": [{"type": "tool_result"}]})
        )
        valid = {"tool_call_id": "c1", "content": [{"type": "tool_result", "tool_result": {"a": 1}}]}
        assert validator._validate_tool_message(valid) is None

    def test_validate_message_dict(self):
        """Cover message-dict validation across roles."""
        validator = self._validator()
        self._assert_exc(validator._validate_message_dict({"content": "x"}))
        self._assert_exc(validator._validate_message_dict({"role": "user"}))
        self._assert_exc(validator._validate_message_dict({"role": "user", "content": 1}))
        self._assert_exc(validator._validate_message_dict({"role": "user", "content": ""}))
        self._assert_exc(validator._validate_message_dict({"role": "user", "content": [{"text": "x"}]}))
        assert validator._validate_message_dict({"role": "user", "content": "hi"}) is None
        assert validator._validate_message_dict(
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
        ) is None
        assert validator._validate_message_dict(
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "tool_result", "tool_result": {}}]}
        ) is None

    def test_validate_input_messages_list(self):
        """Cover input-messages-list validation paths."""
        validator = self._validator()
        self._assert_exc(validator._validate_input_messages_list(None, "Query"), ErrorCategory.MISSING_FIELD)
        self._assert_exc(validator._validate_input_messages_list("", "Query"), ErrorCategory.MISSING_FIELD)
        assert validator._validate_input_messages_list("hi", "Query") is None
        self._assert_exc(validator._validate_input_messages_list(1, "Query"), ErrorCategory.INVALID_VALUE)
        self._assert_exc(validator._validate_input_messages_list([], "Query"), ErrorCategory.MISSING_FIELD)
        self._assert_exc(validator._validate_input_messages_list([1], "Query"), ErrorCategory.INVALID_VALUE)
        self._assert_exc(validator._validate_input_messages_list([{"role": "user"}], "Query"))
        assert validator._validate_input_messages_list([{"role": "user", "content": "hi"}], "Query") is None

    def test_validate_conversation(self):
        """Cover conversation-dict validation paths."""
        validator = self._validator()
        self._assert_exc(validator._validate_conversation([]), ErrorCategory.INVALID_VALUE)
        self._assert_exc(validator._validate_conversation({}))
        assert validator._validate_conversation({"messages": [{"role": "user", "content": "hi"}]}) is None

    def test_validate_query_and_response(self):
        """Cover query/response validation, required and optional."""
        validator = self._validator(requires_query=True)
        self._assert_exc(validator._validate_query(None))
        assert validator._validate_query([{"role": "user", "content": "hi"}]) is None
        v2 = self._validator(requires_query=False)
        assert v2._validate_query(None) is None
        self._assert_exc(validator._validate_response(None))
        assert validator._validate_response([{"role": "assistant", "content": "ok"}]) is None

    def test_validate_eval_input_conversation_path(self):
        """Cover validate_eval_input via the conversation branch."""
        validator = self._validator()
        assert validator.validate_eval_input(
            {"conversation": {"messages": [{"role": "user", "content": "hi"}]}}
        ) is True
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"conversation": {"messages": [1]}})

    def test_validate_eval_input_query_response_path(self):
        """Cover validate_eval_input via the query/response branch."""
        validator = self._validator()
        assert validator.validate_eval_input(
            {
                "query": [{"role": "user", "content": "hi"}],
                "response": [{"role": "assistant", "content": "ok"}],
            }
        ) is True
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"query": None, "response": [{"role": "assistant", "content": "ok"}]})
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"query": [{"role": "user", "content": "hi"}], "response": None})


class ConversationValidatorToolCheckUnitTests(_ValidatorUnitTestSupport):
    """Test the ``ConversationValidator`` unsupported-tool rejection path (needs ``check_for_unsupported_tools``)."""

    def test_validate_assistant_message_unsupported_tools(self):
        """Cover the unsupported-tool rejection path."""
        validator = self._validator(check_for_unsupported_tools=True)
        bad = {"type": "tool_call", "name": "bing_grounding", "arguments": {}, "tool_call_id": "c1"}
        self._assert_exc(validator._validate_assistant_message({"content": [bad]}), ErrorCategory.NOT_APPLICABLE)
        openapi = {"type": "openapi_call", "name": "x", "arguments": {}, "tool_call_id": "c2"}
        self._assert_exc(validator._validate_assistant_message({"content": [openapi]}), ErrorCategory.NOT_APPLICABLE)


class ToolDefinitionsValidatorUnitTests(_ValidatorUnitTestSupport):
    """Branch tests for the tool-family ``ToolDefinitionsValidator``."""

    def test_validate_tool_definitions_error_branches(self):
        """Exercise the list/type/optional/openapi tool-definitions validation branches."""
        cls = self._tool_definitions_validator_cls()
        target = self._target()
        validator = cls(error_target=target)
        assert validator._validate_tool_definitions(None) is None
        assert validator._validate_tool_definitions("free-form-string") is None
        assert validator._validate_tool_definitions([{"name": "f", "parameters": {}}]) is None
        assert isinstance(validator._validate_tool_definitions(123), EvaluationException)
        assert isinstance(validator._validate_tool_definitions([123]), EvaluationException)
        openapi = [{"type": "openapi", "functions": [{"name": "f", "parameters": {}}]}]
        assert validator._validate_tool_definitions(openapi) is None
        # An openapi definition whose ``functions`` field is not a list surfaces the list-field error.
        assert isinstance(
            validator._validate_tool_definitions([{"type": "openapi", "functions": "not-a-list"}]),
            EvaluationException,
        )
        # An openapi definition whose nested function definition is invalid surfaces that error.
        assert isinstance(
            validator._validate_tool_definitions([{"type": "openapi", "functions": [{"parameters": {}}]}]),
            EvaluationException,
        )
        required = cls(error_target=target, optional_tool_definitions=False)
        assert isinstance(required._validate_tool_definitions(None), EvaluationException)

    def test_validate_tool_definition_field_branches(self):
        """Exercise the single tool-definition field validation branches."""
        cls = self._tool_definitions_validator_cls()
        validator = cls(error_target=self._target())
        assert validator._validate_tool_definition({"name": "f", "parameters": {}}) is None
        assert isinstance(validator._validate_tool_definition("not-a-dict"), EvaluationException)
        assert isinstance(validator._validate_tool_definition({"parameters": {}}), EvaluationException)
        assert isinstance(validator._validate_tool_definition({"name": "f"}), EvaluationException)

    def test_validate_eval_input_with_tool_definitions(self):
        """Cover the tool-definitions ``validate_eval_input`` override (happy path and raise)."""
        cls = self._tool_definitions_validator_cls()
        target = self._target()
        validator = cls(error_target=target)
        base_input = {
            "query": [{"role": "user", "content": "hi"}],
            "response": [{"role": "assistant", "content": "ok"}],
        }
        assert validator.validate_eval_input(
            {**base_input, "tool_definitions": [{"name": "f", "parameters": {}}]}
        ) is True
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({**base_input, "tool_definitions": 123})


class ConversationSerializationUnitTests(_ValidatorUnitTestSupport):
    """Tests for the conversation serialization, evaluation-level and result-building utils."""

    def test_merge_query_response_messages(self):
        """Cover merging of query and response message lists."""
        fn = self._module_fn("_merge_query_response_messages")
        assert fn([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]

    def test_split_messages_at_latest_user(self):
        """Cover splitting messages at the latest user turn."""
        fn = self._module_fn("_split_messages_at_latest_user")
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        query, response = fn(msgs)
        assert query == msgs[:3]
        assert response == msgs[3:]

    def test_wrap_string_messages(self):
        """Cover wrapping plain strings into message dicts."""
        fn = self._module_fn("_wrap_string_messages")
        query, response = fn("hello", "world")
        assert query == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert response == [{"role": "assistant", "content": [{"type": "text", "text": "world"}]}]

    def test_resolve_evaluation_level(self):
        """Cover evaluation-level resolution and invalid inputs."""
        fn = self._module_fn("_resolve_evaluation_level")
        target = self._target()
        level_enum = getattr(self._module(), "EvaluationLevel")
        assert fn(None, target) is None
        assert fn("", target) is None
        assert fn(level_enum.TURN, target) == level_enum.TURN
        assert fn("turn", target) == level_enum.TURN
        assert fn("conversation", target) == level_enum.CONVERSATION
        with pytest.raises(EvaluationException):
            fn("bogus", target)
        with pytest.raises(EvaluationException):
            fn(123, target)

    def test_serialize_messages(self):
        """Cover serialization of a mixed-role conversation."""
        fn = self._module_fn("serialize_messages")
        assert fn([]) == ""
        msgs = [
            "not a dict",
            {"foo": "bar"},
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": "Hello there"},
            {"role": "user", "content": "Another question"},
            {"role": "assistant", "content": [{"type": "text", "text": "Final answer"}]},
        ]
        out = fn(msgs)
        assert isinstance(out, str)
        assert "Final answer" in out

    def test_serialize_messages_system_list_content(self):
        """Cover serialization when the system message uses list content."""
        fn = self._module_fn("serialize_messages")
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "System prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "Question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
        ]
        out = fn(msgs)
        assert isinstance(out, str)
        assert "Answer" in out

    def test_serialize_messages_trailing_user(self):
        """Cover the trailing-user-query flush branch of serialization."""
        fn = self._module_fn("serialize_messages")
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "First question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "First answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "Trailing question"}]},
        ]
        out = fn(msgs)
        assert isinstance(out, str)
        assert "Trailing question" in out

    def test_should_use_conversation_level(self):
        """Cover the conversation-vs-turn level decision logic."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_should_use_conversation_level"):
            pytest.skip("no _should_use_conversation_level")
        level_enum = getattr(self._module(), "EvaluationLevel", None)
        if level_enum is None:
            pytest.skip("no EvaluationLevel")
        evaluator._evaluation_level = level_enum.CONVERSATION
        assert evaluator._should_use_conversation_level({}) is True
        evaluator._evaluation_level = level_enum.TURN
        assert evaluator._should_use_conversation_level({"messages": [1]}) is False
        evaluator._evaluation_level = None
        assert evaluator._should_use_conversation_level({"messages": [1]}) is True
        assert evaluator._should_use_conversation_level({}) is False

    def test_build_result(self):
        """Cover the result-dict builder for populated and empty scores."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_build_result"):
            pytest.skip("no _build_result")
        result_key = evaluator._result_key
        # quality_grader has a bespoke keyword-only _build_result signature
        # (passed/failure_reasons/stage1_parsed/...); see EVALUATOR_DISCREPANCIES.md.
        if result_key == "quality_grader":
            pytest.skip("quality_grader uses a bespoke _build_result signature")
        # task_adherence is the only builder that omits the `status` kwarg.
        has_status = result_key != "task_adherence"
        kwargs = dict(
            score=4, result="pass", reason="good", properties={"a": 1}, prompty_output_dict={"input_token_count": 2}
        )
        if has_status:
            kwargs["status"] = "completed"
        res = evaluator._build_result(**kwargs)
        assert res[result_key] == 4
        assert res[f"{result_key}_result"] == "pass"
        if has_status:
            assert res[f"{result_key}_status"] == "completed"
        assert res[f"{result_key}_properties"]["a"] == 1
        assert res[f"{result_key}_prompt_tokens"] == 2
        empty = dict(score=None, result="error", reason="bad", properties={}, prompty_output_dict={})
        if has_status:
            empty["status"] = "error"
        res2 = evaluator._build_result(**empty)
        assert res2[result_key] is None

    def test_parse_prompty_output(self):
        """Cover parsing of completed, skipped and malformed prompty outputs."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_parse_prompty_output"):
            pytest.skip("no _parse_prompty_output")
        result_key = evaluator._result_key
        res = evaluator._parse_prompty_output(
            {"llm_output": {"status": "completed", "score": 4, "reason": "r", "properties": {}}}
        )
        assert res[f"{result_key}_result"] in ("pass", "fail")
        assert isinstance(res[result_key], (int, float)) and not isinstance(res[result_key], bool)
        # A skipped prompty status normally maps to "not_applicable"; see
        # EVALUATOR_DISCREPANCIES.md for the evaluators that diverge.
        res2 = evaluator._parse_prompty_output({"llm_output": {"status": "skipped", "reason": "skip"}})
        assert res2[f"{result_key}_result"] == self._PARSE_SKIPPED_RESULT.get(result_key, "not_applicable")
        # Malformed llm_outputs are handled deterministically but inconsistently
        # per evaluator (recorded in _PARSE_MALFORMED_EXPECTATIONS); the invariant
        # is no unhandled TypeError from math.isnan.
        expectations = self._PARSE_MALFORMED_EXPECTATIONS.get(result_key)
        if expectations is None:
            pytest.skip(f"no recorded _parse_prompty_output expectations for {result_key}")
        malformed = (
            {"llm_output": {"status": "completed", "reason": "r"}},
            {"llm_output": {"status": "completed", "score": "abc"}},
            {"llm_output": "not a dict"},
        )
        for inp, expected in zip(malformed, expectations):
            if isinstance(expected, type) and issubclass(expected, BaseException):
                with pytest.raises(expected):
                    evaluator._parse_prompty_output(inp)
            else:
                out = evaluator._parse_prompty_output(inp)
                assert out[f"{result_key}_status"] == expected

    def test_real_call_turn_level_splits_messages(self):
        """Split conversation messages into query/response at turn level."""
        evaluator = self._make_evaluator()
        if not hasattr(evaluator, "_real_call"):
            pytest.skip("no _real_call")
        level_enum = getattr(self._module(), "EvaluationLevel", None)
        if level_enum is None:
            pytest.skip("no EvaluationLevel")
        evaluator._evaluation_level = level_enum.TURN
        captured = {}

        async def fake_super(**kwargs):
            captured.update(kwargs)
            return {}

        evaluator._the_super_real_call = fake_super
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
        ]
        self._run_async(evaluator._real_call(messages=msgs))
        assert "query" in captured
        assert "response" in captured
        assert "messages" not in captured


class AgentResponseReformatUnitTests(_ValidatorUnitTestSupport):
    """Tests for the tool-family ``reformat_agent_response`` helper."""

    def test_reformat_agent_response_with_tool_messages(self):
        """Format agent text, flat/nested tool calls, and tool results into a string."""
        module = self._tool_reformat_module()
        reformat = module.reformat_agent_response
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "fetch_weather",
                        "arguments": {"city": "Seattle", "units": None, "days": 3},
                    },
                    {
                        "type": "tool_call",
                        "tool_call": {"id": "c2", "function": {"name": "send_email", "arguments": {"to": "a@b.com"}}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "tool_result", "tool_result": "sunny"}]},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "tool_result", "tool_result": "sent"}]},
        ]
        out = reformat(msgs, include_tool_messages=True)
        assert "TOOL_CALL" in out and "fetch_weather" in out and "send_email" in out
        assert "TOOL_RESULT" in out

    def test_reformat_agent_response_empty_and_fallback(self):
        """Return empty string for None/[] and fall back to raw input when unparseable."""
        module = self._tool_reformat_module()
        reformat = module.reformat_agent_response
        assert reformat(None) == ""
        assert reformat([]) == ""
        # A response with no agent text extracts nothing; passing a logger exercises the
        # empty-extraction debug-log branch before falling back to the raw input.
        weird = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert reformat(weird, logger=logging.getLogger("agent_reformat_test")) == weird
        # A non-dict message item makes parsing raise; the except-branch logs a safe
        # summary and falls back to the raw input unchanged.
        malformed = [123]
        assert reformat(malformed, logger=logging.getLogger("agent_reformat_test")) == malformed


class LogSafeSummaryUnitTests(_ValidatorUnitTestSupport):
    """Test the tool-family ``_log_safe_summary`` helper."""

    def test_log_safe_summary_variants(self):
        """Summarize list, dict, and scalar payloads without leaking values."""
        module = self._module()
        fn = getattr(module, "_log_safe_summary", None)
        if fn is None:
            pytest.skip("no _log_safe_summary")
        assert "type=list" in fn([{"role": "user"}, "x"])
        assert "type=dict" in fn({"b": 1, "a": 2})
        assert "type=int" in fn(5)

        class _RaisesOnLen:
            def __len__(self):
                raise RuntimeError("boom")

        # The except-branch returns a safe placeholder instead of leaking or raising.
        assert "summary unavailable" in fn(_RaisesOnLen())


class MessagesOrQueryResponseUnitTests(_ValidatorUnitTestSupport):
    """Corner-case tests for ``MessagesOrQueryResponseInputValidator``."""

    def test_mqr_messages_not_list(self):
        """Reject a non-list ``messages`` input."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": "x"})

    def test_mqr_messages_empty(self):
        """Reject an empty ``messages`` list."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": []})

    def test_mqr_message_not_dict(self):
        """Reject a non-dict message item."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": [1]})

    def test_mqr_message_missing_role(self):
        """Reject a message item missing the role."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": [{"content": "x"}]})

    def test_mqr_invalid_role(self):
        """Reject an unknown message role."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": [{"role": "bogus", "content": "x"}]})

    def test_mqr_missing_user(self):
        """Reject input that has no user message."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": [{"role": "assistant", "content": "x"}]})

    def test_mqr_missing_assistant(self):
        """Reject input that has no assistant message."""
        validator = self._messages_or_query_response_validator()
        with pytest.raises(EvaluationException):
            validator.validate_eval_input({"messages": [{"role": "user", "content": "x"}]})

    def test_mqr_last_message_no_text(self):
        """Accept input whose last message has no text content.

        The mid-execution text guard (which rejected a final assistant message
        lacking text content, e.g. a trailing tool_call) has been removed, so
        such mid-execution inputs are now valid.
        """
        validator = self._messages_or_query_response_validator()
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "f", "arguments": {}, "tool_call_id": "c1"}],
            },
        ]
        assert validator.validate_eval_input({"messages": msgs}) is True

    def test_mqr_valid_messages(self):
        """Accept a well-formed messages list."""
        validator = self._messages_or_query_response_validator()
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        assert validator.validate_eval_input({"messages": msgs}) is True

    def test_mqr_delegates_to_super_without_messages(self):
        """Delegate to the base validator when no messages key is present."""
        validator = self._messages_or_query_response_validator()
        assert validator.validate_eval_input(
            {
                "query": [{"role": "user", "content": "hi"}],
                "response": [{"role": "assistant", "content": "ok"}],
            }
        ) is True


class ToolDefinitionExtractionUnitTests(_ValidatorUnitTestSupport):
    """Tests for the inlined ``_extract_needed_tool_definitions`` util."""

    def test_extract_needed_tool_definitions_happy_path(self):
        """Return the matching definition for a well-formed converter tool call."""
        extract = self._extract_tool_defs_callable()
        if extract is None:
            pytest.skip("no _extract_needed_tool_definitions")
        needed = extract(
            [{"type": "tool_call", "name": "fetch_weather", "arguments": {}}],
            [{"name": "fetch_weather", "parameters": {}}],
        )
        assert any(t.get("name") == "fetch_weather" for t in needed)

    def test_extract_needed_tool_definitions_missing_definition_raises(self):
        """Raise when a converter tool call has no matching definition."""
        extract = self._extract_tool_defs_callable()
        if extract is None:
            pytest.skip("no _extract_needed_tool_definitions")
        with pytest.raises(EvaluationException):
            extract([{"type": "tool_call", "name": "unknown_tool", "arguments": {}}], [])

    def test_extract_needed_tool_definitions_missing_name_raises(self):
        """Raise when a converter tool call is missing its name."""
        extract = self._extract_tool_defs_callable()
        if extract is None:
            pytest.skip("no _extract_needed_tool_definitions")
        with pytest.raises(EvaluationException):
            extract([{"type": "tool_call"}], [])

    def test_extract_needed_tool_definitions_unsupported_format_raises(self):
        """Raise for non-converter tool-call formats."""
        extract = self._extract_tool_defs_callable()
        if extract is None:
            pytest.skip("no _extract_needed_tool_definitions")
        with pytest.raises(EvaluationException):
            extract([{"type": "function", "name": "x"}], [])

    def test_extract_needed_tool_definitions_non_dict_raises(self):
        """Raise when a tool call is not a dictionary."""
        extract = self._extract_tool_defs_callable()
        if extract is None:
            pytest.skip("no _extract_needed_tool_definitions")
        with pytest.raises(EvaluationException):
            extract(["not-a-dict"], [])


class ConversationHistoryReformatUnitTests(_ValidatorUnitTestSupport):
    """Tests for the tool-family ``reformat_conversation_history`` (``include_tool_calls`` variant)."""

    def test_reformat_conversation_history_with_tool_calls(self):
        """Format a multi-turn conversation with system, tool calls, and results."""
        module = self._module()
        reformat = getattr(module, "reformat_conversation_history", None)
        if getattr(module, "_get_conversation_history", None) is None or reformat is None:
            pytest.skip("no inlined conversation-history helpers")
        if "include_tool_calls" not in inspect.signature(reformat).parameters:
            pytest.skip("reformat_conversation_history variant without include_tool_calls")
        query = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Weather?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "fetch_weather",
                        "arguments": {"city": "Seattle", "units": None},
                    },
                    {
                        "type": "tool_call",
                        "tool_call": {"id": "c2", "function": {"name": "send_email", "arguments": {"to": "a@b.com"}}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "tool_result", "tool_result": "sunny"}]},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "tool_result", "tool_result": "sent"}]},
            {"role": "user", "content": [{"type": "text", "text": "Thanks"}]},
        ]
        out = reformat(query, include_system_messages=True, include_tool_calls=True)
        assert "SYSTEM_PROMPT" in out
        assert "User turn 1" in out and "Agent turn 1" in out
        assert "fetch_weather" in out and "send_email" in out

    def test_reformat_conversation_history_malformed_fallback(self):
        """Fall back to the raw input (and log a safe summary) when malformed."""
        module = self._module()
        reformat = getattr(module, "reformat_conversation_history", None)
        if getattr(module, "_get_conversation_history", None) is None or reformat is None:
            pytest.skip("no inlined conversation-history helpers")
        if "include_tool_calls" not in inspect.signature(reformat).parameters:
            pytest.skip("reformat_conversation_history variant without include_tool_calls")
        query = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        out = reformat(query, logger=logging.getLogger("tool_reformat_test"))
        assert out == query
        # A role-less message is skipped; with no balanced turns the helper falls back
        # to the raw query unchanged (covers the role-less continue branch).
        role_less = [{"content": "no role"}]
        assert reformat(role_less, logger=logging.getLogger("tool_reformat_test")) == role_less


class SimplifyMessagesUnitTests(_ValidatorUnitTestSupport):
    """Test the groundedness ``simplify_messages`` util."""

    def test_simplify_messages_variants(self):
        """Simplify messages across passthrough, drop-system, and error-fallback branches."""
        fn = getattr(self._module(), "simplify_messages", None)
        if fn is None:
            pytest.skip("no simplify_messages")
        assert fn("already a string") == "already a string"
        assert fn(123) == 123
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_result": "r"}]},
            "not-a-dict-msg",
        ]
        out = fn(messages)
        assert "system" not in [m.get("role") for m in out if isinstance(m, dict)]
        out_drop = fn(messages, drop_tool_calls=True)
        assert all(not (isinstance(m, dict) and m.get("role") == "tool") for m in out_drop)
        bad = [{"role": "user", "content": 123}]
        assert fn(bad, logger=logging.getLogger("simplify_test")) == bad


class ToolResponseEvalUnitTests(_ValidatorUnitTestSupport):
    """Shared convert-kwargs / do_eval / real_call error-path tests for response-based tool evaluators.

    ``tool_input_accuracy`` and ``tool_selection`` expose byte-identical
    ``_convert_kwargs_to_eval_input`` error branches, the same ``_do_eval``
    missing-input / invalid-output raises, and the same ``_real_call``
    error->not-applicable path. Composing this mixin runs them against each
    evaluator's own inlined copies (resolved via ``evaluator_class``); the
    per-evaluator ``_result_key`` drives the not-applicable assertion. None of
    these paths reach the judge flow (they short-circuit on input validation or
    override ``_flow``), so the unmocked evaluator built by ``_make_evaluator`` is
    sufficient.
    """

    _TOOL_CALL_RESPONSE = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_call", "tool_call_id": "c1", "name": "search", "arguments": {"q": "x"}}
            ],
        }
    ]
    _TEXT_RESPONSE = [{"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}]

    def test_convert_kwargs_no_tool_calls_in_response(self):
        """Return the no-tool-calls error when the response has no tool calls."""
        evaluator = self._make_evaluator()
        result = evaluator._convert_kwargs_to_eval_input(query="q", response=self._TEXT_RESPONSE)
        assert result["error_message"] == evaluator._NO_TOOL_CALLS_MESSAGE

    def test_convert_kwargs_wraps_dict_definitions(self):
        """Wrap dict-shaped tool definitions when the response is a string."""
        evaluator = self._make_evaluator()
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            response="just a string response",
            tool_definitions={"name": "x", "type": "function", "parameters": {}},
        )
        assert isinstance(result, dict)

    def test_convert_kwargs_string_response_empty_definitions(self):
        """Return the no-definitions error when a string response has no definitions."""
        evaluator = self._make_evaluator()
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", response="just a string response", tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_convert_kwargs_extraction_fails_no_definitions(self):
        """Return the no-definitions error when extraction fails with an empty list."""
        evaluator = self._make_evaluator()
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", response=[dict(m) for m in self._TOOL_CALL_RESPONSE], tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_convert_kwargs_extraction_fails_unmatched_definitions(self):
        """Return the missing-definitions error when a used tool has no definition."""
        evaluator = self._make_evaluator()
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            response=[dict(m) for m in self._TOOL_CALL_RESPONSE],
            tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
        )
        assert result["error_message"] == evaluator._TOOL_DEFINITIONS_MISSING_MESSAGE

    def test_do_eval_missing_query_raises(self):
        """Raise when ``_do_eval`` is invoked without the required inputs."""
        evaluator = self._make_evaluator()
        with pytest.raises(EvaluationException):
            self._run_async(evaluator._do_eval({}))

    def test_do_eval_invalid_output_raises(self):
        """Raise when the judge returns a non-dict ``llm_output`` payload."""
        evaluator = self._make_evaluator()

        async def _bad_flow(**kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = _bad_flow
        with pytest.raises(EvaluationException):
            self._run_async(evaluator._do_eval({"query": "q"}))

    def test_real_call_error_message_returns_not_applicable(self):
        """Return a not-applicable result when input conversion yields an error message."""
        evaluator = self._make_evaluator()
        result = self._run_async(
            evaluator._real_call(
                query="q",
                response=[dict(m) for m in self._TEXT_RESPONSE],
                tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
            )
        )
        assert result[evaluator._result_key] is None
