# Evaluator behaviour discrepancies

The prompty-based evaluators under `assets/evaluators/builtin/**/_<name>.py` share a
large amount of copy-pasted helper code (validators, `_build_result`,
`_parse_prompty_output`, `_return_not_applicable_result`, `_do_eval`, ...). Those
copies have drifted apart, so the shared unit-test mixin
[`base_validator_unit_test.py`](base_validator_unit_test.py) currently has to branch
per evaluator to pin the divergent behaviour.

This document catalogues those divergences so they can be harmonised. Once the
source is consistent, the per-evaluator maps in the mixin
(`_PARSE_SKIPPED_RESULT`, `_PARSE_MALFORMED_EXPECTATIONS`,
`_DO_EVAL_MISSING_SCORE_KEYERROR`, `_DO_EVAL_MISSING_SCORE_RETURNS_DICT`,
`_SUPER_DO_EVAL_NOT_APPLICABLE_UNSUPPORTED`) should collapse to a single expected
outcome and be removed.


## 1. `_return_not_applicable_result` signature

| Evaluator | Signature |
| --- | --- |
| `quality_grader` | `(self, error_message)` |
| all 15 others | `(self, error_message, threshold)` |

The base `PromptyEvaluatorBase._do_eval` calls
`self._return_not_applicable_result(skip_reason, self._threshold)` (two args), so
`quality_grader`'s one-arg override is **incompatible with the base `_do_eval`**
and raises `TypeError` if the base path is ever taken.

**Recommended convergence:** give `quality_grader._return_not_applicable_result`
the same `(error_message, threshold)` signature as every other evaluator (accept
and ignore `threshold` if unused).

## 2. `_build_result` signature

Only 6 evaluators define `_build_result`.

| Evaluator | Divergence from the common `(score, result, reason, status, properties, prompty_output_dict=None)` |
| --- | --- |
| `coherence`, `customer_satisfaction`, `task_completion` | none (canonical) |
| `groundedness` | param order differs; `status` is a trailing keyword with default `None` |
| `task_adherence` | **omits `status`**; adds keyword-only `threshold` |
| `quality_grader` | bespoke keyword-only signature: `(*, passed, failure_reasons, stage1_parsed, stage2_parsed, stage1_output, stage2_output, prompt_tokens, completion_tokens, total_tokens, model_id)` |

**Recommended convergence:** adopt one canonical `_build_result` signature across
all evaluators (`score, result, reason, status, properties, prompty_output_dict=None`)
and have `quality_grader` map its grader-specific fields onto that shape.

## 3. `_parse_prompty_output` — skipped status

For `{"llm_output": {"status": "skipped", ...}}`:

| Evaluator | `<key>_result` |
| --- | --- |
| `customer_satisfaction` | `"skipped"` |
| `coherence`, `groundedness`, `task_adherence`, `task_completion` | `"not_applicable"` |

**Recommended convergence:** `customer_satisfaction` should return
`"not_applicable"` like the others.

## 4. `_parse_prompty_output` — malformed outputs

Behaviour for three malformed `llm_output` shapes (only the 5 evaluators that
define `_parse_prompty_output` are shown). A cell value is the returned
`<key>_status`; `raise X` means it raises.

| Evaluator | `{status: completed, reason}` (no score) | `{status: completed, score: "abc"}` | `"not a dict"` |
| --- | --- | --- | --- |
| `coherence` | `error` | `error` | `error` |
| `groundedness` | `completed` | `error` | `error` |
| `customer_satisfaction` | `completed` | `raise TypeError` | `error` |
| `task_adherence` | `completed` | `raise ValueError` | `raise EvaluationException` |
| `task_completion` | `completed` | `raise ValueError` | `raise EvaluationException` |

**Recommended convergence:** all evaluators should handle malformed output the
same deterministic way — prefer `coherence`'s behaviour (return a result dict with
`<key>_status == "error"`) rather than raising `TypeError`/`ValueError`, and never
leave `status == "completed"` when the score is missing/invalid.

## 5. `_do_eval` — missing score (turn level)

With a turn-level flow returning `{"status": "completed", "reason": "no score field"}`
(i.e. no `score`), and calling `_do_eval({"query": ..., "response": ...})`:

| Outcome | Evaluators |
| --- | --- |
| `raise KeyError` (require extra input before scoring) | `groundedness` (`'context'`), `tool_output_utilization` (`'tool_definitions'`) |
| return a result dict (degrade, no raise) | `relevance`, `customer_satisfaction`, `task_adherence`, `task_completion`, `deflection_rate`, `quality_grader`, `tool_call_success` |
| `raise EvaluationException` | `coherence`, `fluency`, `retrieval`, `similarity`, `response_completeness`, `intent_resolution`, `tool_call_accuracy`, `tool_input_accuracy`, `tool_selection` |

The invariant that must always hold (and does today) is that a missing score is
never an unhandled `TypeError` from `math.isnan(None)`.

**Recommended convergence:** a missing/`NaN` score should consistently raise
`EvaluationException` with a clear message across all evaluators. The `KeyError`
cases (`groundedness`, `tool_output_utilization`) should validate their required
inputs and raise a typed `EvaluationException` instead of a bare `KeyError`.

## 6. `deflection_rate` ignores `status="skipped"`

Every other prompty evaluator short-circuits a skipped LLM output (`score=None`,
`status="skipped"`) into a not-applicable result. `deflection_rate._do_eval`
has **no** skipped-status check: it reads `llm_output.get("score", 0)` and, because
`None` is falsy, coerces the score to `0` — which then evaluates to `result="pass"`
(deflected). So a skipped/None score is silently reported as a passing deflection
rather than not-applicable.

The regression test `test_skipped_llm_status_coerces_to_pass` pins this current
behavior (see `test_deflection_rate_evaluator_behavior.py`).

**Recommended convergence:** `deflection_rate._do_eval` should honor
`status="skipped"` / a `None` score the same way the others do (return a
not-applicable result) instead of coercing to `0`/`pass`.

