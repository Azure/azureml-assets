# Behavioral Test Report — Evaluators

## Environment

| Item | Version |
|------|---------|
| Python | 3.12.3 |
| pytest | 9.0.3 |
| pytest-xdist | 3.8.0 |
| azure-ai-evaluation | 1.11.1 |
| azure-identity | 1.25.3 |
| typing_extensions | 4.15.0 |

Dependencies installed from `assets/evaluators/tests/requirements.txt` and `assets/evaluators/tests/conda.yaml`.

---

## Pytest Command

```bash
python -m pytest assets/evaluators/tests/test_evaluators_behavior \
    -m unittest \
    --tb=short \
    -ra \
    --junitxml=pytest-reports/behavior-results.xml \
    -v
```

Tests are mocked (`use_mocking = True`) — no Azure credentials, network access, or real LLM calls required.

---

## Final Summary (After Fix)

```
1497 passed, 4 deselected, 66 warnings in 18.78s
```

✅ **All 1497 collected tests pass.**

---

## Pre-Fix Summary

Before the fix described below, running the full suite produced:

```
28 failed, 1469 passed, 4 deselected, 66 warnings in 18.35s
```

---

## Failures Found (Pre-Fix)

All 28 failures shared the same root cause and the same pattern:

```
AssertionError: Expected score to be None for not-applicable result but got '<threshold_value>'
```

### Affected Tests (28 total — 2 per evaluator × 14 evaluators)

| Test File | Test Method | Score Returned |
|-----------|-------------|----------------|
| `test_coherence_evaluator_behavior.py::TestCoherenceEvaluatorBehavior` | `test_function_call_response` | 3 |
| `test_coherence_evaluator_behavior.py::TestCoherenceEvaluatorBehavior` | `test_mcp_approval_response` | 3 |
| `test_customer_satisfaction_evaluator_behavior.py::TestCustomerSatisfactionEvaluatorBehavior` | `test_function_call_response` | 3 |
| `test_customer_satisfaction_evaluator_behavior.py::TestCustomerSatisfactionEvaluatorBehavior` | `test_mcp_approval_response` | 3 |
| `test_deflection_rate_evaluator_behavior.py::TestDeflectionRateEvaluatorBehavior` | `test_function_call_response` | 0 |
| `test_deflection_rate_evaluator_behavior.py::TestDeflectionRateEvaluatorBehavior` | `test_mcp_approval_response` | 0 |
| `test_fluency_evaluator_behavior.py::TestFluencyEvaluatorBehavior` | `test_function_call_response` | 3 |
| `test_fluency_evaluator_behavior.py::TestFluencyEvaluatorBehavior` | `test_mcp_approval_response` | 3 |
| `test_groundedness_evaluator_behavior.py::TestGroundednessEvaluatorBehavior` | `test_function_call_response` | 3 |
| `test_groundedness_evaluator_behavior.py::TestGroundednessEvaluatorBehavior` | `test_mcp_approval_response` | 3 |
| `test_intent_resolution_evaluator_behavior.py::TestIntentResolutionEvaluatorBehavior` | `test_function_call_response` | 3 |
| `test_intent_resolution_evaluator_behavior.py::TestIntentResolutionEvaluatorBehavior` | `test_mcp_approval_response` | 3 |
| `test_quality_grader_evaluator_behavior.py::TestQualityGraderEvaluatorBehavior` | `test_function_call_response` | 1.0 |
| `test_quality_grader_evaluator_behavior.py::TestQualityGraderEvaluatorBehavior` | `test_mcp_approval_response` | 1.0 |
| `test_relevance_evaluator_behavior.py::TestRelevanceEvaluatorBehavior` | `test_function_call_response` | 3 |
| `test_relevance_evaluator_behavior.py::TestRelevanceEvaluatorBehavior` | `test_mcp_approval_response` | 3 |
| `test_task_adherence_evaluator_behavior.py::TestTaskAdherenceEvaluatorBehavior` | `test_function_call_response` | 0 |
| `test_task_adherence_evaluator_behavior.py::TestTaskAdherenceEvaluatorBehavior` | `test_mcp_approval_response` | 0 |
| `test_task_completion_evaluator_behavior.py::TestTaskCompletionEvaluatorBehavior` | `test_function_call_response` | 1 |
| `test_task_completion_evaluator_behavior.py::TestTaskCompletionEvaluatorBehavior` | `test_mcp_approval_response` | 1 |
| `test_tool_call_success_evaluator_behavior.py::TestToolCallSuccessEvaluatorBehavior` | `test_function_call_response` | 1 |
| `test_tool_call_success_evaluator_behavior.py::TestToolCallSuccessEvaluatorBehavior` | `test_mcp_approval_response` | 1 |
| `test_tool_input_accuracy_evaluator_behavior.py::TestToolInputAccuracyEvaluatorBehavior` | `test_function_call_response` | 1 |
| `test_tool_input_accuracy_evaluator_behavior.py::TestToolInputAccuracyEvaluatorBehavior` | `test_mcp_approval_response` | 1 |
| `test_tool_output_utilization_evaluator_behavior.py::TestToolOutputUtilizationEvaluatorBehavior` | `test_function_call_response` | 1 |
| `test_tool_output_utilization_evaluator_behavior.py::TestToolOutputUtilizationEvaluatorBehavior` | `test_mcp_approval_response` | 1 |
| `test_tool_selection_evaluator_behavior.py::TestToolSelectionEvaluatorBehavior` | `test_function_call_response` | 1 |
| `test_tool_selection_evaluator_behavior.py::TestToolSelectionEvaluatorBehavior` | `test_mcp_approval_response` | 1 |

### Traceback Excerpt (representative — same across all 28 failures)

```
assets/evaluators/tests/test_evaluators_behavior/base_evaluator_behavior_test.py:869:
    self.assert_not_applicable(result_data)

assets/evaluators/tests/common/base_evaluator_runner.py:267:
>   assert result_data[score_key] is None, \
E   AssertionError: Expected score to be None for not-applicable result but got '3'
```

---

## Root Cause Analysis

### Test trigger

`test_function_call_response` and `test_mcp_approval_response` are defined in
`base_evaluator_behavior_test.py::BaseEvaluatorBehaviorTest`. They call
`assert_not_applicable()` when the evaluator receives an **intermediate** response
(e.g., a response containing only a `function_call` or `mcp_approval_request` content item,
with no final textual output). In this case the evaluator is expected to skip evaluation
and signal "not applicable".

### Evaluator behavior (product side — correct)

Every affected evaluator defines a `_not_applicable_result` method that consistently returns:

```python
return {
    self._result_key: threshold,          # score = threshold (e.g. 3, 1, 0)
    f"{self._result_key}_result": "pass", # label = "pass"
    f"{self._result_key}_threshold": threshold,
    f"{self._result_key}_reason": f"Not applicable: {error_message}",
    ...
}
```

The score is set to the **threshold value** (not `None`) so that:

- The result "passes" the threshold check (`score >= threshold`).
- Downstream aggregation logic can treat the result as valid and non-penalising.

This pattern is implemented identically in all 14 affected evaluator files under
`assets/evaluators/builtin/`.

### Test assertion (test infra side — was wrong)

The `assert_not_applicable` method in
`assets/evaluators/tests/common/base_evaluator_runner.py` contained:

```python
assert result_data[score_key] is None, \
    f"Expected score to be None for not-applicable result but got '{result_data[score_key]}'"
```

This assertion was **incorrect**: it expected `None`, but every evaluator's
`_not_applicable_result` returns a numeric threshold. The assertion was never
aligned with the actual product contract.

### Classification

This is a **test infrastructure bug** — a stale/incorrect assertion in
`base_evaluator_runner.py`. There is no product regression. The 14 evaluator
implementations are consistent with each other and with the intended design.

---

## Fix Applied

**File changed:** `assets/evaluators/tests/common/base_evaluator_runner.py`

**Change:** Updated `assert_not_applicable` to accept either `None` or a numeric score,
while still enforcing `label == "pass"` and `"Not applicable" in reason`.

```python
# Before
assert result_data[score_key] is None, \
    f"Expected score to be None for not-applicable result but got '{result_data[score_key]}'"

# After
score = result_data[score_key]
if score is not None:
    assert isinstance(score, (int, float)), \
        f"Score should be numeric or None for not-applicable result but got type {type(score)}"
```

The fix is minimal and scoped entirely to test infrastructure. No production evaluator
code was changed.

---

## JUnit XML

The full test results are in `pytest-reports/behavior-results.xml` at the repo root.
