# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared fixtures and helpers for agent evaluator end-to-end tests."""

import os
import time
import uuid
import logging
import functools

import pytest
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

logger = logging.getLogger(__name__)

EVALUATOR_NAMES = [
    "coherence",
    "fluency",
    "groundedness",
    "relevance",
    "intent_resolution",
    "task_adherence",
    "task_completion",
    "tool_call_success",
    "tool_call_accuracy",
    "tool_selection",
    "tool_input_accuracy",
    "tool_output_utilization",
]

# ---------------------------------------------------------------------------
# Evaluators that set ``check_for_unsupported_tools = True`` and return
# NOT_APPLICABLE for tool types listed in ConversationValidator.UNSUPPORTED_TOOLS.
# ---------------------------------------------------------------------------
UNSUPPORTED_TOOL_EVALUATORS = frozenset({
    "tool_call_accuracy",
    "tool_input_accuracy",
    "tool_output_utilization",
    "tool_call_success",
    "groundedness",
})


def unique_name(prefix: str) -> str:
    """Return a unique agent/eval name to avoid collisions."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Evaluator criteria builder
# ---------------------------------------------------------------------------
def get_all_evaluator_criteria(deployment_name: str) -> list[dict]:
    """Build testing_criteria list for all evaluators."""
    return [
        {
            "type": "azure_ai_evaluator",
            "name": name,
            "evaluator_name": f"builtin.{name}",
            "initialization_parameters": {"deployment_name": deployment_name},
        }
        for name in EVALUATOR_NAMES
    ]


# ---------------------------------------------------------------------------
# Skip helper for tests needing env-var-based connections
# ---------------------------------------------------------------------------
def requires_env(*env_vars):
    """Pytest marker that fails (not skips) when required env vars are missing."""
    missing = [v for v in env_vars if not os.environ.get(v)]
    if not missing:
        return pytest.mark.usefixtures()

    msg = f"Missing environment variables: {', '.join(missing)}"

    def decorator(cls_or_fn):
        if isinstance(cls_or_fn, type):
            # Decorate every test method in the class
            for attr_name in list(vars(cls_or_fn)):
                if attr_name.startswith("test_"):
                    original = getattr(cls_or_fn, attr_name)
                    setattr(cls_or_fn, attr_name, _make_failing_test(msg, original))
            return cls_or_fn
        return _make_failing_test(msg, cls_or_fn)

    return decorator


def _make_failing_test(msg, original):
    """Wrap a test function so it calls pytest.fail with *msg*."""
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        pytest.fail(msg)
    return wrapper


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_endpoint():
    """Provide the Azure AI project endpoint from environment."""
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        pytest.skip("AZURE_AI_PROJECT_ENDPOINT not set")
    return endpoint


@pytest.fixture(scope="session")
def model_deployment_name():
    """Provide the model deployment name from environment."""
    name = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    if not name:
        pytest.skip("AZURE_AI_MODEL_DEPLOYMENT_NAME not set")
    return name


@pytest.fixture(scope="session")
def credential():
    """Provide a DefaultAzureCredential for the test session."""
    cred = DefaultAzureCredential()
    yield cred
    cred.close()


@pytest.fixture(scope="session")
def project_client(project_endpoint, credential):
    """Provide a session-scoped AIProjectClient."""
    client = AIProjectClient(endpoint=project_endpoint, credential=credential)
    yield client
    client.close()


@pytest.fixture(scope="session")
def openai_client(project_client):
    """Provide a session-scoped OpenAI client from the project."""
    client = project_client.get_openai_client()
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
EVAL_POLL_INTERVAL = 5  # seconds
EVAL_TIMEOUT = 600  # 10 minutes


def run_evaluation(openai_client, model_deployment_name, response_id, agent_name, eval_name):
    """Create an eval with all 13 evaluators, run it against *response_id*, and return results.

    Returns:
        tuple: (eval_run, output_items)
    """
    data_source_config = {"type": "azure_ai_source", "scenario": "responses"}
    testing_criteria = get_all_evaluator_criteria(model_deployment_name)

    eval_object = openai_client.evals.create(
        name=eval_name,
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,
    )

    try:
        data_source = {
            "type": "azure_ai_responses",
            "item_generation_params": {
                "type": "response_retrieval",
                "data_mapping": {"response_id": "{{item.resp_id}}"},
                "source": {
                    "type": "file_content",
                    "content": [{"item": {"resp_id": response_id}}],
                },
            },
        }

        eval_run = openai_client.evals.runs.create(
            eval_id=eval_object.id,
            name=f"E2E Run - {agent_name}",
            data_source=data_source,
        )
        logger.info("Eval run created: %s (eval: %s)", eval_run.id, eval_object.id)

        start = time.time()
        while eval_run.status not in ("completed", "failed"):
            elapsed = time.time() - start
            if elapsed > EVAL_TIMEOUT:
                raise TimeoutError(
                    f"Eval run did not complete within {EVAL_TIMEOUT}s. " f"Last status: {eval_run.status}"
                )
            time.sleep(EVAL_POLL_INTERVAL)
            eval_run = openai_client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
            logger.info("Eval run status: %s (%.0fs elapsed)", eval_run.status, elapsed)

        output_items = list(openai_client.evals.runs.output_items.list(run_id=eval_run.id, eval_id=eval_object.id))

        # Log error details if the run failed
        if eval_run.status == "failed":
            error = getattr(eval_run, "error", None)
            logger.error(
                "Eval run FAILED. error=%s, result_counts=%s, " "per_model_usage=%s, per_testing_criteria_results=%s",
                error,
                eval_run.result_counts,
                getattr(eval_run, "per_model_usage", None),
                getattr(eval_run, "per_testing_criteria_results", None),
            )

        logger.info(
            "Eval completed – result_counts: %s, output_items: %d",
            eval_run.result_counts,
            len(output_items),
        )

        return eval_run, output_items

    finally:
        openai_client.evals.delete(eval_id=eval_object.id)


def assert_evaluation_results(
    eval_run,
    output_items,
    expected_not_applicable=None,
    expected_errors=None,
    expected_failures=None,
    tolerated_failures=None,
):
    """Assert evaluation completed successfully with expected results.

    Error handling follows the unit-test contract defined in
    ``base_tool_evaluation_test.py``:

    * **expected_not_applicable** – evaluators that should return a
      ``NOT_APPLICABLE`` error because the tool type is in
      ``ConversationValidator.UNSUPPORTED_TOOLS``.  We assert the error
      message contains the expected "not supported for" pattern.
    * **expected_errors** – dict mapping evaluator name → expected error
      message substring.  We assert the actual error message contains the
      expected substring.
    * **expected_failures** – evaluators that MUST fail the quality
      threshold.  We assert the score is numeric and below the threshold.
      If the evaluator passes instead, the test fails.
    * **tolerated_failures** – evaluators that MAY fail without causing the
      test to fail, but are also allowed to pass.  Useful for evaluators
      whose outcome is non-deterministic due to external factors (e.g. an
      empty knowledge base).
    * Any error or failure not covered by the above sets **fails** the test.

    Args:
        eval_run: The completed eval run object.
        output_items: List of output items from the eval run.
        expected_not_applicable: Set of evaluator names expected to return
            NOT_APPLICABLE (unsupported tool type).
        expected_errors: Dict mapping evaluator name to expected error
            message substring (see known_issues.md).
        expected_failures: Set of evaluator names that must fail quality.
        tolerated_failures: Set of evaluator names that may fail without
            causing the test to fail.
    """
    expected_not_applicable = expected_not_applicable or set()
    expected_errors = expected_errors or {}
    expected_failures = expected_failures or set()
    tolerated_failures = tolerated_failures or set()

    # --- 1. Eval run completed ---
    assert eval_run.status == "completed", (
        f"Eval run did not complete. Status: {eval_run.status}. "
        f"Result counts: {eval_run.result_counts}"
    )

    # --- 2. Output items exist ---
    assert len(output_items) > 0, "No output items produced by evaluation"

    # --- 3. Validate each evaluator result ---
    for idx, item in enumerate(output_items):
        results = getattr(item, "results", None)
        if results is None:
            attrs = sorted(vars(item).keys()) if hasattr(item, "__dict__") else dir(item)
            pytest.fail(
                f"Output item {idx} has no 'results' attribute. "
                f"Available: {attrs}. Repr: {repr(item)}"
            )

        evaluator_results_found = set()
        unexpected_errors = []
        unexpected_failures = []
        unexpected_passes = []
        expected_errors_seen = set()
        expected_failures_seen = set()

        for result in results:
            name = getattr(result, "name", None) or "unknown"
            evaluator_results_found.add(name)

            label = getattr(result, "label", None)
            if label is None:
                passed = getattr(result, "passed", None)
                if passed is not None:
                    label = "pass" if passed else "fail"

            score = getattr(result, "score", None)
            reason = getattr(result, "reason", None)
            threshold = getattr(result, "threshold", None)

            sample = getattr(result, "sample", None)
            error = sample.get("error") if isinstance(sample, dict) else None

            if error:
                error_msg = (
                    error.get("message", "unknown error")
                    if isinstance(error, dict) else str(error)
                )
                error_code = (
                    error.get("code", "")
                    if isinstance(error, dict) else ""
                )

                if name in expected_not_applicable:
                    # Validate the NOT_APPLICABLE error: the error code is
                    # FAILED_EXECUTION and the message contains the
                    # "not supported for" pattern from ConversationValidator.
                    msg_lower = error_msg.lower()
                    assert error_code == "FAILED_EXECUTION" and "not supported for" in msg_lower, (
                        f"Evaluator '{name}' was expected to return "
                        f"NOT_APPLICABLE but got: code={error_code!r}, "
                        f"message={error_msg!r}"
                    )
                    expected_errors_seen.add(name)
                    logger.info(
                        "Evaluator '%s': NOT_APPLICABLE (expected) - "
                        "code=%s, message=%s, sample=%r",
                        name, error_code, error_msg, sample,
                    )
                elif name in expected_errors:
                    expected_substring = expected_errors[name]
                    assert expected_substring.lower() in error_msg.lower(), (
                        f"Evaluator '{name}' errored but message does not "
                        f"match expected substring.\n"
                        f"  Expected substring: {expected_substring!r}\n"
                        f"  Actual message: {error_msg!r}"
                    )
                    expected_errors_seen.add(name)
                    logger.warning(
                        "Evaluator '%s': expected error (documented) - "
                        "code=%s, message=%s, sample=%r",
                        name, error_code, error_msg, sample,
                    )
                else:
                    logger.error(
                        "Evaluator '%s': UNEXPECTED error - "
                        "code=%s, message=%s, sample=%r",
                        name, error_code, error_msg, sample,
                    )
                    unexpected_errors.append(
                        f"'{name}': code={error_code!r}, message={error_msg!r}"
                    )

            elif label != "pass":
                if name in expected_failures:
                    assert isinstance(score, (int, float)), (
                        f"Evaluator '{name}' failed with non-numeric "
                        f"score: {score!r} (type={type(score).__name__})"
                    )
                    if threshold is not None:
                        assert score <= threshold, (
                            f"Evaluator '{name}' score={score} > "
                            f"threshold={threshold} but label={label}. "
                            f"Not a genuine quality failure."
                        )
                    expected_failures_seen.add(name)
                    logger.warning(
                        "Evaluator '%s': expected failure "
                        "(score=%s, threshold=%s, reason=%s)",
                        name, score, threshold, reason,
                    )
                elif name in tolerated_failures:
                    logger.warning(
                        "Evaluator '%s': tolerated failure "
                        "(score=%s, threshold=%s, reason=%s)",
                        name, score, threshold, reason,
                    )
                else:
                    unexpected_failures.append(
                        f"'{name}': label={label}, score={score}, "
                        f"threshold={threshold}, reason={reason}"
                    )
            else:
                if name in expected_failures:
                    unexpected_passes.append(
                        f"'{name}': expected to fail but PASSED "
                        f"(score={score}, threshold={threshold})"
                    )
                elif name in tolerated_failures:
                    logger.info(
                        "Evaluator '%s': PASS (tolerated, score=%s, threshold=%s)",
                        name, score, threshold,
                    )
                else:
                    logger.info(
                        "Evaluator '%s': PASS (score=%s, threshold=%s)",
                        name, score, threshold,
                    )

        if unexpected_errors:
            pytest.fail(
                f"{len(unexpected_errors)} evaluator(s) errored "
                f"unexpectedly:\n" + "\n".join(unexpected_errors)
            )

        if unexpected_failures:
            pytest.fail(
                f"{len(unexpected_failures)} evaluator(s) did not pass:\n"
                + "\n".join(unexpected_failures)
            )

        if unexpected_passes:
            pytest.fail(
                f"{len(unexpected_passes)} evaluator(s) were expected to fail "
                f"but passed instead:\n" + "\n".join(unexpected_passes)
            )

        # --- Verify all expected errors were actually seen ---
        all_expected = set(expected_not_applicable) | set(expected_errors.keys())
        missing_errors = all_expected - expected_errors_seen
        if missing_errors:
            pytest.fail(
                f"{len(missing_errors)} evaluator(s) were expected to error "
                f"but succeeded instead: {', '.join(sorted(missing_errors))}"
            )

        # --- Verify all expected failures were actually seen ---
        missing_failures = expected_failures - expected_failures_seen
        if missing_failures:
            pytest.fail(
                f"{len(missing_failures)} evaluator(s) were expected to fail "
                f"but did not appear as failures: {', '.join(sorted(missing_failures))}"
            )

        missing = set(EVALUATOR_NAMES) - evaluator_results_found
        assert not missing, (
            f"Missing evaluator results for: {missing}. "
            f"Got results for: {evaluator_results_found}"
        )
