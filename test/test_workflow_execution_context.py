# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Protect the shared CI policy against label-based fork approval."""

from pathlib import Path

import pytest
import yaml


WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
GATE_WORKFLOW = "check-execution-context.yaml"
GATE_JOB = "check-execution-context"
REJECT_FORK = (
    "github.event.pull_request && "
    "github.event.pull_request.head.repo.full_name != github.repository"
)
ALLOW_EXECUTION = (
    "github.event_name == 'workflow_dispatch' || "
    "(github.event_name == 'pull_request' && "
    "github.event.pull_request.head.repo.full_name == github.repository)"
)
REPORT_GATE = (
    "always() && needs.check-execution-context.result == 'success' && "
    "needs.check-execution-context.outputs.continue == 'true'"
)
CALLERS = [
    "assets-test.yaml",
    "environments-ci.yaml",
    "scripts-test.yaml",
    "batch-score-ci.yaml",
    "batch-score-oss-ci.yaml",
    "model-monitoring-ci.yml",
    "model-monitoring-gsq-ci.yml",
]


def load_workflow(name):
    """Read workflow keys literally, including YAML 1.1's boolean-like 'on'."""
    with (WORKFLOWS_DIR / name).open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.BaseLoader)


def test_fork_rejection_precedes_all_other_steps():
    """Reject every cross-repository PR without consulting event type or labels."""
    job = load_workflow(GATE_WORKFLOW)["jobs"][GATE_JOB]
    rejection = job["steps"][0]
    assert rejection["if"] == REJECT_FORK
    assert "::error::" in rejection["run"]
    assert rejection["run"].strip().endswith("exit 1")
    assert "continue-on-error" not in rejection
    assert "continue-on-error" not in job
    assert all("uses" not in step for step in job["steps"])


def test_allowed_contexts_and_output_contract():
    """Keep same-repository PR/manual callers and their existing output names."""
    workflow = load_workflow(GATE_WORKFLOW)
    job = workflow["jobs"][GATE_JOB]
    steps = job["steps"]
    assert steps[1]["run"] == (
        'echo "continue=${{ ' + ALLOW_EXECUTION + ' }}" >> "$GITHUB_ENV"'
    )
    assert steps[2]["if"] == "fromJSON(env.continue)"
    assert steps[2]["run"] == 'echo "forked_pr=false" >> "$GITHUB_ENV"'
    assert job["outputs"] == {
        "continue": "${{ env.continue }}",
        "forked_pr": "${{ env.forked_pr }}",
    }
    assert set(workflow["on"]["workflow_call"]["outputs"]) == {"continue", "forked_pr"}


def test_no_label_override_remains():
    """Neither spelling of the deprecated label can influence the shared gate."""
    text = (WORKFLOWS_DIR / GATE_WORKFLOW).read_text(encoding="utf-8")
    assert "safe to test" not in text
    assert "safe_to_test" not in text
    assert "labels" not in text


@pytest.mark.parametrize("name", CALLERS)
def test_callers_have_no_privileged_or_label_trigger(name):
    """Fork approval must not reappear through pull_request_target or labels."""
    workflow = load_workflow(name)
    assert set(workflow["on"]) <= {"pull_request", "workflow_dispatch"}
    assert "labeled" not in workflow["on"]["pull_request"].get("types", [])
    assert workflow["jobs"][GATE_JOB]["uses"].endswith(f"/{GATE_WORKFLOW}@main")


@pytest.mark.parametrize("name", CALLERS)
def test_all_downstream_jobs_respect_rejection(name):
    """Every consumer depends on the gate, including always-running reports."""
    jobs = load_workflow(name)["jobs"]
    gated = {GATE_JOB}
    while True:
        previous = gated.copy()
        for job_id, job in jobs.items():
            needs = job.get("needs", [])
            if isinstance(needs, str):
                needs = [needs]
            if set(needs) & gated:
                gated.add(job_id)
        if gated == previous:
            break
    assert gated == set(jobs)

    for job in jobs.values():
        if "always()" in job.get("if", ""):
            assert GATE_JOB in job["needs"]
            assert job["if"] == REPORT_GATE
