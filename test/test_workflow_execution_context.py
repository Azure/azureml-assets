# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Protect the shared CI policy against label-based fork approval."""

from pathlib import Path
import runpy
import shlex
import subprocess
import xml.etree.ElementTree as ET

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


def test_assets_test_has_fixed_summary_check():
    """Keep the required check separate from matrix job display names."""
    jobs = load_workflow("assets-test.yaml")["jobs"]
    assert jobs["test"]["name"] == "Test asset"
    assert jobs["test-summary"]["name"] == "Test"
    assert jobs["test-summary"]["if"] == REPORT_GATE
    assert set(jobs["test-summary"]["needs"]) == {
        GATE_JOB, "check-directory-file-changes", "setup", "test",
        "released-sdk-compatibility", "candidate-native-sdk",
    }
    compatibility = jobs["released-sdk-compatibility"]
    assert compatibility["permissions"] == {"contents": "read"}
    assert compatibility["if"] == (
        "contains(needs.check-directory-file-changes.outputs.files-changed, 'assets/evaluators/')"
    )
    assert set(compatibility["needs"]) == {GATE_JOB, "check-directory-file-changes"}
    assert "released-sdk-compatibility" in jobs["report"]["needs"]
    assert compatibility["env"]["AZUREML_COMPAT_SDK_ROOT"] == (
        "${{ github.workspace }}/.released-sdk-compatibility"
    )
    native = jobs["candidate-native-sdk"]
    assert native["permissions"] == {"contents": "read"}
    assert native["if"] == compatibility["if"]
    assert native["needs"] == compatibility["needs"]
    assert "candidate-native-sdk" in jobs["report"]["needs"]
    command = next(step["run"] for step in native["steps"] if "run" in step)
    assert "python -m pip wheel --no-deps" in command
    assert "python scripts/test/check_native_sdk.py" in command
    assert "environment" not in native


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "2", "TEST_RESULT": "success"}, 0),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "2", "TEST_RESULT": "failure"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "2", "TEST_RESULT": "cancelled"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "failure",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "cancelled",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "skipped",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "", "SETUP_RESULT": "skipped",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 0),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "0", "TEST_RESULT": "skipped"}, 0),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "1", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "invalid", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "00", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "01", "TEST_RESULT": "success"}, 1),
        ({"CHANGES_RESULT": "failure", "CHANGED_FILES": "", "SETUP_RESULT": "skipped",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "cancelled", "CHANGED_FILES": "", "SETUP_RESULT": "skipped",
          "TEST_COUNT": "", "TEST_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/evaluators/builtin/output_quality/spec.yaml",
          "SETUP_RESULT": "success", "TEST_COUNT": "1", "TEST_RESULT": "success",
          "RELEASED_SDK_RESULT": "success", "NATIVE_SDK_RESULT": "success"}, 0),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/evaluators/tests/test_decision_models/conftest.py",
          "SETUP_RESULT": "success", "TEST_COUNT": "0", "TEST_RESULT": "skipped",
          "RELEASED_SDK_RESULT": "success", "NATIVE_SDK_RESULT": "success"}, 0),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/evaluators/builtin/output_quality/spec.yaml",
          "SETUP_RESULT": "success", "TEST_COUNT": "1", "TEST_RESULT": "success",
          "RELEASED_SDK_RESULT": "failure"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/evaluators/builtin/output_quality/spec.yaml",
          "SETUP_RESULT": "success", "TEST_COUNT": "1", "TEST_RESULT": "success",
          "RELEASED_SDK_RESULT": "cancelled"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/evaluators/builtin/output_quality/spec.yaml",
          "SETUP_RESULT": "success", "TEST_COUNT": "1", "TEST_RESULT": "success",
          "RELEASED_SDK_RESULT": "skipped"}, 1),
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "1", "TEST_RESULT": "success", "RELEASED_SDK_RESULT": "success"}, 1),
        *[
            ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/evaluators/tests/test_decision_models/conftest.py",
              "SETUP_RESULT": "success", "TEST_COUNT": "0", "TEST_RESULT": "skipped",
              "RELEASED_SDK_RESULT": "success", "NATIVE_SDK_RESULT": result}, 1)
            for result in ("failure", "cancelled", "skipped")
        ],
        ({"CHANGES_RESULT": "success", "CHANGED_FILES": "assets/a", "SETUP_RESULT": "success",
          "TEST_COUNT": "0", "TEST_RESULT": "skipped", "NATIVE_SDK_RESULT": "success"}, 1),
    ],
)
def test_assets_test_summary_result(state, expected, tmp_path):
    """Exercise successful, failed, cancelled and legitimate skip outcomes."""
    step = load_workflow("assets-test.yaml")["jobs"]["test-summary"]["steps"][0]
    environment = "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in {"RELEASED_SDK_RESULT": "skipped", "NATIVE_SDK_RESULT": "skipped", **state}.items()
    )
    script = tmp_path / "test-summary.sh"
    script.write_text(f"{environment}\n{step['run']}", encoding="utf-8", newline="\n")
    result = subprocess.run(
        ["bash", script.name],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize("change", [
    None, "additive", "missing_module", "missing_cases", "missing_native_variant",
    "skipped", "failure", "error",
])
def test_candidate_native_report_rejects_incomplete_or_skipped_coverage(tmp_path, change):
    """Reject missing native modules, required cases or nonpassing results."""
    runner = runpy.run_path(str(WORKFLOWS_DIR.parents[1] / "scripts" / "test" / "check_native_sdk.py"))
    tree = ET.Element("testsuite")
    for module, minimum in runner["MINIMUM_CASES"].items():
        required = sorted(name for name in runner["REQUIRED_CASES"]
                          if ("public_evaluate" in name) == (module == "test_composites"))
        names = required + [f"test_additional[{index}]" for index in range(minimum - len(required))]
        for name in names:
            ET.SubElement(tree, "testcase", classname=f"assets.tests.{module}", name=name)
    first = tree[0]
    if change == "additive":
        ET.SubElement(tree, "testcase", classname=first.get("classname"), name=first.get("name") + "[new]")
    elif change == "missing_module":
        for case in list(tree):
            if case.get("classname") == first.get("classname"):
                tree.remove(case)
    elif change == "missing_cases":
        tree.remove(first)
    elif change == "missing_native_variant":
        first.set("name", "test_replacement")
    elif change in ("skipped", "failure", "error"):
        ET.SubElement(first, change)
    report = tmp_path / "native.xml"
    ET.ElementTree(tree).write(report)
    if change in (None, "additive"):
        runner["verify_report"](report)
    else:
        with pytest.raises(AssertionError):
            runner["verify_report"](report)
