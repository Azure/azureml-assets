# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate workflow fork policy without executing candidate repository code.

Examples:
    python scripts/validation/workflow_policy.py --repo .
    python scripts/validation/workflow_policy.py --repo . --ref <40-character-commit>

Policy is always loaded beside this trusted script, never from --ref. Reference
mode reads workflow blobs from Git without checking out or importing PR files.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
GATE_FILE = "check-execution-context.yaml"
GATE_JOB = "check-execution-context"
GATE_USES = "Azure/azureml-assets/.github/workflows/check-execution-context.yaml@main"
POLICY_FILE = "workflow-policy.yaml"
POLICY_REUSABLE = "check-workflow-policy.yaml"
POLICY_USES = f"./.github/workflows/{POLICY_REUSABLE}"
REPORT_CONDITION = (
    "always() && needs.check-execution-context.result == 'success' && "
    "needs.check-execution-context.outputs.continue == 'true'"
)
REJECT_CONDITION = (
    "github.event.pull_request && github.event.pull_request.head.repo.full_name != github.repository"
)
CONTINUE_COMMAND = (
    'echo "continue=${{ github.event_name == \'workflow_dispatch\' || '
    "(github.event_name == 'pull_request' && "
    'github.event.pull_request.head.repo.full_name == github.repository) }}" >> "$GITHUB_ENV"'
)
FETCH_COMMAND = 'git fetch --no-tags --depth=1 --no-recurse-submodules origin "$PR_HEAD_SHA"'
VALIDATE_COMMAND = 'python scripts/validation/workflow_policy.py --repo . --ref "$PR_HEAD_SHA"'
MAX_WORKFLOW_BYTES = 1024 * 1024
STATUS_FUNCTION = re.compile(r"\b(?:always|cancelled|failure|success)\s*\(", re.IGNORECASE)


class PolicyError(ValueError):
    """Indicate malformed or noncompliant workflow policy input."""


class WorkflowLoader(yaml.BaseLoader):
    """Keep 'on' and boolean-like scalars literal and reject duplicate keys."""

    def construct_mapping(self, node, deep=False):
        """Reject ambiguous mappings instead of silently accepting the last key."""
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result or key == "<<":
                raise PolicyError("Workflow mappings require unique scalar keys; YAML merge keys are unsupported")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def parse_workflow(content):
    """Parse a bounded YAML document as data only."""
    if len(content) > MAX_WORKFLOW_BYTES:
        raise PolicyError("Workflow exceeds the 1 MiB policy limit")
    try:
        document = yaml.load(content.decode("utf-8"), Loader=WorkflowLoader)
    except (UnicodeError, yaml.YAMLError, RecursionError) as error:
        raise PolicyError(f"Invalid workflow YAML: {error}") from error
    if not isinstance(document, dict):
        raise PolicyError("Workflow must be a mapping")
    return document


def workflow_events(document):
    """Normalize the supported GitHub event declaration forms."""
    events = document.get("on")
    if isinstance(events, str):
        return {events}
    if isinstance(events, dict):
        return set(events)
    if isinstance(events, list) and all(isinstance(event, str) for event in events):
        return set(events)
    raise PolicyError("Workflow must declare its events")


def condition(value):
    """Normalize expression wrappers without interpreting candidate expressions."""
    if not isinstance(value, str):
        raise PolicyError("Job conditions must be strings")
    value = value.strip()
    if value.startswith("${{") and value.endswith("}}"):
        value = value[3:-2].strip()
    return " ".join(value.split())


def require(value, message):
    """Fail closed when a policy invariant is not met."""
    if not value:
        raise PolicyError(message)


def dependencies(job):
    """Read a static job dependency list."""
    needs = job.get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    require(isinstance(needs, list) and all(isinstance(name, str) for name in needs),
            "Job dependencies must be job names")
    return set(needs)


def validate_gated_workflow(document):
    """Require the trusted gate and fail-closed dependencies for every other job."""
    require(workflow_events(document) <= {"pull_request", "workflow_dispatch"},
            "Non-exempt workflows support only pull_request and workflow_dispatch")
    jobs = document.get("jobs")
    require(isinstance(jobs, dict) and all(isinstance(job, dict) for job in jobs.values()),
            "Workflow jobs must be mappings")
    gate = jobs.get(GATE_JOB, {})
    require(gate.get("uses") == GATE_USES, f"Missing trusted {GATE_JOB} reusable workflow")
    require(set(gate) <= {"name", "uses", "permissions"}, "The execution gate cannot be conditional or overridden")
    graph = {name: dependencies(job) for name, job in jobs.items()}
    require(all(needs <= jobs.keys() for needs in graph.values()), "Unknown job dependency")

    # Topological traversal also rejects cycles and independent, ungated roots.
    reached = {GATE_JOB}
    remaining = set(jobs) - reached
    while remaining:
        ready = {name for name in remaining if graph[name] and graph[name] <= reached}
        require(ready, "Every job must depend on the execution gate; ungated root or dependency cycle found")
        reached.update(ready)
        remaining.difference_update(ready)

    for name, job in jobs.items():
        expression = condition(job.get("if", ""))
        if STATUS_FUNCTION.search(expression):
            require(GATE_JOB in graph[name] and expression == REPORT_CONDITION,
                    f"{name}: status-function conditions must explicitly require successful gate approval")


def validate_gate(document):
    """Protect the shared rejection and output contract from being weakened."""
    require(workflow_events(document) == {"workflow_call"}, "The fork gate must remain reusable-only")
    jobs = document.get("jobs", {})
    require(isinstance(jobs, dict) and set(jobs) == {GATE_JOB}, "The fork gate must have exactly one job")
    job = jobs[GATE_JOB]
    require(isinstance(job, dict) and set(job) <= {"name", "runs-on", "permissions", "steps", "outputs"},
            "The fork gate must run unconditionally and propagate failure")
    require(document.get("defaults") == {"run": {"shell": "bash"}}
            and set(document) <= {"name", "on", "defaults", "jobs"},
            "The fork gate commands require Bash")
    require(job.get("runs-on") == "ubuntu-latest" and job.get("permissions") == {"contents": "read"},
            "The fork gate must use a read-only hosted runner")
    steps = job.get("steps", [])
    require(isinstance(steps, list) and len(steps) == 3, "Unexpected fork gate steps")
    require(all(isinstance(step, dict) and set(step) <= {"name", "if", "run"} for step in steps),
            "Fork gate steps must not override failure handling or execute other actions")
    require(condition(steps[0].get("if", "")) == REJECT_CONDITION, "Fork rejection must not depend on labels")
    require(isinstance(steps[0].get("run"), str), "Fork rejection must have a shell command")
    rejection = steps[0]["run"].strip().splitlines()
    require(len(rejection) == 2 and re.fullmatch(r'echo "::error::[A-Za-z0-9 .-]+"', rejection[0])
            and rejection[1] == "exit 1", "Fork rejection must report an error and exit 1")
    require("if" not in steps[1] and steps[1].get("run") == CONTINUE_COMMAND,
            "Only same-repository PRs and manual runs may continue")
    require(condition(steps[2].get("if", "")) == "fromJSON(env.continue)"
            and steps[2].get("run") == 'echo "forked_pr=false" >> "$GITHUB_ENV"',
            "Allowed runs must retain the non-fork output")
    require(job.get("outputs") == {
        "continue": "${{ env.continue }}", "forked_pr": "${{ env.forked_pr }}",
    }, "The execution gate output contract must be preserved")
    trigger = document.get("on")
    require(isinstance(trigger, dict) and isinstance(trigger.get("workflow_call"), dict),
            "The reusable gate must declare its outputs")
    outputs = trigger["workflow_call"].get("outputs", {})
    require(isinstance(outputs, dict) and all(isinstance(value, dict) for value in outputs.values()),
            "Reusable gate outputs must be mappings")
    require(all(outputs.get(key, {}).get("value") == f"${{{{ jobs.{GATE_JOB}.outputs.{key} }}}}"
                for key in ("continue", "forked_pr")), "Reusable gate outputs must come from the gate job")


def validate_policy_entrypoint(document):
    """Keep the mandatory checker on the base-side PR event with no path filters."""
    require(document.get("permissions") == {"contents": "read"}, "Policy entry point must be read-only")
    require(document.get("on") == {
        "pull_request_target": {
            "branches": ["main"], "types": ["opened", "synchronize", "reopened", "edited"],
        },
    }, "Policy entry point must run on every PR update to main, without path filters")
    require(document.get("jobs") == {
        "workflow-policy": {"uses": POLICY_USES, "permissions": {"contents": "read"}},
    }, "Policy entry point must call the trusted reusable validator without a condition")


def validate_policy_reusable(document):
    """Ensure privileged event context never executes or checks out PR code."""
    require(workflow_events(document) == {"workflow_call"}, "Policy implementation must be reusable-only")
    require(document.get("permissions") == {"contents": "read"}, "Policy implementation must be read-only")
    jobs = document.get("jobs", {})
    require(isinstance(jobs, dict) and set(jobs) == {"validate"},
            "Policy implementation must have one validation job")
    job = jobs["validate"]
    require(isinstance(job, dict) and set(job) == {"name", "runs-on", "timeout-minutes", "env", "steps"}
            and job.get("runs-on") == "ubuntu-latest" and job.get("timeout-minutes") == "5",
            "Policy validation must run unconditionally on a bounded hosted runner")
    require(job.get("env") == {"PR_HEAD_SHA": "${{ github.event.pull_request.head.sha }}"},
            "Only the immutable PR commit may be supplied as candidate input")
    steps = job.get("steps", [])
    require(isinstance(steps, list) and all(isinstance(step, dict) for step in steps),
            "Invalid policy implementation steps")
    steps = [{key: value for key, value in step.items() if key != "name"} for step in steps]
    require(len(steps) == 5, "Unexpected policy implementation steps")
    require(isinstance(steps[0].get("uses"), str)
            and re.fullmatch(r"actions/checkout@[a-f0-9]{40}", steps[0]["uses"]) is not None
            and set(steps[0]) == {"uses", "with"} and steps[0]["with"] == {
                "ref": "${{ github.event.pull_request.base.sha }}", "persist-credentials": "false",
            }, "Check out only the trusted base commit, without persisted credentials")
    require(isinstance(steps[1].get("uses"), str)
            and re.fullmatch(r"actions/setup-python@[a-f0-9]{40}", steps[1]["uses"]) is not None
            and set(steps[1]) == {"uses", "with"} and steps[1]["with"] == {"python-version": "3.12"},
            "Use a pinned Python setup action")
    require(steps[2:] == [
        {"run": "python -m pip install -r scripts/validation/requirements.txt"},
        {"run": FETCH_COMMAND}, {"run": VALIDATE_COMMAND},
    ], "Run only trusted dependencies and validator; fetch candidate Git objects without checkout")
    require(set(document) <= {"name", "on", "permissions", "jobs"},
            "Policy implementation must not override trusted execution defaults")


def validate_workflows(workflows, policy):
    """Return all workflow policy errors, including entirely new ungated files."""
    require(isinstance(policy, dict), "Workflow policy must be a mapping")
    exceptions = policy.get("exceptions")
    require(policy.get("schema_version") == 1 and isinstance(exceptions, dict), "Invalid workflow policy schema")
    for name, entry in exceptions.items():
        require(isinstance(entry, dict) and isinstance(entry.get("reason"), str) and entry["reason"].strip()
                and isinstance(entry.get("events"), list) and entry["events"]
                and all(isinstance(event, str) for event in entry["events"]), f"Invalid exception: {name!r}")
        require("pull_request_target" not in entry["events"] or name == POLICY_FILE,
                "Only the trusted policy entry point may use pull_request_target")
        require(set(entry["events"]) <= {
            "pull_request", "pull_request_target", "workflow_dispatch", "workflow_call", "push", "schedule",
        }, f"Unsupported exception events: {name!r}")

    errors = [f"Required workflow missing: {name}" for name in (GATE_FILE, POLICY_FILE, POLICY_REUSABLE)
              if name not in workflows]
    for name, content in sorted(workflows.items()):
        try:
            document = parse_workflow(content)
            events = workflow_events(document)
            require(events, "Workflow must have at least one event")
            if name in exceptions:
                require(events <= set(exceptions[name]["events"]), "Events exceed the reviewed exception")
            else:
                validate_gated_workflow(document)
            if name == GATE_FILE:
                validate_gate(document)
            elif name == POLICY_FILE:
                validate_policy_entrypoint(document)
            elif name == POLICY_REUSABLE:
                validate_policy_reusable(document)
        except PolicyError as error:
            errors.append(f"{name!r}: {error}")
    return errors


def git_output(repo, *args):
    """Read Git data without invoking a shell or checking out candidate files."""
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def read_workflows(repo, ref=None):
    """Discover every workflow in the working tree or an immutable Git commit."""
    if ref is None:
        directory = repo / ".github" / "workflows"
        require(not directory.is_symlink(), "Workflow directory must not be a symbolic link")
        files = [path for path in directory.iterdir() if path.suffix.lower() in {".yml", ".yaml"}]
        require(all(path.is_file() and not path.is_symlink() for path in files),
                "Workflow files must be regular files, not symbolic links")
        require(all(path.stat().st_size <= MAX_WORKFLOW_BYTES for path in files),
                "Workflow exceeds the 1 MiB policy limit")
        return {path.name: path.read_bytes() for path in files}

    require(re.fullmatch(r"[a-fA-F0-9]{40}", ref) is not None, "--ref must be a full commit SHA")
    tree = git_output(repo, "ls-tree", "-z", f"{ref}:.github/workflows")
    workflows = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, filename = entry.split(b"\t", 1)
        name = filename.decode("utf-8")
        if Path(name).suffix.lower() not in {".yml", ".yaml"}:
            continue
        mode, kind, oid = metadata.decode("ascii").split()
        require(mode in {"100644", "100755"} and kind == "blob", f"Not a regular workflow file: {name!r}")
        size = int(git_output(repo, "cat-file", "-s", oid))
        require(size <= MAX_WORKFLOW_BYTES, f"Workflow exceeds the 1 MiB policy limit: {name!r}")
        workflows[name] = git_output(repo, "cat-file", "blob", oid)
    return workflows


def parse_args():
    """Parse local or trusted-base validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=ROOT, help="Repository whose workflows are inspected")
    parser.add_argument("--ref", help="Inspect this full commit SHA as data instead of the working tree")
    return parser.parse_args()


def main():
    """Print a structured result and fail on unreadable or noncompliant input."""
    args = parse_args()
    try:
        policy = json.loads((ROOT / ".github" / "workflow-policy.json").read_text(encoding="utf-8"))
        workflows = read_workflows(args.repo, args.ref)
        errors = validate_workflows(workflows, policy)
    except (PolicyError, OSError, UnicodeError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"errors": [str(error)]}, indent=2))
        return 1
    print(json.dumps({"workflows": len(workflows), "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
