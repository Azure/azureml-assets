# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Pull request CI policy

Workflows that use [the shared execution-context check](.github/workflows/check-execution-context.yaml)
only run their build and test jobs for pull requests from branches in this repository
or supported manual runs. Fork pull requests fail that check before those jobs run.
The `safe to test` label is deprecated and no longer authorizes testing a fork.

For contributions that need these workflows, work with a maintainer to review the
changes before moving them to an in-repository branch. Other fork-safe validation
workflows are unchanged.

### Adding a workflow

Use the existing reusable gate rather than copying its implementation:

```yaml
on:
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  check-execution-context:
    uses: Azure/azureml-assets/.github/workflows/check-execution-context.yaml@main

  build:
    needs: check-execution-context
    runs-on: ubuntu-latest
    steps:
      - run: echo "Add build/test steps here"
```

Every other job must depend on the gate, directly or through gated jobs.
Conditions without status functions retain GitHub's implicit `success()` check.
A report that must run after a failed test needs a direct dependency on the gate
and this explicit condition:

```yaml
needs: [check-execution-context, build]
if: always() && needs.check-execution-context.result == 'success' && needs.check-execution-context.outputs.continue == 'true'
```

### Repository-wide enforcement

The [workflow-policy entry point](.github/workflows/workflow-policy.yaml) runs
independently on every PR update to `main`, including workflow-only changes.
It calls [the reusable validator](.github/workflows/check-workflow-policy.yaml).
There is no need to copy the checker into each build pipeline. Another PR
pipeline in this repository can call it with:

```yaml
jobs:
  workflow-policy:
    uses: Azure/azureml-assets/.github/workflows/check-workflow-policy.yaml@main
    permissions:
      contents: read
```

The checker discovers **all** `.yml` and `.yaml` workflows, including newly added
files. A workflow must use the fork gate unless its exact filename and event
scope have a reasoned exception in
[workflow-policy.json](.github/workflow-policy.json). Existing fork-safe checks,
non-PR workflows, and reusable helpers are explicitly classified there. New
ungated jobs, bypassing status conditions, conditional/replaced gates, and
unapproved trigger changes fail validation.

The one `pull_request_target` policy entry point is read-only: it checks out the
trusted **base commit**, loads the validator, dependencies and exception policy
from that commit, and reads candidate Git blobs without checking out or running
PR code. The validator also checks its own workflow wiring and the fork gate's
rejection/output contract. Candidate changes to the validator or exception list
cannot authorize themselves in the same run.

Validate locally with:

```powershell
python scripts\validation\workflow_policy.py --repo .
python -m pytest test\test_workflow_execution_context.py test\test_workflow_policy.py
```

The read-only `scripts-syntax` PR workflow also runs these regression tests.
Those tests exercise proposed validator changes; the independent trusted-base
policy check remains the enforcement authority.

An exception for a new workflow must be reviewed and landed in the base policy
first, then the workflow PR must receive a new update. Do not relax the checker
in the same PR just to make a new workflow pass. Policy-sensitive changes are
owned by `@azure/aml-assets` in `CODEOWNERS`.

**Administrator setup:** after the initial policy implementation is merged into
`main` and its check has appeared, require
`workflow-policy / Validate workflow policy` for PR merges and require code-owner
approval for workflow and validation-policy changes. Require branches to be up
to date so a passing check used the current base policy. Adding YAML alone does not
configure those repository rules. The first policy PR cannot run its new
base-side handler before that handler exists on `main`; its policy is validated
locally during bootstrap. Merge-queue users also need a separately reviewed
`merge_group` entry point before making this check mandatory for their queue.