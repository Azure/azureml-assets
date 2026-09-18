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