# Trigger OneBranch Pipeline Workflow

## Overview
This workflow automatically triggers the Azure DevOps "AzureML-Assets-Unified-OneBranch" pipeline when the `assets-release` workflow completes successfully on the `main` branch.

## Configuration

### Trigger
The workflow is triggered when the `assets-release` workflow completes with a `success` status on the `main` branch.

### Pipeline Variables
The OneBranch pipeline is triggered with the following configuration:
- **Branch**: `main`
- **Variables**:
  - `pattern`: `environment/llm-rag.*/.+`

## Required Secrets

The following GitHub secrets must be configured for this workflow to function:

1. **AZURE_DEVOPS_ORG**: Your Azure DevOps organization name
   - Example: `myorganization`

2. **AZURE_DEVOPS_PROJECT**: Your Azure DevOps project name
   - Example: `MyProject`

3. **AZURE_DEVOPS_PAT**: Personal Access Token (PAT) with permissions to trigger pipelines
   - Required scopes: `Build (Read & execute)`
   - Generate a PAT at: `https://dev.azure.com/{org}/_usersSettings/tokens`

## How It Works

1. Monitors the `assets-release` workflow for completion
2. If the workflow completes successfully:
   - Queries the Azure DevOps API to find the pipeline by name
   - Triggers a new run of the pipeline with the specified branch and variables
   - Reports the run ID and URL

## Testing

To test this workflow:
1. Ensure all required secrets are configured
2. Push a change to the `main` branch that triggers the `assets-release` workflow
3. Wait for `assets-release` to complete successfully
4. Check the Actions tab for the `Trigger OneBranch Pipeline` workflow run

## Troubleshooting

- **Pipeline not found**: Verify the pipeline name matches exactly in Azure DevOps
- **Authentication failed**: Check that the PAT is valid and has the correct permissions
- **API call failed**: Review the error message in the workflow logs for details
