# OneBranch Pipeline Trigger Setup - Summary

## What Was Implemented

A new GitHub Actions workflow has been created that automatically triggers the Azure DevOps "AzureML-Assets-Unified-OneBranch" pipeline when the `assets-release` workflow completes successfully on the `main` branch.

### Files Created

1. **`.github/workflows/trigger-onebranch-pipeline.yaml`**
   - Main workflow file that handles the pipeline trigger
   - Runs automatically after successful completion of assets-release workflow
   - Configured with:
     - Branch: `main`
     - Variables: `pattern=environment/llm-rag.*/.+`

2. **`.github/workflows/TRIGGER_ONEBRANCH_README.md`**
   - Comprehensive documentation
   - Setup instructions
   - Troubleshooting guide

## Next Steps Required

To complete the setup and enable this workflow, you need to:

### 1. Configure GitHub Secrets

Add the following secrets to your GitHub repository (Settings > Secrets and variables > Actions):

- **`AZURE_DEVOPS_ORG`**
  - Value: Your Azure DevOps organization name
  - Example: If your org URL is `https://dev.azure.com/myorg/`, use `myorg`

- **`AZURE_DEVOPS_PROJECT`**
  - Value: Your Azure DevOps project name
  - Example: The project containing the OneBranch pipeline

- **`AZURE_DEVOPS_PAT`**
  - Value: A Personal Access Token with Build (Read & execute) permissions
  - To create:
    1. Go to `https://dev.azure.com/{org}/_usersSettings/tokens`
    2. Click "New Token"
    3. Set appropriate expiration
    4. Select scope: **Build (Read & execute)**
    5. Copy the token value

### 2. Verify Pipeline Name

Ensure the pipeline is named exactly **"AzureML-Assets-Unified-OneBranch"** in Azure DevOps, or update line 37 in the workflow file if the name is different.

### 3. Test the Workflow

After configuring the secrets:
1. The workflow will automatically trigger when the next `assets-release` workflow completes successfully
2. Monitor the workflow in: GitHub > Actions > "Trigger OneBranch Pipeline"
3. Check the logs to verify the Azure DevOps pipeline was triggered successfully

## How It Works

```
┌─────────────────────────┐
│ Code pushed to main     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ assets-release workflow │
│ runs and completes      │
└────────────┬────────────┘
             │
             ▼ (on success)
┌─────────────────────────┐
│ trigger-onebranch       │
│ workflow activates      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Azure DevOps API call   │
│ triggers OneBranch      │
│ pipeline with:          │
│ - branch: main          │
│ - pattern: env/llm-rag  │
└─────────────────────────┘
```

## Verification

After the workflow runs successfully, you should see:
- ✓ Pipeline triggered successfully
- Run ID: [numeric ID]
- Run URL: [Azure DevOps pipeline run URL]

## Troubleshooting

If the workflow fails, check:
1. All three secrets are configured correctly
2. PAT has not expired and has correct permissions
3. Pipeline name matches exactly in Azure DevOps
4. You have access to the Azure DevOps project

For more details, see `.github/workflows/TRIGGER_ONEBRANCH_README.md`

## Security Notes

- Workflow uses explicit permissions (actions: read) for security
- Passed CodeQL security analysis with no vulnerabilities
- Secrets are handled securely and not exposed in logs
- Comprehensive error handling prevents exposure of sensitive data
