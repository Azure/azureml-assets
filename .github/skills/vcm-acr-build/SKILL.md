---
name: vcm-acr-build
description: "Build Azure ML environment Docker images on ACR using VCM. Use when: vcm build, acr build, remote build, cloud build, vcm image build, build on registry, submit build, queue build, build with SBOM, build all images, batch build, build from PR"
---

# VCM ACR Build

Builds Azure ML environment Docker images remotely on Azure Container Registry (ACR) using the VCM CLI (`vcm image build`). Builds run on ACR cloud compute — no local Docker Desktop required.

## When to Use

- Build images on ACR instead of locally (faster, no local Docker needed)
- Build images with automatic SBOM generation for vulnerability scanning
- Batch-build multiple images from a PR branch
- Submit builds before running `vcm image vulnerabilities show` scans

## Prerequisites

- Virtual environment active with VCM and `azureml-assets` tooling installed:
  ```powershell
  .venv\Scripts\Activate.ps1
  # VCM — install from the Vienna feed
  uv pip install vcm -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/
  # azureml-assets tooling (for update_assets)
  uv pip install -e scripts/azureml-assets
  ```
- Azure CLI logged in (`az login`)
- An ACR registry created and accessible. To create one:
  ```powershell
  az acr create --resource-group <rg-name> --name <acr-name> --sku Standard --location eastus
  ```
- Working directory: repo root (the `azureml-assets` clone)

## Step-by-Step Procedure

### Step 1 — Generate the build context

Build contexts must be generated from source environment folders before building. The `update_assets` tool resolves template variables like `{{latest-image-tag}}` and `{{latest-pypi-version}}`.

```powershell
python -m azureml.assets.update_assets -i <environment-folder> -o <output-folder> -r .
```

- `-i` — **comma-separated list** of source environment folders under `assets/`, e.g. `assets/training/general/environments/acpt-pytorch-2.8-cuda12.6`. For multiple images, join with commas: `"path1,path2,path3"`
- `-o` — output directory, e.g. `build_contexts`
- `-r .` — **REQUIRED** for `version: auto` assets (which is all training environments). Use `-r .` for first-time generation, or `-r empty_release_dir` (an empty directory) to force regeneration.

The resolved context appears at `<output-folder>/environment/<image-name>/context/`.

**IMPORTANT**: `update_assets` tracks which versions have been generated. On repeat runs it may skip already-generated contexts. To force regeneration:

1. **Empty release directory trick (recommended)** — delete the output folder, then point `-r` at an empty directory instead of `.`:
   ```powershell
   Remove-Item -Recurse -Force <output-folder> -ErrorAction SilentlyContinue
   New-Item -ItemType Directory -Path empty_release_dir -Force | Out-Null
   python -m azureml.assets.update_assets -i $envList -o <output-folder> -r empty_release_dir
   Remove-Item -Recurse -Force empty_release_dir
   ```
   **Why this works:** `-r .` compares against the current working tree, so only truly new versions get output. An empty release dir makes every asset appear unreleased.

2. **Manual template resolution** — copy the source `context/` directory and resolve templates yourself using known values (last resort)

### Step 2 — Verify the build context

Before submitting to ACR, confirm the context is valid:

```powershell
# Check Dockerfile exists and has no unresolved templates
Get-Content <output-folder>\environment\<image-name>\context\Dockerfile | Select-String "latest-image-tag|latest-pypi-version"
# If any matches found, templates are NOT resolved — do not submit
```

### Step 3 — Submit the build to ACR

```powershell
vcm image build `
    --registry <acr-name> `
    --repository azureml/<image-name> `
    --tag <tag> `
    --context <output-folder>\environment\<image-name>\context `
    --generate-sbom
```

**Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--registry` | ACR registry name (without `.azurecr.io`) | `myteamacr` |
| `--repository` | Image repository path (conventionally `azureml/<name>`) | `azureml/acft-grpo` |
| `--tag` | Image tag for this build | `pr-test-v1`, `pr-test-v2` |
| `--context` | Path to the resolved build context directory | `build_contexts\environment\acft-grpo\context` |
| `--generate-sbom` | Generate SBOM during build (required for later scanning) | (flag, no value) |

**IMPORTANT flags:**
- `--generate-sbom` is **required** if you plan to scan the image afterward with `vcm image vulnerabilities show`
- Without it, the scan will fail with missing SBOM data

### Step 4 — Save the build log

Redirect output to a JSON file for debugging:

```powershell
vcm image build `
    --registry <acr-name> `
    --repository azureml/<image-name> `
    --tag <tag> `
    --context <output-folder>\environment\<image-name>\context `
    --generate-sbom `
    --output vuln_reports\<image-name>-build.json
```

Or capture via PowerShell:

```powershell
$buildOutput = vcm image build --registry <acr-name> --repository azureml/<image-name> --tag <tag> --context <path> --generate-sbom 2>&1
$buildOutput | Out-File vuln_reports\<image-name>-build.json
```

### Step 5 — Verify the build succeeded

Check the build log for success:

```powershell
# Look for the final status
Get-Content vuln_reports\<image-name>-build.json | Select-Object -Last 5
# Should show "Run ID: ..." and no error messages
```

A successful build ends with ACR run ID output and no error lines. Failed builds show error details in the log.

## Batching Multiple Images

### Option A — Parallel builds across PowerShell sessions (recommended for large batches)

VCM builds run on ACR cloud compute, so multiple builds can run concurrently in separate PowerShell sessions. This is significantly faster for large batches:

```powershell
# Batch builds into groups of 3-4 per PowerShell session
# Run each batch in a separate async session for parallelism

$registry = "<acr-name>"
$tag = "pr-test-v1"
$contextBase = "build_contexts\environment"

# Session 1: batch of 3
$batch1 = @("acft-grpo", "acft-hf-nlp-gpu", "acft-rft-training")
foreach ($img in $batch1) {
    Write-Output "=== Building $img ==="
    vcm image build --registry $registry --repository "azureml/$img" --tag $tag `
        --context "$contextBase\$img\context" --generate-sbom `
        --output "vuln_reports\$img-build.json"
    Write-Output "=== Done: $img ==="
}

# Session 2: another batch of 3 (runs concurrently with session 1)
# ... same pattern with different images
```

**Parallelism guidance:**
- Run 4–6 parallel PowerShell sessions, each with 3–4 images sequentially
- ACR handles concurrent builds well since each runs as a separate ACR task
- Total wall-clock time ≈ duration of the slowest batch (not sum of all builds)

### Option B — Sequential builds in a single session (simple, but slow)

```powershell
$images = @(
    @{ Name = "acft-grpo"; Context = "build_contexts\environment\acft-grpo\context" },
    @{ Name = "acft-hf-nlp-gpu"; Context = "build_contexts\environment\acft-hf-nlp-gpu\context" }
)

$registry = "<acr-name>"
$tag = "pr-test-v1"

foreach ($img in $images) {
    Write-Output "Building $($img.Name)..."
    vcm image build `
        --registry $registry `
        --repository "azureml/$($img.Name)" `
        --tag $tag `
        --context $img.Context `
        --generate-sbom `
        --output "vuln_reports\$($img.Name)-build.json"
    Write-Output "Build complete: $($img.Name)"
}
```

**Sequential batch size guidance:**
- Each `vcm image build` waits for the build to complete before returning
- Sequential is simplest but slowest — use parallel sessions for large batches
- GPU images with large pip installs take the longest

## Image Name → Source Environment Mapping

Common training image mappings (the build context directory name may differ from the source folder):

| Image Name | Source Environment Folder |
|-----------|--------------------------|
| acft-grpo | `assets/training/finetune_acft_hf_nlp/environments/acpt-grpo` |
| acft-hf-nlp-gpu | `assets/training/finetune_acft_hf_nlp/environments/acpt` |
| acft-rft-training | `assets/training/finetune_acft_hf_nlp/environments/acpt-rft` |
| acft-transformers-image-gpu | `assets/training/finetune_acft_image/environments/acft_image_huggingface` |
| acpt-automl-image-framework-selector-gpu | `assets/training/finetune_acft_image/environments/acpt_image_framework_selector` |
| acpt-pytorch-2.2-cuda12.1 | `assets/training/general/environments/acpt-pytorch-2.2-cuda12.1` |
| acpt-pytorch-2.8-cuda12.6 | `assets/training/general/environments/acpt-pytorch-2.8-cuda12.6` |
| tensorflow-2.16-cuda11 | `assets/training/general/environments/tensorflow-2.16-cuda11` |
| tensorflow-2.16-cuda12 | `assets/training/general/environments/tensorflow-2.16-cuda12` |

## Azure Resources Setup

If you need to create a new ACR registry:

```powershell
# Create resource group (if needed)
az group create --name <rg-name> --location eastus

# Create ACR (Standard SKU is sufficient for builds)
az acr create --resource-group <rg-name> --name <acr-name> --sku Standard --location eastus

# Verify
az acr show --name <acr-name> --query "{name:name, loginServer:loginServer, sku:sku.name}" -o table
```

## Typical Durations

| Operation | Typical Duration | Examples |
|-----------|-----------------|----------|
| `update_assets` (per image) | 5–30 seconds | |
| `vcm image build` (simple ACFT image) | 10–15 minutes | acft-medimageinsight-*, acft-mmdetection |
| `vcm image build` (medium image) | 15–30 minutes | acft-grpo, acft-hf-nlp-gpu, automl images |
| `vcm image build` (large GPU image) | 30–50 minutes | tensorflow-2.16-cuda11 (~41 min), tensorflow-2.16-cuda12 (~47 min), acpt-pytorch-* |
| SBOM auto-generation (if missing) | 5–10 minutes | Added on top of scan time when `--generate-sbom` was not used |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `vcm: command not found` | VCM not installed in active venv | `uv pip install vcm -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/` |
| `can't be updated because no release directory was specified` | `-r .` omitted from `update_assets` | Add `-r .` to the command |
| Build fails with unresolved `{{latest-image-tag}}` | Template not resolved in Dockerfile | Re-run `update_assets` or manually resolve templates |
| ACR build timeout | Image too complex or network issues | Check ACR build logs; retry |
| `unauthorized` or `403` from ACR | Not logged in or no access | Run `az login` and `az acr login --name <acr-name>` |
| Build succeeds but scan fails later | `--generate-sbom` was omitted | Rebuild with `--generate-sbom` flag |
| `update_assets` skips already-generated contexts | Tool tracks versions internally | Use the empty release dir trick: delete output, `-r empty_release_dir` pointing to an empty directory |
| Windows cp1252 encoding crash during build log streaming | Unicode chars (→, —, emoji) in build output can't be encoded in Windows cp1252 | Set `$env:PYTHONUTF8 = "1"` before running `vcm` or `az` commands |

## Important Notes

### VCM builds are blocking
`vcm image build` waits for the ACR build to complete before returning. There is no `--no-wait` flag. To achieve parallelism, run multiple builds in separate PowerShell sessions (see Option A above).

### Windows encoding (PYTHONUTF8)
On Windows, always set `$env:PYTHONUTF8 = "1"` before running VCM or Azure CLI commands. Build logs and scan output may contain Unicode characters that crash the default Windows cp1252 encoding. Add this to the top of every PowerShell session:
```powershell
$env:PYTHONUTF8 = "1"
```

### SBOM auto-generation on scan
If `--generate-sbom` was not used during the build (or SBOM generation failed), `vcm image vulnerabilities show` will automatically generate an SBOM by pulling the image and running Trivy as an ACR task. This adds ~5-10 minutes to the scan but means you don't need to rebuild just for the SBOM.

### Dependency override ordering in Dockerfiles
When fixing vulnerabilities in transitive pip dependencies, the pip install that applies the override **must be the last pip command** in the Dockerfile. Subsequent pip installs (e.g., `pip install horovod[tensorflow]`) can re-resolve and downgrade the package. Use `--no-deps` to prevent re-resolution when installing on top of an existing environment:
```dockerfile
# BAD: protobuf gets re-resolved by horovod
RUN pip install --no-cache-dir 'protobuf>=5.29.6'
RUN pip install --no-cache-dir horovod[tensorflow]==0.28.1

# GOOD: override sticks because --no-deps prevents re-resolution
RUN pip install --no-cache-dir horovod[tensorflow]==0.28.1
RUN pip install --no-cache-dir --no-deps 'protobuf>=5.29.6'
```
