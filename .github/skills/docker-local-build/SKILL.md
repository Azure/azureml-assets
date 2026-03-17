---
name: docker-local-build
description: "Build Azure ML environment Docker images locally. Use when: build docker image, build image locally, docker build, generate build context, update_assets, build environment, local docker build, build from Dockerfile, build context, build azureml image"
---

# Docker Local Build

Builds Azure ML environment Docker images locally by first generating a resolved build context via `update_assets`, then running `docker build` from the generated output.

## When to Use

- Build an Azure ML environment image locally before scanning or testing
- Generate a resolved build context (templates expanded, files copied) from an `assets/` environment folder
- Reproduce the exact build context that CI/CD would use

## Critical Rules

- **NEVER** run `docker build` directly from the `assets/` source tree
- **NEVER** manually copy files into a context directory
- **ALWAYS** use `python -m azureml.assets.update_assets` to generate the build context first
- **ALWAYS** pass `-r .` — most environments use `version: auto` in `asset.yaml`, and omitting `-r .` causes an immediate failure with: `can't be updated because no release directory was specified`

## Prerequisites

- Docker Desktop running locally
- Virtual environment active with `azureml-assets` tooling installed:
  ```powershell
  # Check / create (uv preferred)
  if (!(Test-Path ".venv\Scripts\activate")) { uv venv .venv --python 3.11 }
  .venv\Scripts\Activate.ps1
  uv pip install -e scripts/azureml-assets
  ```
- Working directory: repo root (`c:\Users\hnamburi\azureml-assets` or equivalent)

## Step-by-Step Procedure

### Step 1 — Locate the environment folder

Find the environment under `assets/` by image name or environment name:
```powershell
# Example: find all asset.yaml files for a named environment
Get-ChildItem -Path assets/ -Recurse -Filter asset.yaml | Select-String -Pattern "<image-name>"
```

Note the folder path, e.g. `assets/training/general/environments/acpt-pytorch-2.8-cuda12.6/`.

### Step 2 — Check for `version: auto`

```powershell
Get-Content <environment-folder>/asset.yaml | Select-String "version"
```

If the output contains `version: auto`, the `-r .` flag is **required** in the next step.

### Step 3 — Generate the build context

```powershell
python -m azureml.assets.update_assets -i <environment-folder> -o <output-folder> -r .
```

- `<environment-folder>` — source folder containing `asset.yaml`, e.g. `assets/training/general/environments/acpt-pytorch-2.8-cuda12.6`
- `<output-folder>` — destination for the resolved context, e.g. `build_output`
- `-r .` — always include; required for `version: auto` assets, harmless otherwise

The command produces a directory tree under `<output-folder>/environment/<image-name>/context/` containing the fully resolved `Dockerfile` and all support files.

### Step 4 — Build the image

Run the build from the generated context directory. Always tee output to a log file for progress monitoring:

```powershell
docker build -t <image-name> <output-folder>/environment/<image-name>/context 2>&1 |
    Tee-Object -FilePath build_logs/<image-name>-build.log
```

Or to run as a background process (recommended for GPU / large images):
```powershell
Start-Process -NoNewWindow powershell -ArgumentList `
  "-Command docker build -t <image-name> <output-folder>/environment/<image-name>/context 2>&1 | Tee-Object -FilePath build_logs/<image-name>-build.log"
```

### Step 5 — Verify the build succeeded

```powershell
# Check last lines of log for success marker
Get-Content build_logs/<image-name>-build.log | Select-Object -Last 20

# Confirm image exists
docker images <image-name> --format "{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}"
```

A successful build ends with a line containing `Successfully built` or the final `DONE` step from BuildKit.

## Concrete Example

```powershell
# 1. Activate venv (if not already active)
.venv\Scripts\Activate.ps1

# 2. Generate build context
python -m azureml.assets.update_assets `
    -i assets/training/general/environments/acpt-pytorch-2.8-cuda12.6 `
    -o build_output `
    -r .

# 3. Build image (tee to log)
docker build -t acpt-pytorch-2.8-cuda12.6 `
    build_output/environment/acpt-pytorch-2.8-cuda12.6/context `
    2>&1 | Tee-Object -FilePath build_logs/acpt-pytorch-2.8-cuda12.6-build.log

# 4. Verify
docker images acpt-pytorch-2.8-cuda12.6 --format "{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}"
```

## Batching Multiple Images

When building multiple images, kick off all `update_assets` runs first (fast), then build images sequentially (Docker builds share the same daemon):

```powershell
# Generate all contexts first
python -m azureml.assets.update_assets -i assets/.../image-a -o build_output -r .
python -m azureml.assets.update_assets -i assets/.../image-b -o build_output -r .

# Then build sequentially
docker build -t image-a build_output/environment/image-a/context 2>&1 | Tee-Object build_logs/image-a-build.log
# Wait for completion, then:
docker build -t image-b build_output/environment/image-b/context 2>&1 | Tee-Object build_logs/image-b-build.log
```

## Typical Durations

| Operation | Typical Duration |
|-----------|-----------------|
| `update_assets` | 5–30 seconds |
| `docker build` (CPU-only) | 5–20 minutes |
| `docker build` (GPU image) | 20–90 minutes |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `can't be updated because no release directory was specified` | `-r .` was omitted | Add `-r .` to the `update_assets` command |
| `python: No module named azureml.assets` | venv not active or tooling not installed | Activate venv and run `uv pip install -e scripts/azureml-assets` |
| `docker: command not found` | Docker Desktop not running | Start Docker Desktop |
| `ERROR [internal] load metadata for ...` | Base image pull failed | Check network / VPN / Docker login |
| Build log shows no `Successfully built` | Build failed mid-way | Search log for `ERROR` lines to find the failing layer |
