---
name: build-context
description: "Generate resolved Docker build contexts for Azure ML environment images. Use when: generate build context, resolve templates, update_assets, create build context, prepare build, resolve Dockerfile templates, latest-image-tag, latest-pypi-version, regenerate context, batch context generation"
---

# Build Context Generation

Generates fully resolved Docker build contexts from Azure ML environment source folders. Resolves template variables (`{{latest-image-tag}}`, `{{latest-pypi-version}}`) into concrete values so the Dockerfile and supporting files are ready for `docker build` or `vcm image build`.

## When to Use

- Before building any Azure ML environment image (locally or on ACR)
- After editing source Dockerfiles/requirements and needing a fresh build context
- When `update_assets` refuses to regenerate (version tracking issue)
- To batch-generate contexts for multiple images at once

## Critical Rules

- **NEVER** run `docker build` or `vcm image build` directly from the `assets/` source tree — templates like `{{latest-image-tag}}` will cause build failures
- **ALWAYS** verify no unresolved `{{...}}` templates remain before building
- **ALWAYS** pass `-r` to `update_assets` — all training environments use `version: auto`. Use `-r .` for first-time generation, or `-r empty_release_dir` (pointing to an empty directory) to force regeneration of all contexts

## Prerequisites

- Virtual environment active with `azureml-assets` tooling installed:
  ```powershell
  .venv\Scripts\Activate.ps1
  uv pip install -e scripts/azureml-assets
  ```
- Working directory: repo root (the `azureml-assets` clone)

## Method 1 — `update_assets` (Preferred)

The official tool that resolves all templates by querying PyPI feeds and MCR tags.

### Single image

```powershell
python -m azureml.assets.update_assets `
    -i <environment-folder> `
    -o <output-folder> `
    -r .
```

Example:
```powershell
python -m azureml.assets.update_assets `
    -i assets/training/general/environments/acpt-pytorch-2.8-cuda12.6 `
    -o build_contexts `
    -r .
```

Output appears at: `build_contexts/environment/acpt-pytorch-2.8-cuda12.6/context/`

### Multiple images at once

**IMPORTANT**: The `-i` flag accepts a **comma-separated list** of directories. Do NOT use multiple `-i` flags or PowerShell splatting — use a single `-i` with comma-separated paths:

```powershell
$envList = @(
    "assets/training/general/environments/acpt-pytorch-2.8-cuda12.6",
    "assets/training/general/environments/tensorflow-2.16-cuda12",
    "assets/training/finetune_acft_hf_nlp/environments/acpt-grpo"
) -join ","

python -m azureml.assets.update_assets -i $envList -o build_contexts -r .
```

### All 19 training images (from vuln fix PR)

```powershell
$envList = @(
    "assets/training/finetune_acft_hf_nlp/environments/acpt-grpo",
    "assets/training/finetune_acft_hf_nlp/environments/acpt",
    "assets/training/finetune_acft_hf_nlp/environments/acpt-rft",
    "assets/training/finetune_acft_image/environments/acft_image_medimageinsight_adapter_finetune",
    "assets/training/finetune_acft_image/environments/acft_image_medimageinsight_embedding_generator",
    "assets/training/finetune_acft_image/environments/acft_image_medimageinsight_embedding",
    "assets/training/finetune_acft_image/environments/acft_image_medimageparse_finetune",
    "assets/training/finetune_acft_image/environments/acft_image_mmdetection",
    "assets/training/finetune_acft_image/environments/acft_video_mmtracking",
    "assets/training/finetune_acft_multimodal/environments/acpt_multimodal",
    "assets/training/finetune_acft_image/environments/acft_image_huggingface",
    "assets/training/finetune_acft_image/environments/acpt_image_framework_selector",
    "assets/training/general/environments/acpt-pytorch-2.2-cuda12.1",
    "assets/training/general/environments/acpt-pytorch-2.8-cuda12.6",
    "assets/training/automl/environments/ai-ml-automl-dnn-text-gpu",
    "assets/training/automl/environments/ai-ml-automl-dnn-text-gpu-ptca",
    "assets/training/automl/environments/ai-ml-automl-dnn-vision-gpu",
    "assets/training/general/environments/tensorflow-2.16-cuda11",
    "assets/training/general/environments/tensorflow-2.16-cuda12"
) -join ","

python -m azureml.assets.update_assets -i $envList -o build_contexts -r .
```

### Known limitation: version tracking

`update_assets` tracks which asset versions have been generated. On repeat runs after source edits, it may **skip** images it considers already up-to-date. There is no `--force` flag.

**Workarounds (in order of preference):**

1. **Empty release directory trick (recommended)** — point `-r` at an empty directory instead of `.` so all assets appear "new" and are regenerated:
   ```powershell
   # Delete prior output so it starts fresh
   Remove-Item -Recurse -Force build_contexts -ErrorAction SilentlyContinue

   # Create a temporary empty release directory
   New-Item -ItemType Directory -Path empty_release_dir -Force | Out-Null

   # Run with -r pointing to the empty dir (not ".")
   python -m azureml.assets.update_assets -i $envList -o build_contexts -r empty_release_dir

   # Cleanup
   Remove-Item -Recurse -Force empty_release_dir
   ```
   **Why this works:** `-r .` compares against the current working tree (same branch), so only truly new versions get output. Pointing `-r` at an empty directory makes every asset appear unreleased, so all are regenerated.

2. **Fresh output directory** — use a new output directory name (e.g., `build_contexts_v2`). This alone is NOT sufficient if `-r .` still considers versions current; combine with the empty release dir trick above.

3. Use Method 2 (manual resolution) below as a last resort.

## Method 2 — Manual Template Resolution (Fallback)

When `update_assets` refuses to regenerate, copy source contexts and resolve templates directly. This is a reliable fallback that always works.

### Step 1 — Identify template values

Check prior build logs or query the values:

```powershell
# Find the base image tag from a prior build log
Select-String "biweekly" vuln_reports\<image-name>-build.json | Select-Object -First 1

# Or query MCR for latest tag matching the pattern
# The pattern is typically: biweekly.YYYYMM.N (e.g., biweekly.202603.1)
```

### Step 2 — Run the resolution script

```python
import os, re, shutil

# === CONFIGURE THESE VALUES ===
# Get IMAGE_TAG from prior build logs or MCR query
IMAGE_TAG = 'biweekly.202603.1'

# Get PYPI_VERSIONS from prior build logs, pip index, or update_assets output
# These are internal packages on the Vienna PyPI feed — not public PyPI
PYPI_VERSIONS = {
    'azureml-acft-common-components': '0.0.87',
    'azureml-acft-contrib-hf-nlp': '0.0.87',
    'azureml-acft-accelerator': '0.0.87',
    'azureml-acft-image-components': '0.0.87',
    'azureml_acft_common_components': '0.0.87',
    'azureml_acft_multimodal_components': '0.0.87',
    'azureml-evaluate-mlflow': '0.0.87',
    'azureml-metrics': '0.0.87',
    'azureml-core': '1.62.0',
    'azureml-defaults': '1.62.0',
    'azureml-mlflow': '1.62.0.post1',
    'azureml-telemetry': '1.62.0',
    'azureml-dataset-runtime': '1.62.0',
    'azureml-contrib-services': '1.62.0',
    'azureml-responsibleai': '1.62.0',
    'azureml-interpret': '1.62.0',
    'azureml-automl-core': '1.62.0',
    'azureml-automl-runtime': '1.62.0',
    'azureml-automl-dnn-nlp': '1.62.0',
    'azureml-automl-dnn-vision': '1.62.0',
    'azureml-train-automl-client': '1.62.0',
    'azureml-train-automl-runtime': '1.62.0',
    'azure-ml': '1.62.0',
    'azure-ml-component': '1.62.0',
    'mltable': '1.62.0',
}
# === END CONFIGURATION ===

# Image name → source context directory
IMAGES = [
    ('assets/training/finetune_acft_hf_nlp/environments/acpt-grpo', 'acft-grpo'),
    ('assets/training/finetune_acft_hf_nlp/environments/acpt', 'acft-hf-nlp-gpu'),
    ('assets/training/finetune_acft_hf_nlp/environments/acpt-rft', 'acft-rft-training'),
    ('assets/training/finetune_acft_image/environments/acft_image_medimageinsight_adapter_finetune', 'acft-medimageinsight-adapter-finetune'),
    ('assets/training/finetune_acft_image/environments/acft_image_medimageinsight_embedding_generator', 'acft-medimageinsight-embedding-generator'),
    ('assets/training/finetune_acft_image/environments/acft_image_medimageinsight_embedding', 'acft-medimageinsight-embedding'),
    ('assets/training/finetune_acft_image/environments/acft_image_medimageparse_finetune', 'acft-medimageparse-finetune'),
    ('assets/training/finetune_acft_image/environments/acft_image_mmdetection', 'acft-mmdetection-image-gpu'),
    ('assets/training/finetune_acft_image/environments/acft_video_mmtracking', 'acft-mmtracking-video-gpu'),
    ('assets/training/finetune_acft_multimodal/environments/acpt_multimodal', 'acft-multimodal-gpu'),
    ('assets/training/finetune_acft_image/environments/acft_image_huggingface', 'acft-transformers-image-gpu'),
    ('assets/training/finetune_acft_image/environments/acpt_image_framework_selector', 'acpt-automl-image-framework-selector-gpu'),
    ('assets/training/general/environments/acpt-pytorch-2.2-cuda12.1', 'acpt-pytorch-2.2-cuda12.1'),
    ('assets/training/general/environments/acpt-pytorch-2.8-cuda12.6', 'acpt-pytorch-2.8-cuda12.6'),
    ('assets/training/automl/environments/ai-ml-automl-dnn-text-gpu', 'ai-ml-automl-dnn-text-gpu'),
    ('assets/training/automl/environments/ai-ml-automl-dnn-text-gpu-ptca', 'ai-ml-automl-dnn-text-gpu-ptca'),
    ('assets/training/automl/environments/ai-ml-automl-dnn-vision-gpu', 'ai-ml-automl-dnn-vision-gpu'),
    ('assets/training/general/environments/tensorflow-2.16-cuda11', 'tensorflow-2.16-cuda11'),
    ('assets/training/general/environments/tensorflow-2.16-cuda12', 'tensorflow-2.16-cuda12'),
]

def resolve_templates(content):
    # Resolve {{latest-image-tag:...}} with regex filter
    content = re.sub(r'\{\{latest-image-tag:[^}]+\}\}', IMAGE_TAG, content)
    # Resolve {{latest-image-tag}} without regex filter
    content = content.replace('{{latest-image-tag}}', IMAGE_TAG)
    # Resolve {{latest-pypi-version}} per package
    for pkg, ver in PYPI_VERSIONS.items():
        content = content.replace(f'{pkg}=={{{{latest-pypi-version}}}}', f'{pkg}=={ver}')
        content = content.replace(f'{pkg}~={{{{latest-pypi-version}}}}', f'{pkg}~={ver}')
    return content

out_base = 'build_contexts/environment'
for src_dir, image_name in IMAGES:
    ctx_src = os.path.join(src_dir, 'context')
    ctx_dst = os.path.join(out_base, image_name, 'context')
    os.makedirs(ctx_dst, exist_ok=True)
    for fname in os.listdir(ctx_src):
        src_file = os.path.join(ctx_src, fname)
        dst_file = os.path.join(ctx_dst, fname)
        if os.path.isfile(src_file):
            with open(src_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            with open(dst_file, 'w', encoding='utf-8') as f:
                f.write(resolve_templates(content))
        elif os.path.isdir(src_file):
            if os.path.exists(dst_file):
                shutil.rmtree(dst_file)
            shutil.copytree(src_file, dst_file)
    print(f'Generated: {image_name}')

print(f'\nTotal: {len(IMAGES)} contexts generated')
```

### Step 3 — Verify no unresolved templates remain

```powershell
$files = Get-ChildItem -Path "build_contexts\environment" -Recurse -File
$unresolved = $files | Select-String "latest-image-tag|latest-pypi-version"
if ($unresolved) {
    Write-Output "UNRESOLVED TEMPLATES FOUND:"
    $unresolved | ForEach-Object { "$($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
} else {
    Write-Output "All templates resolved successfully!"
}
```

If unresolved templates are found, add the missing package versions to `PYPI_VERSIONS` and re-run.

## Finding Template Values

### Base image tags

```powershell
# From prior build logs
Select-String "biweekly" vuln_reports\*-build.json | Select-Object -First 1

# From MCR (requires network)
# Pattern: biweekly.YYYYMM.N (e.g., biweekly.202603.1)
```

### PyPI package versions (internal packages)

These are Azure ML internal packages hosted on the Vienna PyPI feed, not public PyPI. Values come from:

1. **Prior `update_assets` output** — the first successful run resolves all versions
2. **Prior build logs** — search for the package name in build output
3. **Vienna feed query:**
   ```powershell
   pip index versions <package-name> -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/
   ```

### Package version families

Most azureml-* packages share a version. Common families:
- **azureml-core family** (`1.62.0`): azureml-core, azureml-defaults, azureml-telemetry, azureml-dataset-runtime, azureml-contrib-services, azureml-interpret, azureml-responsibleai, azure-ml, azure-ml-component, mltable
- **azureml-automl family** (`1.62.0`): azureml-automl-core, azureml-automl-runtime, azureml-automl-dnn-nlp, azureml-automl-dnn-vision, azureml-train-automl-client, azureml-train-automl-runtime
- **azureml-acft family** (`0.0.87`): azureml-acft-common-components, azureml-acft-contrib-hf-nlp, azureml-acft-accelerator, azureml-acft-image-components, azureml-evaluate-mlflow, azureml-metrics

**Note:** These version numbers change with each release cycle. Always verify against the latest build or PyPI feed before using.

## Image Name → Source Folder Mapping

| Image Name | Source Environment Folder |
|-----------|--------------------------|
| acft-grpo | `assets/training/finetune_acft_hf_nlp/environments/acpt-grpo` |
| acft-hf-nlp-gpu | `assets/training/finetune_acft_hf_nlp/environments/acpt` |
| acft-rft-training | `assets/training/finetune_acft_hf_nlp/environments/acpt-rft` |
| acft-medimageinsight-adapter-finetune | `assets/training/finetune_acft_image/environments/acft_image_medimageinsight_adapter_finetune` |
| acft-medimageinsight-embedding-generator | `assets/training/finetune_acft_image/environments/acft_image_medimageinsight_embedding_generator` |
| acft-medimageinsight-embedding | `assets/training/finetune_acft_image/environments/acft_image_medimageinsight_embedding` |
| acft-medimageparse-finetune | `assets/training/finetune_acft_image/environments/acft_image_medimageparse_finetune` |
| acft-mmdetection-image-gpu | `assets/training/finetune_acft_image/environments/acft_image_mmdetection` |
| acft-mmtracking-video-gpu | `assets/training/finetune_acft_image/environments/acft_video_mmtracking` |
| acft-multimodal-gpu | `assets/training/finetune_acft_multimodal/environments/acpt_multimodal` |
| acft-transformers-image-gpu | `assets/training/finetune_acft_image/environments/acft_image_huggingface` |
| acpt-automl-image-framework-selector-gpu | `assets/training/finetune_acft_image/environments/acpt_image_framework_selector` |
| acpt-pytorch-2.2-cuda12.1 | `assets/training/general/environments/acpt-pytorch-2.2-cuda12.1` |
| acpt-pytorch-2.8-cuda12.6 | `assets/training/general/environments/acpt-pytorch-2.8-cuda12.6` |
| ai-ml-automl-dnn-text-gpu | `assets/training/automl/environments/ai-ml-automl-dnn-text-gpu` |
| ai-ml-automl-dnn-text-gpu-ptca | `assets/training/automl/environments/ai-ml-automl-dnn-text-gpu-ptca` |
| ai-ml-automl-dnn-vision-gpu | `assets/training/automl/environments/ai-ml-automl-dnn-vision-gpu` |
| tensorflow-2.16-cuda11 | `assets/training/general/environments/tensorflow-2.16-cuda11` |
| tensorflow-2.16-cuda12 | `assets/training/general/environments/tensorflow-2.16-cuda12` |

## Output Structure

```
build_contexts/
└── environment/
    ├── acft-grpo/
    │   └── context/
    │       ├── Dockerfile          ← templates resolved
    │       └── requirements.txt    ← templates resolved
    ├── acft-hf-nlp-gpu/
    │   └── context/
    │       ├── Dockerfile
    │       └── requirements.txt
    └── ...
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `can't be updated because no release directory was specified` | `-r .` omitted | Add `-r .` to `update_assets` |
| `python: No module named azureml.assets` | Tooling not installed | `uv pip install -e scripts/azureml-assets` |
| `update_assets` generates 0 of N contexts | Version tracking thinks they're current | Use the empty release dir trick: `-r empty_release_dir` pointing to an empty directory (see Method 1 workarounds above) |
| Unresolved `{{latest-pypi-version}}` after Method 2 | Package not in `PYPI_VERSIONS` dict | Add missing package with correct version |
| Unresolved `{{latest-image-tag}}` after Method 2 | Regex pattern variant not matched | Check the exact pattern in Dockerfile and update the regex substitution |
