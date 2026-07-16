# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-transformers-image-gpu:85`

Base image resolved from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

| Finding | Package(s) | Source | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Ubuntu package inherited from the base image; explicitly upgraded in this image layer so the rendered build stays patched | Base SBOM has `3.37.2-2ubuntu0.6` | `context/Dockerfile` | `3.37.2-2ubuntu0.6` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Ubuntu packages inherited from the base image; explicitly upgraded in this image layer so the rendered build stays patched | Base SBOM has `7.81.0-1ubuntu1.25` | `context/Dockerfile` | `7.81.0-1ubuntu1.25` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Ubuntu package inherited from the base image; explicitly upgraded in this image layer so the rendered build stays patched | Base SBOM has `1.43.0-1ubuntu0.4` | `context/Dockerfile` | `1.43.0-1ubuntu0.4` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Transitive dependency of `mlflow`/`mlflow-skinny` introduced by this image requirements | Base SBOM has `1.2.1`; final image retains `1.2.1` in both detected Python environments | `context/requirements.txt` | `>=1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Base image PyTorch runtime in `/opt/conda/envs/ptca` | Not fixed in this image; inherited unchanged from base image and VCM evaluation is compliant | None | Requires future PyTorch base image `>=2.9.1` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Base image PyTorch runtime in `/opt/conda/envs/ptca` | Not fixed in this image; inherited unchanged from base image and VCM evaluation is compliant | None | Requires future PyTorch base image `>=2.10.0` |
| 5015223 / GHSA-wf93-45jw-7689 | `pip` | Existing root tool override in both conda environments | Base SBOM has `26.1.2`; final image has `26.1.2` in base and ptca environments | `context/Dockerfile` | `26.1.2` |

Notes:
- Direct SBOM download from `mcr.microsoft.com` reset during the base scan attempt. The retained `base-sbom.json` and `base-vulnerabilities.json` were copied from an existing scan of the same resolved base tag (`biweekly.202607.1`) and are kept untracked for manual verification.
- Final image artifacts are retained untracked as `sbom.json` and `vulnerabilities.json`. VCM evaluation for `vulnscan1779267129n1.azurecr.io/public/azureml/curated/acft-transformers-image-gpu:test-fix` reported compliant.
- Existing Dockerfile and requirement pins were audited; the stale `pip=26.1.1` override was updated to `26.1.2`, and existing active transitive dependency overrides were retained.
