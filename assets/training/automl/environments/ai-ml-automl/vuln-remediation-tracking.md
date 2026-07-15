# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-vision-gpu:51`

Resolved image asset: `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu`

The prompt-provided asset path points to `ai-ml-automl`, but the vulnerable install paths and repository asset metadata map to the sibling `ai-ml-automl-dnn-vision-gpu` asset. Remediation changes were made in that image's Dockerfile.

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Base-image OS package still vulnerable in inherited layer | No | `ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | reinstall/upgrade to Ubuntu patched package `3.37.2-2ubuntu0.6` or newer |
| 6035912 / USN-8477-1 | `tar` | Base-image OS package | Yes; base SBOM has `1.34+dfsg-1ubuntu0.1.22.04.3` | None | No local override |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Base image provides `ptca` torch; image dependency installs torch into `azureml-automl-dnn-vision-gpu` env | No | `ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | `torch>=2.10.0` in both envs |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Base image provides `ptca` torch; image dependency installs torch into `azureml-automl-dnn-vision-gpu` env | No | `ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | `torch>=2.10.0` in both envs |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base-image Python package in `/opt/conda` | Yes; base SBOM has `1.2.1` | None | No local override |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base-image Python package in `/opt/conda` | Yes; base SBOM has `2.14.2` | None | No local override |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base-image OS package still vulnerable in inherited layer | No | `ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | reinstall/upgrade to Ubuntu patched package `1.43.0-1ubuntu0.4` or newer |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Base-image OS packages still vulnerable in inherited layer | No | `ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | reinstall/upgrade to Ubuntu patched package `7.81.0-1ubuntu1.25` or newer |

Base SBOM note: direct SBOM download from MCR returned no attached SBOM artifacts. The base image was imported into the provided ACR for VCM SBOM generation so `base-sbom.json` and `base-vulnerabilities.json` can be retained with the post-build `sbom.json` for manual review.
