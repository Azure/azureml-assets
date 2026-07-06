# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-hf-nlp-gpu:124`

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

Note: `vcm image sbom download` found no SBOM artifact for the MCR base image, so the resolved base digest was imported into `vulnscan1779267129n8.azurecr.io/public/azureml/curated/acft-hf-nlp-gpu-base:biweekly.202606.3` and scanned there. `base-sbom.json` and `sbom.json` are retained next to this file for manual review.

| ID | Package | Source | Base coverage | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base image `/opt/conda` | Already covered: base ships `msgpack 1.2.1` | None | Base `1.2.1` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base image `/opt/conda` | Already covered: base ships `pydantic-settings 2.14.2` | `context/Dockerfile` removes stale override | Base `2.14.2` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Base image `ptca` env, still vulnerable | Not covered: base ships `torch 2.8.0+cu126` | `context/Dockerfile` | `torch 2.10.0+cu126` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Base image `ptca` env, still vulnerable | Not covered: base ships `torch 2.8.0+cu126` | `context/Dockerfile` | `torch 2.10.0+cu126` |

Existing Dockerfile overrides were audited against the resolved base image. Base-env overrides for packages now patched in the base image were removed rather than re-pinned in this image. The final `test-fix` image SBOM shows `torch 2.10.0+cu126`, `msgpack 1.2.1`, and `pydantic-settings 2.14.2`.
