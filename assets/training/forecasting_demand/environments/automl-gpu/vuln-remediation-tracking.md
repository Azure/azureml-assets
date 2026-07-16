# Vulnerability remediation tracking

Image: `public/azureml/curated/automl-gpu:73`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260621.v1` (resolved from existing base SBOM generation log). `vcm image sbom download` was retried for this base tag and failed while resolving the MCR image digest with `ConnectionResetError(10054)`, so base package comparison is recorded as unavailable.

Verification: built `vulnscan1779267129n4.azurecr.io/public/azureml/curated/automl-gpu:test-fix`, generated and downloaded `sbom.json`, wrote detailed findings to `vulnerabilities.json`, and `vcm image vulnerabilities evaluate --sbom sbom.json` reported compliant with 0 non-compliant findings.

| Finding | Package(s) | Source | Base already covered? | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014620 / GHSA-qfhq-4f3w-5fph | torch | This image, via `azureml-contrib-automl-dnn-forecasting` dependency chain | No base SBOM available; installed package path is in this image's `/azureml-envs/automl` conda env | `context/Dockerfile` | `torch==2.10.0` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | torch | This image, via `azureml-contrib-automl-dnn-forecasting` dependency chain | No base SBOM available; installed package path is in this image's `/azureml-envs/automl` conda env | `context/Dockerfile` | `torch==2.10.0` |
| 6035947 / USN-8495-1 | libnghttp2-14 | Base image OS package, upgraded in this image layer | Unknown; base SBOM unavailable and current image scan reported vulnerable version | `context/Dockerfile` | `1.43.0-1ubuntu0.4` or newer from Ubuntu security repo |
| 6035933 / USN-8487-1 | curl, libcurl3-gnutls, libcurl4 | Base image OS packages, upgraded in this image layer | Unknown; base SBOM unavailable and current image scan reported vulnerable versions | `context/Dockerfile` | `7.81.0-1ubuntu1.25` or newer from Ubuntu security repo |
| 6035919 / USN-8480-1 | libsqlite3-0 | Base image OS package, upgraded in this image layer | Unknown; base SBOM unavailable and current image scan reported vulnerable version | `context/Dockerfile` | `3.37.2-2ubuntu0.6` or newer from Ubuntu security repo |
| 6035912 / USN-8477-1 | tar | Base image OS package, upgraded in this image layer | Unknown; base SBOM unavailable and current image scan reported vulnerable version | `context/Dockerfile` | `1.34+dfsg-1ubuntu0.1.22.04.3` or newer from Ubuntu security repo |
| 5015223 / GHSA-wf93-45jw-7689 | pip | This image conda env and base miniconda pip install path | No base SBOM available; generated image SBOM reported `pip 26.1.1` in `/azureml-envs/automl` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `pip 26.1.2` or newer; stale `pip-26.1.1.dist-info` metadata removed |
