# Vulnerability remediation tracking

Image: `public/azureml/curated/automl-gpu:73`

Base image from `context\Dockerfile`: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}`; resolved by the azureml-assets pinning utility to `mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260621.v1` on 2026-07-06.

Base SBOM download from MCR: attempted with `vcm image sbom download --registry mcr.microsoft.com --repository azureml/openmpi5.0-cuda12.4-ubuntu22.04 --tag 20260621.v1 --output base-sbom.json`; MCR returned no SBOM artifacts for this base image. For manual comparison, the base image was imported into `vulnscan1779267129n7.azurecr.io/base-sbom/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260621.v1`, an SBOM was generated there, and `base-sbom.json` was downloaded from that artifact.

| Vulnerability | Package | Source | Base covered? | Changed files | Patched version |
| --- | --- | --- | --- | --- | --- |
| GHSA-qfhq-4f3w-5fph | `torch` | This image; installed into `azureml-envs\automl` by the AutoML DNN forecasting Python dependency stack. | Not covered by the base image; `base-sbom.json` contains no `torch` package and the vulnerable install path is the environment created in this image. | `context\Dockerfile` | `torch==2.10.0` |
| GHSA-vgrw-7cvw-pwgx | `torch` | This image; installed into `azureml-envs\automl` by the AutoML DNN forecasting Python dependency stack. | Not covered by the base image; `base-sbom.json` contains no `torch` package and the vulnerable install path is the environment created in this image. | `context\Dockerfile` | `torch==2.10.0` |

Existing pins and overrides were reviewed against the available scan input. The current supplied image scan only flags `torch`; no existing Dockerfile overrides were removed because the pre-scan SBOM was unavailable and the base SBOM artifact could not be downloaded for confirmation.
