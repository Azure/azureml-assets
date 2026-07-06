# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-forecasting-gpu:47`

Base images checked:

- Provided source asset: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260621.v1`
- Image-matching GPU forecasting asset: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260303.v5`

Base SBOM status: MCR did not publish an SBOM referrer for the checked base tags. The available generated `base-sbom.json` for the provided source base shows patched `msgpack==1.2.1` and `pydantic-settings==2.14.2`. The regenerated image SBOM is saved as `sbom.json`; SBOM generation logs are saved as `sbom-generate-dnn-forecasting.log`.

| Advisory | Package | Source classification | Base already covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-qfhq-4f3w-5fph | `torch` | Introduced by this image's AutoML environment; both relevant Dockerfiles installed `torch==2.8.0`. | No | `assets/training/automl/environments/ai-ml-automl/context/Dockerfile`, `assets/training/automl/environments/ai-ml-automl-dnn-forecasting-gpu/context/Dockerfile` | `torch==2.10.0` (`2.10.0+cu128` in the generated SBOM) |
| GHSA-vgrw-7cvw-pwgx | `torch` | Introduced by this image's AutoML environment; both relevant Dockerfiles installed `torch==2.8.0`. | No | `assets/training/automl/environments/ai-ml-automl/context/Dockerfile`, `assets/training/automl/environments/ai-ml-automl-dnn-forecasting-gpu/context/Dockerfile` | `torch==2.10.0` (`2.10.0+cu128` in the generated SBOM) |
| GHSA-6v7p-g79w-8964 | `msgpack` | Inherited from the provided source base image `/opt/miniconda` Python prefix; pinned in the image-matching GPU forecasting asset because that base SBOM was unavailable. | Yes for the provided source base; `base-sbom.json` has `msgpack==1.2.1`, and `sbom.json` has `msgpack==1.2.1`. | `assets/training/automl/environments/ai-ml-automl-dnn-forecasting-gpu/context/Dockerfile` | No provided-source image-layer pin; GPU forecasting asset pins `msgpack>=1.2.1`. |
| GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Inherited from the provided source base image `/opt/miniconda` Python prefix; pinned in the image-matching GPU forecasting asset because that base SBOM was unavailable. | Yes for the provided source base; `base-sbom.json` has `pydantic-settings==2.14.2`, and `sbom.json` has `pydantic-settings==2.14.2`. | `assets/training/automl/environments/ai-ml-automl-dnn-forecasting-gpu/context/Dockerfile` | No provided-source image-layer pin; GPU forecasting asset pins `pydantic-settings>=2.14.2`. |

Existing base-prefix pins in the provided source Dockerfile were audited. The final generated SBOM contains patched `PyJWT==2.13.0`, `idna==3.18`, `py-rattler==0.25.0`, and `cryptography==49.0.0`, so the provided source Dockerfile keeps only the AutoML environment `pip>=26.1` update and does not add stale base-prefix overrides.

VCM evaluation note: `vcm image vulnerabilities evaluate --sbom sbom.json` could not complete because `tvmscannerprd.azure-api.net` refused the connection. The generated SBOM verifies the requested packages are at patched versions.
