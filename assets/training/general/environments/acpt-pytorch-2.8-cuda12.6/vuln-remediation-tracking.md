# Vulnerability remediation tracking

Image: `public/azureml/curated/acpt-pytorch-2.8-cuda12.6:13`

Base image from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}`, resolved to `biweekly.202606.3`.

Base SBOM note: `vcm image sbom download` found no SBOM referrer directly on MCR, so the exact base image tag was imported to `vulnscan1779267129n12.azurecr.io/sbom-base/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3` and VCM generated `base-sbom.json` there for reviewer verification.

| Vulnerability | Package | Source classification | Base already covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base image ships `2.14.2`, but this image's `requirements.txt` layer can resolve the transitive `azureml-defaults -> azureml-inference-server-http` dependency back to `2.12.0` in `/opt/conda`. | Yes for base; no after this image's dependency layer | `context/Dockerfile` | `pydantic-settings>=2.14.2` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base image ships `1.2.1`, but this image's AzureML/MLflow dependency layer can resolve the transitive `azureml-mlflow -> mlflow-skinny` dependency back to `1.1.1` in `/opt/conda`. | Yes for base; no after this image's dependency layer | `context/Dockerfile` | `msgpack>=1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Inherited from the ACPT base image in `/opt/conda/envs/ptca`; no patched `torch280` base line is available, so this image overrides the inherited CUDA PyTorch stack. | No | `context/Dockerfile` | `torch==2.10.0` from the CUDA 12.6 PyTorch index |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Inherited from the ACPT base image in `/opt/conda/envs/ptca`; no patched `torch280` base line is available, so this image overrides the inherited CUDA PyTorch stack. | No | `context/Dockerfile` | `torch==2.10.0` from the CUDA 12.6 PyTorch index |

Existing pins and overrides audited:

| Pin/override | Current disposition |
| --- | --- |
| `pip>=26.1.1` in `ptca` and `base` | Kept; pip is root tooling and not controlled by a parent Python package. |
| `setuptools>=82.0.1` | Kept; base SBOM still contains older setuptools metadata, and the existing comment documents GHSA coverage. |
| `urllib3>=2.7.0`, `idna>=3.15`, `aiohttp>=3.14.0`, `python-dotenv>=1.2.2`, `click>=8.3.3` | Kept; the base SBOM is patched, but this image installs AzureML SDK and inference dependencies with loose transitive ranges after the base layer. |
| `cryptography>=48.0.1`, `pyarrow>=23.0.1`, `PyJWT>=2.13.0`, `py-rattler>=0.24.0` | Kept; existing comments document transitive dependency CVE remediation and no current base/image evidence proves they are stale. |
