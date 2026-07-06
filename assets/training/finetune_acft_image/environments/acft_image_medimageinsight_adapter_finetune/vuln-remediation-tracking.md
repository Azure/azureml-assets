# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-medimageinsight-adapter-finetune:30`

Base image from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}`, resolved to `biweekly.202606.3` for verification.

Base SBOM note: `vcm image sbom download` found no SBOM referrer directly on MCR for the resolved base image. `base-sbom.json` in this directory is the matching generated SBOM for digest `sha256:6bce3d2f6bbafded883f22fd20986293eed13a2b4700c8fde9a12a3995835ddc`, copied from the existing ACPT base verification artifact for reviewer comparison.

| Vulnerability | Package | Source classification | Base already covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base image ships `2.14.2`, but this image's AzureML dependency layer can resolve the transitive `azureml-defaults -> azureml-inference-server-http` dependency back to `2.12.0` in `/opt/conda`. | Yes for base; no after this image's dependency layer | `context/Dockerfile` | `pydantic-settings>=2.14.2` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base image ships `1.2.1`, but this image's AzureML/MLflow dependency layer can resolve the transitive `azureml-mlflow -> mlflow-skinny` dependency back to `1.1.1` in `/opt/conda`. | Yes for base; no after this image's dependency layer | `context/Dockerfile` | `msgpack>=1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Inherited from the ACPT base image in `/opt/conda/envs/ptca`; this image also directly required the vulnerable `torch==2.8.0`. | No | `context/Dockerfile`, `context/requirements.txt` | `torch==2.10.0` from the CUDA 12.6 PyTorch index |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Inherited from the ACPT base image in `/opt/conda/envs/ptca`; this image also directly required the vulnerable `torch==2.8.0`. | No | `context/Dockerfile`, `context/requirements.txt` | `torch==2.10.0` from the CUDA 12.6 PyTorch index |

Existing pins and overrides audited:

| Pin/override | Current disposition |
| --- | --- |
| `pip==26.1.1` in `base` and `ptca` | Kept; pip is root tooling and must be updated before requirement installation. |
| `urllib3==2.7.0`, `idna==3.15`, `click==8.3.3`, `python-dotenv==1.2.2` | Kept; the base is patched, but this image's dependency layer can reintroduce lower transitive versions through AzureML/MLflow parent ranges. |
| `aiohttp>=3.14.0`, `pyarrow>=23.0.1`, `cryptography==48.0.1`, `PyJWT>=2.13.0` | Kept; existing comments document active transitive CVE remediation and there is no current base/image evidence that these overrides are stale. |
| `py-rattler` / `conda-rattler-solver` removal | Kept as cleanup of unused optional conda solver artifacts after build-time package installation. |
