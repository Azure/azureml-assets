# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-medimageinsight-embedding:35`

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

Base SBOM note: MCR did not publish an SBOM artifact for this base image, so the base image was imported to `vulnscan1779267129n14.azurecr.io/temp/acpt-stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3` to generate `base-sbom.json` for manual comparison.

| Vulnerability | Package | Source classification | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-4xgf-cpjx-pc3j | pydantic-settings | Base image ships `2.14.2`, but this image's AzureML requirements layer can downgrade it in `/opt/conda/lib/python3.13`. | Yes for base; fixed here to prevent image-layer downgrade | `context/Dockerfile` | `pydantic-settings>=2.14.2` |
| GHSA-6v7p-g79w-8964 | msgpack | Base image ships `1.2.1`, but this image's AzureML/MLflow requirements layer can downgrade it in `/opt/conda/lib/python3.13`. | Yes for base; fixed here to prevent image-layer downgrade | `context/Dockerfile` | `msgpack>=1.2.1` |
| GHSA-vgrw-7cvw-pwgx | torch | Inherited from the ACPT PyTorch 2.8 base and also directly pinned by this image's `requirements.txt`; this image overrides the inherited CUDA PyTorch stack. | No | `context/requirements.txt` | `torch==2.10.0` |
| GHSA-qfhq-4f3w-5fph | torch | Inherited from the ACPT PyTorch 2.8 base and also directly pinned by this image's `requirements.txt`; this image overrides the inherited CUDA PyTorch stack. | No | `context/requirements.txt` | `torch==2.10.0` |

Existing Dockerfile security overrides were reviewed against the current vulnerability set and `base-sbom.json`. The stale pip workaround was removed because the resolved ACPT base already ships patched `pip 26.1.2`; the remaining overrides apply to dependencies that this image's own pip installs can resolve or downgrade.
