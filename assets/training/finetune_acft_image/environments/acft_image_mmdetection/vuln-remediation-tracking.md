# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-mmdetection-image-gpu:89`

Base image resolved from `context/Dockerfile`: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

Base SBOM: `base-sbom.json`
Final image SBOM: `sbom.json`

| Vulnerability | Package | Source classification | Base image covered? | Changed files | Patched/pinned version | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| GHSA-4xgf-cpjx-pc3j | pydantic-settings | Base image | Yes | None | 2.14.2 from base | Base SBOM shows `pydantic-settings 2.14.2`, so no image-layer pin is needed. |
| GHSA-6v7p-g79w-8964 | msgpack | Base image | Yes | None | 1.2.1 from base | Base SBOM shows `msgpack 1.2.1`, so no image-layer pin is needed. |
| GHSA-vgrw-7cvw-pwgx | torch | Base image ptca env | No | `context/Dockerfile` | 2.10.0 | Base SBOM shows `torch 2.8.0+cu126`; this image upgrades the ptca CUDA PyTorch stack. |
| GHSA-qfhq-4f3w-5fph | torch | Base image ptca env | No | `context/Dockerfile` | 2.10.0 | Base SBOM shows `torch 2.8.0+cu126`; this image upgrades the ptca CUDA PyTorch stack. |
