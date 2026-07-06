# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-mmtracking-video-gpu:80`

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

Base SBOM: `base-sbom.json`
Image SBOM: `sbom.json`

| ID | Package | Source | Base covered? | Changed files | Patched/pinned version | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 5014620 / GHSA-qfhq-4f3w-5fph | torch | Base image ptca env inherited by this image | No; base SBOM has `torch 2.8.0+cu126` | `context/Dockerfile` | `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0` | Pinned in ptca env with the CUDA 12.6 PyTorch wheel index; torchvision and torchaudio are upgraded with torch for compatibility. |
| 5014615 / GHSA-vgrw-7cvw-pwgx | torch | Base image ptca env inherited by this image | No; base SBOM has `torch 2.8.0+cu126` | `context/Dockerfile` | `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0` | Same torch override covers both advisories. |
| 5014292 / GHSA-4xgf-cpjx-pc3j | pydantic-settings | Base image base conda env | Yes; base SBOM has `pydantic-settings 2.14.2` | None | `2.14.2` from base image | Removed the redundant image-level pin because the current base already ships the required version. |
| 5014304 / GHSA-6v7p-g79w-8964 | msgpack | Base image base conda env | Yes; base SBOM has `msgpack 1.2.1` | None | `1.2.1` from base image | No image-level pin needed because the current base already ships the required version. |

Existing remediation overrides were re-checked against the current base SBOM. Redundant base-env overrides for `python-dotenv`, `pip`, `py-rattler`, `urllib3`, `aiohttp`, `click`, `idna`, `PyJWT`, and `pydantic-settings` were removed because the current base SBOM already contains patched versions and the base SBOM vulnerability evaluation is compliant.
