# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-medimageparse-3d-finetune:3`

Base image inspected: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04:latest`

| Finding | Package | Source | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| USN-8477-1 | `tar` | Base image | Yes. Base SBOM has `1.35+dfsg-3ubuntu0.2`, above required `1.35+dfsg-3ubuntu0.1`. | None | Base image version `1.35+dfsg-3ubuntu0.2` |
| USN-8480-1 | `libsqlite3-0` | Base image | Yes. Base SBOM has required `3.45.1-1ubuntu2.6`. | None | Base image version `3.45.1-1ubuntu2.6` |
| USN-8487-1 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base image | Yes. Base SBOM has `8.5.0-2ubuntu10.11`, above required `8.5.0-2ubuntu10.10`. | None | Base image version `8.5.0-2ubuntu10.11` |
| USN-8495-1 | `libnghttp2-14` | Base image | Yes. Base SBOM has required `1.59.0-1ubuntu0.4`. | None | Base image version `1.59.0-1ubuntu0.4` |
| GHSA-vgrw-7cvw-pwgx | `torch` | This image | Not applicable; `torch` is installed in this image's conda environment. | `context/Dockerfile` | `torch==2.10.0`, `torchaudio==2.10.0`, `torchvision==0.25.0` |
| GHSA-qfhq-4f3w-5fph | `torch` | This image | Not applicable; `torch` is installed in this image's conda environment. | `context/Dockerfile` | `torch==2.10.0`, `torchaudio==2.10.0`, `torchvision==0.25.0` |
| CVE-2026-44512 / GHSA-hwpq-hmq9-wj77 | `onnx` | This image | Not applicable; `onnx` is installed in this image's conda environment. | `context/requirements.txt` | `onnx==1.22.0` |

Notes:
- `base-sbom.json` and `base-vulnerabilities.json` were generated from an ACR-hosted copy of the current base image because the MCR image has no downloadable SBOM referrer.
- The stale `/opt/miniconda` `idna`/`cryptography` override was removed from the Dockerfile because the current base image already ships patched versions (`idna 3.18`, `cryptography 49.0.0`).
