# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

Note: MCR did not expose an SBOM referrer for the base image, so the exact base digest was mirrored to `vulnscan1779267129n14.azurecr.io/temp/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1` for SBOM generation. The generated base artifacts are `base-sbom.json` and `base-vulnerabilities.json`.

| Finding | Package(s) | Source | Base status | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-wf93-45jw-7689 | `pip` | Base image | Already covered in base at `26.1.2`; removed image-layer downgrade to `26.1.1`. | `context/Dockerfile` | `26.1.2` from base |
| USN-8509-1 | `libpython3.10-stdlib`, `python3.10`, `python3.10-minimal`, `libpython3.10-minimal` | Base image | Already covered in base at `3.10.12-1~22.04.16`; no image-layer pin needed. | None | `3.10.12-1~22.04.16` from base |
| USN-8510-1 | `tar` | Base image | Already covered in base at `1.34+dfsg-1ubuntu0.1.22.04.4`; no image-layer pin needed. | None | `1.34+dfsg-1ubuntu0.1.22.04.4` from base |
| USN-8512-1 | `gzip` | Base image | Already covered in base at `1.10-4ubuntu4.2`; no image-layer pin needed. | None | `1.10-4ubuntu4.2` from base |
| GHSA-qfhq-4f3w-5fph / CVE-2025-3001 | `torch` | Base image | Base still has vulnerable `torch 2.8.0+cu126`; image layer upgrades the CUDA PyTorch stack. | `context/Dockerfile` | `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0` |
| GHSA-vgrw-7cvw-pwgx / CVE-2025-2999 | `torch` | Base image | Base still has vulnerable `torch 2.8.0+cu126`; image layer upgrades the CUDA PyTorch stack. | `context/Dockerfile` | `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0` |
| GHSA-6v7p-g79w-8964 | `msgpack` | Base image | Already covered in base at `1.2.1`; no image-layer pin needed. | None | `1.2.1` from base |
| USN-8495-1 | `libnghttp2-14` | Base image | Already covered in base at `1.43.0-1ubuntu0.4`; no image-layer pin needed. | None | `1.43.0-1ubuntu0.4` from base |
| USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Base image | Already covered in base at `7.81.0-1ubuntu1.25`; no image-layer pin needed. | None | `7.81.0-1ubuntu1.25` from base |
| USN-8480-1 | `libsqlite3-0` | Base image | Already covered in base at `3.37.2-2ubuntu0.6`; no image-layer pin needed. | None | `3.37.2-2ubuntu0.6` from base |
| GHSA-rgxp-2hwp-jwgg / CVE-2026-25087 | `pyarrow` | This image | Base has patched `pyarrow 25.0.0`, but this image's dependency install downgraded ptca to vulnerable `20.0.0`; image layer restores the patched floor. | `context/Dockerfile` | `pyarrow>=23.0.1` |
| GHSA-537c-gmf6-5ccf | `cryptography` | This image | Base has patched `cryptography 48.0.1`, but this image's dependency install downgraded ptca to vulnerable `46.0.7`; image layer restores the patched floor. | `context/Dockerfile` | `cryptography>=48.0.1` |
