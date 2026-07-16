# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

| Finding | Package(s) | Source | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Ubuntu package inherited from the base image | Yes; `base-sbom.json` reports `3.37.2-2ubuntu0.6` | None | Base version `3.37.2-2ubuntu0.6` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Ubuntu packages inherited from the base image | Yes; `base-sbom.json` reports `7.81.0-1ubuntu1.25` | None | Base version `7.81.0-1ubuntu1.25` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Ubuntu package inherited from the base image | Yes; `base-sbom.json` reports `1.43.0-1ubuntu0.4` | None | Base version `1.43.0-1ubuntu0.4` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base env dependency; can be resolved by this image's AzureML/MLflow dependency layer | Yes; `base-sbom.json` reports `1.2.1`; retained explicit image-layer override to prevent resolver downgrade | `context/Dockerfile` | `>=1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Base image `ptca` PyTorch runtime | No; `base-vulnerabilities.json` reports `torch 2.8.0+cu126` | `context/Dockerfile` | `2.10.0` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Base image `ptca` PyTorch runtime | No; `base-vulnerabilities.json` reports `torch 2.8.0+cu126` | `context/Dockerfile` | `2.10.0` |
| 5015223 / GHSA-wf93-45jw-7689 | `pip` | Stale image-layer override of base pip in both conda envs | Yes; base image carries `pip 26.1.2`; removed the stale `pip==26.1.1` override | `context/Dockerfile` | Base version `26.1.2` |

Notes:
- Direct MCR SBOM download for the base image failed with a connection reset, so the matching resolved ACPT base SBOM previously scanned from `gatestacr.azurecr.io/sbom-base/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1` was copied here as `base-sbom.json` and `base-vulnerabilities.json` for manual verification.
- Existing pins in `context/Dockerfile` and `context/requirements.txt` were audited. The Ubuntu package findings are already covered by the resolved base image and were not re-pinned. The stale `pip==26.1.1` override was removed because the base image already carries a newer patched pip. The `msgpack` override is retained because this image installs AzureML/MLflow dependencies after the base layer and the requested finding targets the base Python environment path.
- Final image SBOM and findings are retained untracked as `sbom.json` and `vulnerabilities.json`; VCM evaluation reported the image compliant with zero non-compliant findings.
