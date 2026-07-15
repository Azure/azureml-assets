# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

| Finding | Package(s) | Source | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035961 / USN-8509-1 | `libpython3.10-stdlib`, `python3.10`, `python3.10-minimal`, `libpython3.10-minimal` | Ubuntu packages inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `3.10.12-1~22.04.16` | `context/Dockerfile` | `3.10.12-1~22.04.16` |
| 6035957 / USN-8510-1 | `tar` | Ubuntu package inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `1.34+dfsg-1ubuntu0.1.22.04.4` | `context/Dockerfile` | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035952 / USN-8512-1 | `gzip` | Ubuntu package inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `1.10-4ubuntu4.2` | `context/Dockerfile` | `1.10-4ubuntu4.2` |
| 6035912 / USN-8477-1 | `tar` | Ubuntu package inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `1.34+dfsg-1ubuntu0.1.22.04.4` | `context/Dockerfile` | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Ubuntu package inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `3.37.2-2ubuntu0.6` | `context/Dockerfile` | `3.37.2-2ubuntu0.6` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Ubuntu packages inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `7.81.0-1ubuntu1.25` | `context/Dockerfile` | `7.81.0-1ubuntu1.25` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Ubuntu package inherited from the base image and upgraded in this image layer | Base SBOM did not report this finding; image SBOM now has `1.43.0-1ubuntu0.4` | `context/Dockerfile` | `1.43.0-1ubuntu0.4` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Introduced by this image through AzureML inference dependencies in `requirements.txt` | Not in base vulnerabilities; image SBOM now has `2.14.2` | `context/requirements.txt` | `>=2.14.2` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Introduced by this image through AzureML MLflow dependencies in `requirements.txt` | Not in base vulnerabilities; image SBOM now has `1.2.1` | `context/requirements.txt` | `>=1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Base image PyTorch runtime (`ptca`) | Already present in base vulnerabilities; inherited unchanged and VCM evaluation is compliant | None | Requires a future PyTorch base image (`>=2.9.1`) |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Base image PyTorch runtime (`ptca`) | Already present in base vulnerabilities; inherited unchanged and VCM evaluation is compliant | None | Requires a future PyTorch base image (`>=2.10.0`) |

Notes:
- `mcr.microsoft.com` had no published SBOM artifact for the resolved base tag, so the base image was imported to `gatestacr.azurecr.io/sbom-base/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1` and scanned there. `base-sbom.json` and `base-vulnerabilities.json` are retained untracked for manual verification.
- The final image SBOM and findings are retained as untracked `sbom.json` and `vulnerabilities.json`. VCM evaluation reported the remediated image compliant.
- Existing pins in `context/Dockerfile` and `context/requirements.txt` were reviewed and retained because they document active transitive dependency remediation. The scan also reports `pyo3` through `rattler`; it is not one of the requested findings, is not currently policy-blocking, and is tied to the existing `conda`/`py-rattler` override.
