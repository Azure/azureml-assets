# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

| Finding | Package(s) | Source | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Ubuntu package inherited from the base image | Yes; `base-sbom.json` reports `3.37.2-2ubuntu0.6` | `context/Dockerfile` | Explicit upgrade guard keeps `3.37.2-2ubuntu0.6` or newer |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Ubuntu packages inherited from the base image | Yes; `base-sbom.json` reports `7.81.0-1ubuntu1.25` | `context/Dockerfile` | Explicit upgrade guard keeps `7.81.0-1ubuntu1.25` or newer |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Ubuntu package inherited from the base image | Yes; `base-sbom.json` reports `1.43.0-1ubuntu0.4` | `context/Dockerfile` | Explicit upgrade guard keeps `1.43.0-1ubuntu0.4` or newer |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base env dependency; can be re-resolved by this image's AzureML/MLflow dependency layer | Yes; `base-sbom.json` reports `1.2.1` | `context/Dockerfile` | `msgpack>=1.2.1`; `sbom.json` reports `1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Base image `ptca` PyTorch runtime | No; `base-vulnerabilities.json` reports `torch 2.8.0+cu126` | `context/Dockerfile` | `torch==2.10.0`; `sbom.json` reports `2.10.0+cu126` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Base image `ptca` PyTorch runtime | No; `base-vulnerabilities.json` reports `torch 2.8.0+cu126` | `context/Dockerfile` | `torch==2.10.0`; `sbom.json` reports `2.10.0+cu126` |

Notes:
- Direct SBOM download for the MCR base image found no SBOM referrers. The matching resolved ACPT base SBOM for digest `sha256:2c139ba03468a302ef9c11507dfd064ecd7ee4bd993cb64670bbe72db69b10ff` was copied here as `base-sbom.json` and `base-vulnerabilities.json` for manual verification.
- Existing pins in `context/Dockerfile` and `context/requirements.txt` were audited against the current scan and base SBOM. The stale `pip==26.1.1` conda override was removed because the resolved base already ships `pip 26.1.2` and no pip finding remains.
- The final image scan for `vulnscan1779267129n9.azurecr.io/public/azureml/curated/acpt-automl-image-framework-selector-gpu:test-fix` is compliant with zero non-compliant findings.
