# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-draft-model-training:17`

Base image from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}`

Resolved base tag used for verification: `biweekly.202607.1`.

VCM could not download an SBOM referrer directly from MCR, so the base image was mirrored into `vulnscan1779267129n10.azurecr.io/sbom-base/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1` for base SBOM generation and comparison.

| Vulnerability | Package(s) | Source | Base already covered? | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035908 / USN-8456-1 | libxml2 | base-image inherited Ubuntu package | Yes, resolved base `biweekly.202607.1` has `2.9.13+dfsg-1ubuntu0.12` | None | `2.9.13+dfsg-1ubuntu0.12` |
| 6035912 / USN-8477-1 | tar | base-image inherited Ubuntu package | Yes, resolved base `biweekly.202607.1` has `1.34+dfsg-1ubuntu0.1.22.04.4` | None | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035919 / USN-8480-1 | libsqlite3-0 | base-image inherited Ubuntu package | Yes, resolved base `biweekly.202607.1` has `3.37.2-2ubuntu0.6` | None | `3.37.2-2ubuntu0.6` |
| 6035933 / USN-8487-1 | curl, libcurl3-gnutls, libcurl4 | base-image inherited Ubuntu packages | Yes, resolved base `biweekly.202607.1` has `7.81.0-1ubuntu1.25` | None | `7.81.0-1ubuntu1.25` |
| 6035947 / USN-8495-1 | libnghttp2-14 | base-image inherited Ubuntu package | Yes, resolved base `biweekly.202607.1` has `1.43.0-1ubuntu0.4` | None | `1.43.0-1ubuntu0.4` |
| 5014304 / GHSA-6v7p-g79w-8964 | msgpack | Python transitive dependency in the base Python 3.13 conda env, also reachable through MLflow/AzureML MLflow layers | No, current image finding reported `1.1.1` under `/opt/conda/lib/python3.13/site-packages` | `context/Dockerfile` (existing pin retained) | `>=1.2.1` |

Verification artifacts kept in this directory for reviewer comparison:

- `base-sbom.json`
- `base-vulnerabilities.json`
- `sbom.json`
- `vulnerabilities.json`

Verification result: rebuilt image `vulnscan1779267129n10.azurecr.io/public/azureml/curated/acft-draft-model-training:17@sha256:350c96155accae96732d0f463673e55cb9adcc06638635ca1a6122e203bc5c77` evaluated as compliant. The listed vulnerabilities were not present in `vulnerabilities.json`.
