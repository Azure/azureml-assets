# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

| Finding | Package(s) | Source | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014607 / GHSA-5jmj-h7xm-6q6v | `com.fasterxml.jackson.core:jackson-databind` in Ray fat JAR | Introduced by this image through `ray` in the ptca Python environment | Not present in base SBOM findings | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.18.8` |
| 5014608 / GHSA-hgj6-7826-r7m5 | `com.fasterxml.jackson.core:jackson-databind` in Ray fat JAR | Introduced by this image through `ray` in the ptca Python environment | Not present in base SBOM findings | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.18.8` |
| 5014600 / GHSA-rmj7-2vxq-3g9f | `com.fasterxml.jackson.core:jackson-databind` in Ray fat JAR | Introduced by this image through `ray` in the ptca Python environment | Not present in base SBOM findings | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.18.8` |
| 5014605 / GHSA-j3rv-43j4-c7qm | `com.fasterxml.jackson.core:jackson-databind` in Ray fat JAR | Introduced by this image through `ray` in the ptca Python environment | Not present in base SBOM findings | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.18.8` |
| 5015223 / GHSA-wf93-45jw-7689 | `pip` | Bootstrap package in conda environments; base SBOM has patched `26.1.2`, but this Dockerfile retained a direct floor for scanner-stable conda metadata cleanup | Yes, base SBOM shows `26.1.2` | `context/Dockerfile` | `>=26.1.2` |
| 6035952 / USN-8512-1 | `gzip` | Ubuntu package inherited from the base image | Yes, base SBOM shows `1.10-4ubuntu4.2` | None | `1.10-4ubuntu4.2` |
| 6035957 / USN-8510-1 | `tar` | Ubuntu package inherited from the base image | Yes, base SBOM shows `1.34+dfsg-1ubuntu0.1.22.04.4` | None | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035961 / USN-8509-1 | `libpython3.10-stdlib`, `python3.10`, `python3.10-minimal`, `libpython3.10-minimal` | Ubuntu packages inherited from the base image | Yes, base SBOM shows `3.10.12-1~22.04.16` | None | `3.10.12-1~22.04.16` |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Ubuntu package inherited from the base image | Yes, base SBOM shows `3.37.2-2ubuntu0.6` | None | `3.37.2-2ubuntu0.6` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Ubuntu packages inherited from the base image | Yes, base SBOM shows `7.81.0-1ubuntu1.25` | None | `7.81.0-1ubuntu1.25` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Ubuntu package inherited from the base image | Yes, base SBOM shows `1.43.0-1ubuntu0.4` | None | `1.43.0-1ubuntu0.4` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Python package inherited from the base conda environment | Yes, base SBOM shows `1.2.1` | None | `1.2.1` |

Notes:
- `mcr.microsoft.com` did not expose an SBOM referrer for the resolved base tag, so `base-sbom.json` and `base-vulnerabilities.json` were copied from the existing scan artifacts for the identical base image tag under `assets/training/general/environments/acpt-pytorch-2.8-cuda12.6`.
- `sbom.json` and `vulnerabilities.json` are retained after the final image scan for manual verification. VCM evaluation for `vulnscan1779267129n2.azurecr.io/public/azureml/curated/acft-group-relative-policy-optimization:test-fix` reported compliant with 0 non-compliant findings.
- Existing pins and overrides in the Dockerfile were reviewed. The stale pip floor was raised from `>=26.1.1` to `>=26.1.2`; other retained pins still document active transitive dependency remediations for packages installed by this image.
