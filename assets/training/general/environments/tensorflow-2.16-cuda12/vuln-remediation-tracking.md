# Vulnerability remediation tracking

Image: `public/azureml/curated/tensorflow-2.16-cuda12:30`

Base image: `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`

Base SBOM status: `vcm image sbom download --registry nvcr.io --repository nvidia/cuda --tag 12.8.1-cudnn-devel-ubuntu22.04 --output base-sbom.json` failed with `401 Unauthorized`, so base package ownership could not be fully verified from the SBOM. The remediations below only affect Ubuntu packages installed or upgraded in this image's Dockerfile.

Final image SBOM status: `sbom.json` was generated for `vulnscan1779267129n10.azurecr.io/public/azureml/curated/tensorflow-2.16-cuda12:test-fix` and `vcm image vulnerabilities evaluate --sbom sbom.json` returned compliant with 0 non-compliant findings.

| Finding | Package(s) | Source classification | Covered by base image? | Changed file(s) | Patched version |
| --- | --- | --- | --- | --- | --- |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Introduced or overridden by this image through Ubuntu package installs (`curl`/`libcurl` dependency) | Unknown; base SBOM unavailable | `context/Dockerfile` | `1.43.0-1ubuntu0.4` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | `curl` and `libcurl4` are installed directly by this image; `libcurl3-gnutls` may be inherited or pulled by Ubuntu package dependencies | Unknown; base SBOM unavailable | `context/Dockerfile` | `7.81.0-1ubuntu1.25` |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Inherited from the Ubuntu/CUDA base or Ubuntu package dependencies, upgraded by this image | Unknown; base SBOM unavailable | `context/Dockerfile` | `3.37.2-2ubuntu0.6` |
| 6035912 / USN-8477-1 | `tar` | Inherited from the Ubuntu/CUDA base, upgraded by this image | Unknown; base SBOM unavailable | `context/Dockerfile` | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035908 / USN-8456-1 | `libxml2` | Introduced or overridden by this image through Ubuntu package installs (`libxml++2.6-2v5` dependency) | Unknown; base SBOM unavailable | `context/Dockerfile` | `2.9.13+dfsg-1ubuntu0.12` |
| 6035897 / USN-8458-1 | `nginx-light`, `libnginx-mod-http-geoip2`, `libnginx-mod-http-echo`, `nginx-common` | Installed directly by this image's inferencing layer | Unknown; base SBOM unavailable | `context/Dockerfile` | `1.18.0-6ubuntu14.16` |

Existing pins and overrides audited:

| File | Pin or override | Action |
| --- | --- | --- |
| `context/Dockerfile` | Exact nginx package pins at `1.18.0-6ubuntu14.15` | Removed stale exact pins and replaced with unpinned `--only-upgrade` entries so current Ubuntu security versions are selected. |
| `context/Dockerfile` | Python package CVE override block (`pip`, `cryptography`, `setuptools`, `requests`, `pillow`, `starlette`, `idna`, `PyJWT`, `pyarrow`) | Left unchanged; unrelated to this Ubuntu OS package remediation and not present in the supplied findings. |
| `context/conda_dependencies.yaml` | Python package CVE pins (`pip`, `setuptools`, `keras`, `idna`, `pillow`) | Left unchanged; unrelated to this Ubuntu OS package remediation and not present in the supplied findings. |
