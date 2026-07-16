# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-text-gpu:52`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260715.v1` (resolved from `{{latest-image-tag}}` on 2026-07-16). The MCR base image had no downloadable SBOM referrer, so it was mirrored to `vulnscan1779267129n11.azurecr.io/temp/base-openmpi5.0-ubuntu24.04:20260715.v1` and VCM generated `base-sbom.json` and `base-vulnerabilities.json` from that mirror.

Post-build scan: VCM evaluation for `vulnscan1779267129n11.azurecr.io/public/azureml/curated/ai-ml-automl-dnn-text-gpu:test-fix` completed compliant with 0 non-compliant findings. The fixed image SBOM is retained as `sbom.json`, and detailed fixed-image findings are retained as `vulnerabilities.json`.

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035952 / USN-8512-1 | `gzip` | Base-image OS package | No; base SBOM has `1.12-1ubuntu3.1` | `context\Dockerfile` | upgraded by `apt-get upgrade` to `1.12-1ubuntu3.2` |
| 6035897 / USN-8458-1 | `nginx`, `nginx-common`, `nginx-light`, `libnginx-mod-http-echo` | Base-image OS packages | Yes; base SBOM has Ubuntu 24.04 nginx packages at or above the Ubuntu 22.04 fixed versions from the finding | None | No local override |
| 6035908 / USN-8456-1 | `libxml2` | Base-image OS package | Yes; base SBOM has `2.9.14+dfsg-1.3ubuntu3.8` | None | No local override |
| 6035912 / USN-8477-1 | `tar` | Base-image OS package | Yes; base SBOM has `1.35+dfsg-3ubuntu0.2` | None | No local override |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Base-image OS package | Yes; base SBOM has `3.45.1-1ubuntu2.6` | None | No local override |
| 6035933 / USN-8487-1 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base-image OS packages | The older Ubuntu 22.04 packages from the finding are not present; current Ubuntu 24.04 base curl packages are upgraded by this Dockerfile | `context\Dockerfile` | `8.5.0-2ubuntu10.11` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base-image OS package | Yes; base SBOM has `1.59.0-1ubuntu0.4` | None | No local override |
| 5013832 / GHSA-537c-gmf6-5ccf | `cryptography` | Base-image Python prefix and AutoML environment transitive dependency | Base SBOM has `cryptography 49.0.0`; image also resolves `49.0.0` | `context\Dockerfile` | `cryptography>=48.0.1` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base-image Python prefix at `/opt/miniconda` | Yes; base SBOM has `pydantic-settings 2.14.2` | None | No local override |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base-image Python prefix at `/opt/miniconda` | Yes; base SBOM has `msgpack 1.2.1` | None | No local override |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 6035961 / USN-8509-1 | `python3.10`, `libpython3.10-stdlib`, `python3.10-minimal`, `libpython3.10-minimal` | Not present in the resolved Ubuntu 24.04 base or fixed image SBOM as dpkg packages | N/A | None | No local OS package override |
| 6035957 / USN-8510-1 | `tar` | Base-image OS package | Yes; base SBOM has `1.35+dfsg-3ubuntu0.2` | None | No local override |

Existing Dockerfile pins and overrides were audited against the fixed image scan and generated base SBOM. Redundant `msgpack` and `pydantic-settings` transitive pins are not present because the resolved base image already ships patched versions. Other pre-existing security overrides were kept because they address separate findings or base-prefix packages outside this request.
