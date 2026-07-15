# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-forecasting-gpu:47`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1` (resolved from `{{latest-image-tag}}` on 2026-07-15). The MCR base image had no downloadable SBOM referrer, so it was mirrored to `gatestacr.azurecr.io/temp/base-openmpi5.0-ubuntu24.04:20260707.v1` and VCM generated `base-sbom.json` and `base-vulnerabilities.json` from that mirror.

Post-build scan: VCM evaluation for `gatestacr.azurecr.io/public/azureml/curated/ai-ml-automl-dnn-forecasting-gpu:test-fix` completed compliant with 0 non-compliant findings. The fixed image SBOM is retained as `sbom.json`, and detailed fixed-image findings are retained as `vulnerabilities.json`.

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base-image Python prefix at `/opt/miniconda` | Yes; base SBOM has `msgpack 1.2.1` | None | No local override |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base-image Python prefix at `/opt/miniconda` | Yes; base SBOM has `pydantic-settings 2.14.2` | None | No local override |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base-image OS package | Yes; base SBOM has `1.59.0-1ubuntu0.4` | None | No local override |
| 6035933 / USN-8487-1 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base-image OS packages | Yes; fixed image SBOM has `8.5.0-2ubuntu10.11` | None | No local override for USN-8487-1 |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Base-image OS package | Yes; base SBOM has `3.45.1-1ubuntu2.6` | None | No local override |
| 6035912 / USN-8477-1 | `tar` | Base-image OS package | Yes; base SBOM has `1.35+dfsg-3ubuntu0.2` | None | No local override |
| 6035908 / USN-8456-1 | `libxml2` | Base-image OS package | Yes; base SBOM has `2.9.14+dfsg-1.3ubuntu3.8` | None | No local override |
| 6035897 / USN-8458-1 | `nginx`, `nginx-common`, `nginx-light` | Base-image OS packages | Yes; base SBOM has `1.24.0-2ubuntu7.13` | None | No local override |

Existing Dockerfile pins and overrides were audited against the fixed image scan and generated base SBOM. Redundant `msgpack` and `pydantic-settings` transitive pins were removed because the resolved base image already ships patched versions. Other pre-existing security overrides were kept because they address separate findings or base-prefix packages outside this request.
