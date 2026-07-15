# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn:46`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1` (resolved from `{{latest-image-tag}}` on 2026-07-15).

Base SBOM note: `vcm image sbom download` found no SBOM artifact for the MCR base image, and `vcm image sbom generate` cannot attach to `mcr.microsoft.com` because the VCM attach flow requires an ACR login server. Classification below uses package install paths, the Dockerfile ownership, and the post-build VCM scan. The fixed image SBOM was generated and retained as `sbom.json`; detailed fixed-image findings were retained as `vulnerabilities.json`.

Post-build scan: VCM evaluation for `gatestacr.azurecr.io/public/azureml/curated/ai-ml-automl-dnn:test-fix` completed compliant with 0 non-compliant findings. None of the requested finding IDs were present in `vulnerabilities.json`.

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl-dnn` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl-dnn` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base-image Python prefix at `/opt/miniconda`; remediated by the Dockerfile's base-prefix security override step | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `msgpack>=1.2.1` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base-image Python prefix at `/opt/miniconda`; remediated by the Dockerfile's base-prefix security override step | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `pydantic-settings>=2.14.2` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base-image OS package upgraded in this image layer | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `1.59.0-1ubuntu0.4` via `apt-get upgrade` / `apt-get install --only-upgrade` |
| 6035933 / USN-8487-1 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base-image OS packages upgraded in this image layer | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `8.5.0-2ubuntu10.10` via `apt-get upgrade` / `apt-get install --only-upgrade` |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Base-image OS package upgraded in this image layer | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `3.45.1-1ubuntu2.6` via `apt-get upgrade` / `apt-get install --only-upgrade` |
| 6035912 / USN-8477-1 | `tar` | Base-image OS package upgraded in this image layer | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `1.35+dfsg-3ubuntu0.1` via `apt-get upgrade` / `apt-get install --only-upgrade` |
| 6035908 / USN-8456-1 | `libxml2` | Base-image OS package upgraded in this image layer | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `2.9.14+dfsg-1.3ubuntu3.8` via `apt-get upgrade` / `apt-get install --only-upgrade` |
| 6035897 / USN-8458-1 | `nginx`, `nginx-common`, `nginx-light` | Base-image OS packages upgraded in this image layer | Not verifiable from base SBOM because no base SBOM artifact was available | `context\Dockerfile` | `1.24.0-2ubuntu7.13` via `apt-get upgrade` / `apt-get install --only-upgrade` |
