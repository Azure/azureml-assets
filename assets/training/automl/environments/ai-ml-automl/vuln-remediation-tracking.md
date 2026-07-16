# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl:53`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260715.v1` (resolved from `{{latest-image-tag}}` on 2026-07-16). The MCR base image was mirrored to `vulnscan1779267129n15.azurecr.io/temp/base-openmpi5.0-ubuntu24.04:20260715.v1` so VCM could generate and attach `base-sbom.json`; detailed base findings are retained in `base-vulnerabilities.json`.

Post-build scan: VCM evaluation for `vulnscan1779267129n15.azurecr.io/public/azureml/curated/ai-ml-automl:test-fix` completed compliant with 0 non-compliant findings. The fixed image SBOM is retained as `sbom.json`, and detailed fixed-image findings are retained as `vulnerabilities.json`.

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base-image OS package | Yes; refreshed base SBOM has `1.59.0-1ubuntu0.4`, which satisfies the required version | None | No local override |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl` | N/A | `context\Dockerfile` | `torch==2.10.0` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image installs torch directly into `/azureml-envs/azureml-automl` | N/A | `context\Dockerfile` | `torch==2.10.0` |

Existing Dockerfile pins and overrides were audited against the fixed image scan and refreshed base SBOM. No stale pin from the listed findings remains: `libnghttp2-14` is already covered by the base image, and the direct `torch` install is pinned to the required patched version.
