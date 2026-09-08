# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl:59`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`

Base SBOM/finding files used for manual verification:

- `base-sbom.json`
- `base-vulnerabilities.json`

Remediated image SBOM/finding files used for manual verification:

- `sbom.json`
- `vulnerabilities.json`

| Finding | Package | Source classification | Base image status | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873148 / USN-8711-1 / CVE-2024-2236 | `libgcrypt20` | Inherited from base image OS packages | Base image `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1` contains vulnerable `1.10.3-2ubuntu0.1`; the rebuilt target image upgrades it through the Dockerfile apt security layer. | `context\Dockerfile` | `1.10.3-2ubuntu0.2` |

Existing Dockerfile pins and overrides were audited conservatively against the current target scan and base SBOM. Active security and compatibility pins were kept; no stale pin was removed. Generated SBOM, vulnerability, and build log artifacts are retained in the working tree for manual verification but are not intended for commit.
