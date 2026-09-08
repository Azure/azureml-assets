# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-gpu:53`

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
| 6873135 / USN-8697-1 | `coreutils` | This image can carry an older package than the current base evidence; upgraded by this image's apt security layer | Base SBOM has patched Ubuntu 24.04 package `9.4-3ubuntu6.3`; target image SBOM evidence showed an older version before remediation. | `context\Dockerfile` | `9.4-3ubuntu6.3` |
| 6873134 / USN-8692-1 | `diffutils` | This image can carry an older package than the current base evidence; upgraded by this image's apt security layer | Base SBOM has patched Ubuntu 24.04 package `1:3.10-1ubuntu0.1`; target image SBOM evidence showed an older version before remediation. | `context\Dockerfile` | `1:3.10-1ubuntu0.1` |
| 6873125 / USN-8699-1 | `libssh-4` | This image can carry an older package than the current base evidence; upgraded by this image's apt security layer | Base SBOM has patched Ubuntu 24.04 package `0.10.6-2ubuntu0.5`; target image SBOM evidence showed an older version before remediation. | `context\Dockerfile` | `0.10.6-2ubuntu0.5` |
| 6873124 / USN-8691-1 | `libattr1` | This image can carry an older package than the current base evidence; upgraded by this image's apt security layer | Base SBOM has patched Ubuntu 24.04 package `1:2.5.2-1ubuntu0.1`; target image SBOM evidence showed an older version before remediation. | `context\Dockerfile` | `1:2.5.2-1ubuntu0.1` |
| 6873121 / USN-8687-1 | `libp11-kit0` | This image can carry an older package than the current base evidence; upgraded by this image's apt security layer | Base SBOM has patched Ubuntu 24.04 package `0.25.3-4ubuntu2.2`; target image SBOM evidence showed an older version before remediation. | `context\Dockerfile` | `0.25.3-4ubuntu2.2` |

Existing Dockerfile pins and overrides were audited conservatively against the current target scan and base SBOM. Active security and compatibility pins were kept; no stale pin was removed. Generated SBOM, vulnerability, and build log artifacts are retained in the working tree for manual verification but are not intended for commit.
