# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-text-gpu-ptca:54`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`

Base SBOM/finding files used for manual verification:

- `base-sbom.json`
- `base-vulnerabilities.json`

Remediated image SBOM/finding files used for manual verification:

- `sbom.json`
- `vulnerabilities.json`

| Finding | Package | Source classification | Base image status | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873121 / USN-8687-1 | `libp11-kit0` | Base image OS package | Already covered in the resolved Ubuntu 24.04 base at `0.25.3-4ubuntu2.2`, above the required Ubuntu 22.04 version. | None | `0.25.3-4ubuntu2.2` |
| 6873124 / USN-8691-1 | `libattr1` | Base image OS package | Already covered in the resolved Ubuntu 24.04 base at `1:2.5.2-1ubuntu0.1`, above the required Ubuntu 22.04 version. | None | `1:2.5.2-1ubuntu0.1` |
| 6873125 / USN-8699-1 | `libssh-4` | Base image OS package | Already covered in the resolved Ubuntu 24.04 base at `0.10.6-2ubuntu0.5`, above the required Ubuntu 22.04 version. | None | `0.10.6-2ubuntu0.5` |
| 6873134 / USN-8692-1 | `diffutils` | Base image OS package | Already covered in the resolved Ubuntu 24.04 base at `1:3.10-1ubuntu0.1`, above the required Ubuntu 22.04 version. | None | `1:3.10-1ubuntu0.1` |
| 6873135 / USN-8697-1 | `coreutils` | Base image OS package | Already covered in the resolved Ubuntu 24.04 base at `9.4-3ubuntu6.3`, above the required Ubuntu 22.04 version. | None | `9.4-3ubuntu6.3` |
| 6873148 / USN-8711-1 | `libgcrypt20` | Base image OS package | Base was already above the required Ubuntu 22.04 version; the rebuilt image upgrades to `1.10.3-2ubuntu0.2` through the existing apt security layer. | `context\Dockerfile` | `1.10.3-2ubuntu0.2` |
| 6873149 / USN-8710-1 | `libevent-core-2.1-7` | Not present in the resolved base or rebuilt image SBOM | No package instance to remediate in this image. | None | Not installed |
| 5017847 / GHSA-xrqw-3rrv-vx5w | `transformers` | PTCA Python environment package when `/opt/conda/envs/ptca` is present | Not present in the resolved base or rebuilt image SBOM, but pinned in the conditional PTCA patch block for image variants that include that environment. | `context\Dockerfile` | `5.10.0` |

Existing Dockerfile pins and overrides were audited conservatively. Active security and compatibility pins were kept; no generated SBOM or build log artifacts should be committed.
